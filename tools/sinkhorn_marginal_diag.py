"""
Standalone diagnostic for the bug "two-sided beats sinkhorn at low tau".

Hypothesis under test: at small tau (0.02-0.05) with T=20 + bf16, our
log-domain Sinkhorn does NOT actually converge to a doubly-stochastic plan.
If the col-marginal of the plan deviates substantially from uniform, then
the resulting "sinkhorn" drift is effectively a weakened row-softmax, and
the two-sided baseline (which is exactly a sym-row-softmax) can beat it.

What this script does:
  - Generates random features matching the DINOv2 stream's shape used in
    actual training: (G, Nf, Nr, D) with normalize_feature semantics.
  - For each (tau, precision, T) cell, builds the cost matrix the SAME way
    compute_drift_split does (l2_sq normalized to mean 1), then runs the
    EXACT _sinkhorn_plan / _two_sided_plan implementations.
  - Reports BEFORE the final row-renormalize:
      * row-sum stats (should be ~1 at log-domain Sinkhorn fixed point)
      * col-sum stats (target = Nf/Nr if doubly stochastic; deviation
        quantifies how far T iters got us from convergence)
  - Reports AFTER the final row-renorm (what train_unc.py actually uses):
      * the same row/col stats so we can quantify the residual mismatch
        in the plan that ends up in V = P @ y.

Invocation:
    python tools/sinkhorn_marginal_diag.py
No GPU required (runs in seconds on CPU). DDP is bypassed: the
col_logsumexp_ddp helper degrades to torch.logsumexp(dim=-2) when no
process group is initialized, which is exactly the single-rank case.
"""

import math
import torch

# Make sure we import THE files used by the running training, not a copy.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drifting_split import _two_sided_plan, _sinkhorn_plan


def _build_cost(x_fake, y_pos):
    """Replicate compute_drift_split's normalize_feature -> -d^2/eps prep."""
    d = torch.cdist(x_fake, y_pos)
    dist_scale = d.mean().clamp_min(1e-3)
    d = d / dist_scale
    return d


def _summarize_marginals(P, target_col):
    """P is (G, Nf, Nr). Returns dict of row/col stats across all groups."""
    row = P.sum(dim=-1)   # (G, Nf), should be ~1
    col = P.sum(dim=-2)   # (G, Nr), target ~ Nf/Nr if doubly-stoch.
    return dict(
        row_mean=row.mean().item(), row_std=row.std().item(),
        row_min=row.min().item(), row_max=row.max().item(),
        col_target=target_col,
        col_mean=col.mean().item(), col_std=col.std().item(),
        col_min=col.min().item(), col_max=col.max().item(),
    )


def _print_row(label, stats):
    print(f"  {label:30s}  row=[{stats['row_min']:.4f}, {stats['row_max']:.4f}] "
          f"mean={stats['row_mean']:.4f} std={stats['row_std']:.4f}  "
          f"col target={stats['col_target']:.4f}  range=[{stats['col_min']:.4f}, "
          f"{stats['col_max']:.4f}] mean={stats['col_mean']:.4f} std={stats['col_std']:.4f}")


def main():
    torch.manual_seed(0)
    # Match a per-rank DINOv2-norm slice of B=2048 / 4 ranks: Nf=Nr=512, D=768
    # Single rank diagnostic so col_logsumexp_ddp falls back to local logsumexp.
    G, Nf, Nr, D = 3, 512, 512, 768
    print(f"shape: G={G}, Nf={Nf}, Nr={Nr}, D={D} (single-rank, no DDP)")

    x_fake = torch.randn(G, Nf, D)
    y_pos = torch.randn(G, Nr, D)
    d = _build_cost(x_fake, y_pos)             # (G, Nf, Nr) with mean ~1
    print(f"raw cost: d.mean={d.mean():.4f}, d.std={d.std():.4f}")

    target_col = float(Nf) / float(Nr)         # = 1 here (square matrix)

    for tau in [0.5, 0.1, 0.05, 0.02]:
        print(f"\n=== tau = {tau} ===")
        for prec_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            d_t = d.to(dtype)
            logit_base = -d_t.pow(2) / tau     # like compute_drift_split's l2_sq path

            # two-sided (mutates input)
            P_ts = _two_sided_plan(logit_base.clone())
            stats = _summarize_marginals(P_ts.float(), target_col)
            _print_row(f"two-sided ({prec_name})", stats)

            for T in [20, 50, 100, 200]:
                P = _sinkhorn_plan(logit_base.clone(), iters=T)
                stats = _summarize_marginals(P.float(), target_col)
                _print_row(f"sinkhorn T={T:>3d} ({prec_name})", stats)

    print("\nKey reading:")
    print("- two-sided is row-stochastic by construction; its col target")
    print("  deviation tells you nothing about Sinkhorn convergence.")
    print("- Sinkhorn quality = how close col_mean and col_std are to (1.0, 0).")
    print("  If at tau=0.02, T=20, bf16 the col_std is large but at T=200, fp32")
    print("  it's small, that's the smoking gun for 'didn't converge'.")


if __name__ == "__main__":
    main()
