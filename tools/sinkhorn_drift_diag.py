"""
Follow-up diagnostic: WHY does two-sided beat sinkhorn at low tau, given that
our sinkhorn implementation IS converging (cf. sinkhorn_marginal_diag.py)?

Hypotheses we're testing here:

  H3 (self-mask asymmetry): two-sided masks the diagonal of P_xx (no fake
      pulls itself toward itself). Sinkhorn relies on doubly-stochastic
      to bound self-coupling at ~1/Nf. At low tau the diagonal of the
      negative-branch plan is sharply on the diagonal, so 1/Nf becomes
      a meaningful self-pull that contaminates V_neg.

  H4 (entropy mismatch): "same tau" comparisons are unfair if the two
      plans have wildly different per-row entropy. Two-sided's row =
      sqrt(softmax_row * softmax_col)/renorm; sinkhorn's row = T-step
      log-domain projection then row-renorm. These can have different
      effective sharpness at the same nominal tau.

We compute on the same random features:
  - per-row entropy H(row) for P_xy under each plan, all taus
  - per-row max-mass max_j P_ij (sharpness proxy)
  - the diagonal mass of P_xx for each plan, no-mask vs with-mask
    (this directly quantifies self-coupling contamination)
  - the *positive-branch* drift V_xy = P_xy @ y_pos under each plan
    and report ||V_ts||, ||V_sh||, and cos(V_ts, V_sh)
  - the *negative-branch* drift V_xx = P_xx @ y_neg under each plan
    in two configs: (a) no_mask (our current sinkhorn default) and
    (b) with_mask (our two-sided default). Report self-pull magnitude
    diff between (a) and (b) for the sinkhorn case to quantify how much
    the missing mask is costing us.
"""

import math
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drifting_split import _two_sided_plan, _sinkhorn_plan


def _build_cost(x_fake, y_pos):
    d = torch.cdist(x_fake, y_pos)
    dist_scale = d.mean().clamp_min(1e-3)
    return d / dist_scale


def _row_entropy(P):
    """H(row_i) = -sum_j P_ij log P_ij, averaged over rows + groups."""
    eps = 1e-30
    H = -(P * (P.clamp_min(eps)).log()).sum(dim=-1)   # (G, Nf)
    return H.mean().item(), H.std().item(), H.min().item(), H.max().item()


def _row_max(P):
    """max_j P_ij (sharpness proxy)."""
    M = P.max(dim=-1).values  # (G, Nf)
    return M.mean().item(), M.std().item()


def _diagonal_mass(P):
    """For square (G, N, N), avg of P[g, i, i] = self-coupling probability."""
    G, N, _ = P.shape
    diag = torch.diagonal(P, dim1=-2, dim2=-1)  # (G, N)
    return diag.mean().item(), diag.max().item()


def _cos_drift(V_a, V_b):
    """Mean cosine similarity per row across all groups."""
    a_n = V_a / V_a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b_n = V_b / V_b.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (a_n * b_n).sum(dim=-1).mean().item()


def _norm(V):
    return V.norm(dim=-1).mean().item()


def _apply_self_mask(logit):
    """In-place: set diagonal of (G, N, N) to -inf."""
    G, N, _ = logit.shape
    idx = torch.arange(N, device=logit.device)
    logit[:, idx, idx] = float("-inf")
    return logit


def main():
    torch.manual_seed(0)
    G, N, D = 3, 512, 768            # square matrix; same shape for P_xy and P_xx
    print(f"shape: G={G}, N={N}, D={D} (single-rank)")

    # Independent x_fake / y_real (positive branch); for the NEGATIVE branch
    # we use y_neg = x_fake (because in train_unc.py negatives = all fakes,
    # and per-rank that's just the local fakes when world_size=1).
    x_fake = torch.randn(G, N, D)
    y_pos = torch.randn(G, N, D)
    y_neg = x_fake.clone()

    d_pos = _build_cost(x_fake, y_pos)
    d_neg = _build_cost(x_fake, y_neg)            # diagonal is 0 by construction

    print(f"d_pos.mean={d_pos.mean():.4f}, d_neg.mean={d_neg.mean():.4f}")
    print(f"d_neg diag = self-distances: {torch.diagonal(d_neg, dim1=-2, dim2=-1).mean():.6f} (should be 0)\n")

    for tau in [0.5, 0.1, 0.05, 0.02]:
        print(f"\n=== tau = {tau} ===")

        # --- POSITIVE BRANCH P_xy
        logit_pos = (-d_pos.pow(2) / tau).to(torch.bfloat16)

        P_ts_xy = _two_sided_plan(logit_pos.clone()).float()
        P_sh_xy = _sinkhorn_plan(logit_pos.clone(), iters=20).float()

        Hts = _row_entropy(P_ts_xy); Hsh = _row_entropy(P_sh_xy)
        Mts = _row_max(P_ts_xy);     Msh = _row_max(P_sh_xy)
        V_ts_xy = P_ts_xy @ y_pos    # (G, N, D)
        V_sh_xy = P_sh_xy @ y_pos

        print(f"  P_xy two-sided  H_row mean={Hts[0]:.3f}  max_row mean={Mts[0]:.3f}  ||V||={_norm(V_ts_xy):.4f}")
        print(f"  P_xy sinkhorn   H_row mean={Hsh[0]:.3f}  max_row mean={Msh[0]:.3f}  ||V||={_norm(V_sh_xy):.4f}")
        print(f"  P_xy V_ts vs V_sh  cosine={_cos_drift(V_ts_xy, V_sh_xy):.4f}")

        # --- NEGATIVE BRANCH P_xx, vary self-mask
        logit_neg = (-d_neg.pow(2) / tau).to(torch.bfloat16)

        # two-sided uses self-mask by default in our pipeline
        P_ts_xx_mask = _two_sided_plan(_apply_self_mask(logit_neg.clone())).float()
        # sinkhorn uses NO self-mask by default
        P_sh_xx_no   = _sinkhorn_plan(logit_neg.clone(), iters=20).float()
        # to test H3 we also build sinkhorn WITH self-mask
        P_sh_xx_mask = _sinkhorn_plan(_apply_self_mask(logit_neg.clone()), iters=20).float()
        # and two-sided WITHOUT mask for symmetric comparison
        P_ts_xx_no   = _two_sided_plan(logit_neg.clone()).float()

        d_ts_mask = _diagonal_mass(P_ts_xx_mask)   # should be 0 (or near 0)
        d_ts_no   = _diagonal_mass(P_ts_xx_no)
        d_sh_no   = _diagonal_mass(P_sh_xx_no)
        d_sh_mask = _diagonal_mass(P_sh_xx_mask)

        print(f"  P_xx self-coupling mass (mean, max):")
        print(f"     two-sided + self-mask:  ({d_ts_mask[0]:.6f}, {d_ts_mask[1]:.6f})  <- our default for two-sided")
        print(f"     two-sided  no    mask:  ({d_ts_no[0]:.6f}, {d_ts_no[1]:.6f})")
        print(f"     sinkhorn   no    mask:  ({d_sh_no[0]:.6f}, {d_sh_no[1]:.6f})  <- our default for sinkhorn (THE LEAK)")
        print(f"     sinkhorn  + self-mask:  ({d_sh_mask[0]:.6f}, {d_sh_mask[1]:.6f})")

        # --- NEGATIVE BRANCH drift; what would V_neg be?
        V_ts_xx_mask = P_ts_xx_mask @ y_neg
        V_sh_xx_no   = P_sh_xx_no @ y_neg
        V_sh_xx_mask = P_sh_xx_mask @ y_neg

        # If self-coupling pulls fake toward itself, V_neg will be biased toward y_neg = x_fake.
        # Quantify: cos(V_neg, x_fake). The bigger this is, the more the negative branch is
        # "pulling fake toward fake" (which subtracts from the drift signal).
        cos_self_ts_mask = _cos_drift(V_ts_xx_mask, x_fake)
        cos_self_sh_no   = _cos_drift(V_sh_xx_no,   x_fake)
        cos_self_sh_mask = _cos_drift(V_sh_xx_mask, x_fake)
        print(f"  cos(V_xx, x_fake)  (high = bad: drift pulls fake toward itself):")
        print(f"     two-sided + self-mask:  {cos_self_ts_mask:+.4f}")
        print(f"     sinkhorn   no    mask:  {cos_self_sh_no:+.4f}  <- contamination if much > the masked one")
        print(f"     sinkhorn  + self-mask:  {cos_self_sh_mask:+.4f}")


if __name__ == "__main__":
    main()
