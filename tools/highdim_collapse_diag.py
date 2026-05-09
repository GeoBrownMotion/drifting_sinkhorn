"""
Diagnostic: at what (dim, eps, normalize_feature) does two-sided collapse?

Toy (2D, l2_sq, no normalize, eps=0.01) shows two-sided emd2=7.33 (random),
sinkhorn emd2=0.33 (learned). The collapse mechanism: at sharp eps and wide
distance distribution, A = sqrt(softmax_row * softmax_col) becomes very
sparse (or zero rows) because row/col argmax positions disagree.

In our cifar setting (768D DINOv2, normalize_feature=True), distances are
tight (CV~2.5%), so two-sided never collapses and the Sinkhorn advantage
disappears.

This script tests: in 768D, does any (eps, normalize_off) combo recreate
the "two-sided sparse / V=0" regime?

Reports per cell:
  - mean ||V_xy|| for two-sided and sinkhorn (V=0 = useless drift)
  - row entropy of P_xy (low entropy = sharp / collapsed)
  - fraction of "zero rows" of two-sided plan after row-renorm
    (close to 0 = healthy, large = collapse)
"""
import math
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drifting_split import _two_sided_plan, _sinkhorn_plan


def _build_cost(x_fake, y_pos, normalize):
    d = torch.cdist(x_fake, y_pos)
    if normalize:
        d = d / d.mean().clamp_min(1e-3)
    return d


def _row_entropy_mean(P):
    P = P.clamp_min(1e-30)
    return -(P * P.log()).sum(dim=-1).mean().item()


def _row_max_mean(P):
    return P.max(dim=-1).values.mean().item()


def _zero_row_frac_pre_renorm(logit, plan_kind, **kw):
    """Run the plan, then count rows whose pre-renorm sum is below floor."""
    from drifting_split import _two_sided_plan as ts, _sinkhorn_plan as sk
    if plan_kind == "two-sided":
        # Replicate the in-place math but stop before the final row-renorm so
        # we can report row mass.
        lse_row = torch.logsumexp(logit, dim=-1, keepdim=True)
        lse_col = torch.logsumexp(logit, dim=-2, keepdim=True)
        log_a = logit - 0.5 * (lse_row + lse_col)
        A = log_a.exp()
        row_sum = A.sum(dim=-1)
        floor = 1e-6
        frac = (row_sum < floor).float().mean().item()
        median = row_sum.median().item()
        return frac, median
    return 0.0, 1.0


def main():
    torch.manual_seed(0)
    print(f"{'dim':>4} {'norm':>5} {'eps':>8}  {'plan':>10}   {'H_row':>7}  {'max_row':>8}  {'||V||':>8}  {'zero_rows%':>10}  {'rs_med':>8}")
    print("-" * 105)

    G, N = 3, 512
    for D in [2, 64, 768]:
        x_fake = torch.randn(G, N, D)
        y_pos = torch.randn(G, N, D)
        for normalize in [True, False]:
            d = _build_cost(x_fake, y_pos, normalize)
            d_mean = d.mean().item()
            d_std = d.std().item()
            for eps in [0.5, 0.1, 0.02, 0.005, 0.001]:
                logit_base = (-d.pow(2) / eps).to(torch.bfloat16)

                # two-sided
                lp = logit_base.clone()
                P_ts = _two_sided_plan(lp).float()
                # also pre-renorm row-mass diagnostic
                frac_zero, rs_med = _zero_row_frac_pre_renorm(logit_base.clone().float(), "two-sided")
                H_ts = _row_entropy_mean(P_ts)
                M_ts = _row_max_mean(P_ts)
                V_ts = (P_ts @ y_pos).norm(dim=-1).mean().item()

                # sinkhorn
                ls = logit_base.clone()
                P_sh = _sinkhorn_plan(ls, iters=20).float()
                H_sh = _row_entropy_mean(P_sh)
                M_sh = _row_max_mean(P_sh)
                V_sh = (P_sh @ y_pos).norm(dim=-1).mean().item()

                norm_tag = "yes" if normalize else "no"
                print(f"{D:>4} {norm_tag:>5} {eps:>8.4f}  {'two-sided':>10}   {H_ts:>7.3f}  {M_ts:>8.4f}  {V_ts:>8.4f}  {frac_zero*100:>9.2f}%  {rs_med:>8.2e}")
                print(f"{D:>4} {norm_tag:>5} {eps:>8.4f}  {'sinkhorn':>10}   {H_sh:>7.3f}  {M_sh:>8.4f}  {V_sh:>8.4f}  {'-':>10}  {'-':>8}")
            print()


if __name__ == "__main__":
    main()
