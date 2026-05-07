"""
Split-form drifting field for CIFAR-10 Sinkhorn-Drifting experiments.

Implements V(x) = P_xy @ y_real - P_xx @ y_neg in the FFHQ-style barycentric
drift framework, with three coupling choices:

  - "two-sided": A = sqrt(softmax_row(K) * softmax_col(K)), row-renormalized.
                 P_xx is built with self-distance masked, matching the
                 original Drifting paper Algorithm 2 self-mask.
  - "sinkhorn":  T-step log-domain Sinkhorn iterations, row-renormalized.
                 P_xx has NO self-mask, relying on the doubly-stochastic
                 marginal constraint to prevent self-coupling collapse.
  - "one-sided": row-softmax only, also with self-mask on P_xx.

Single epsilon (no multi-temperature aggregation), Gaussian kernel by default
(l2_sq), and feature-magnitude normalization to match drifting-models-pytorch's
data-scale convention so the outer MSE loss has a comparable magnitude.

DDP convention:
  - x_fake (and the corresponding rows of all plans) are kept LOCAL.
  - x_real and the gathered fakes used as the column space of plans are GLOBAL.
  - Plan shapes: P_xy is (G, Nf_local, Nr_global), P_xx is (G, Nf_local, Nf_global).
  - Two-sided / Sinkhorn col-reductions cross ranks via col_logsumexp_ddp.

Memory: all large intermediate tensors are mutated in-place where possible,
and P_xy / P_xx are NOT kept alive simultaneously (the positive matmul runs
before the negative plan is built). At B=2048 with the multi-scale DINOv2
streams this drops compute_drift_split's peak overhead from ~12 GB to ~5 GB.
"""

import math

import torch
from torch import Tensor

from utils.distributed import (
    gather_tensor,
    get_rank,
    is_dist_avail_and_initialized,
    reduce_tensor,
)


def col_logsumexp_ddp(logit: Tensor, keepdim: bool = False) -> Tensor:
    """logsumexp over dim=-2 where the row dimension is sharded across DDP ranks.

    Memory note: the original implementation `torch.exp(logit - col_max).sum(...)`
    allocates two full-size temporaries -- (logit - col_max) AND its exp -- which
    OOMs at multi-tau B=2048 on 48 GB cards. We use in-place exp here so peak
    memory is one full-size temporary instead of two (saves ~1.6 GB at our
    coupling matrix sizes).
    """
    if not is_dist_avail_and_initialized():
        return torch.logsumexp(logit, dim=-2, keepdim=keepdim)
    # Subtract the global per-column max for numerical stability.
    col_max = logit.amax(dim=-2, keepdim=True)
    col_max = reduce_tensor(col_max, op="max")
    shifted = logit - col_max          # one full-size temp
    shifted.exp_()                     # in-place; no second full-size temp
    exp_shifted_sum = shifted.sum(dim=-2, keepdim=True)
    del shifted
    exp_shifted_sum = reduce_tensor(exp_shifted_sum, op="sum")
    out = col_max + torch.log(exp_shifted_sum.clamp_min(1e-30))
    if not keepdim:
        out = out.squeeze(-2)
    return out


def _two_sided_plan(logit: Tensor) -> Tensor:
    """A = sqrt(softmax_row(K) * softmax_col(K)), row-renormalized.

    Equivalent to log_A = logit - 0.5 * (lse_row + lse_col); we apply this
    in-place to `logit` to avoid allocating a separate log_A tensor.
    Caller is responsible for not relying on `logit` afterwards.
    """
    lse_row = torch.logsumexp(logit, dim=-1, keepdim=True)  # (G, N, 1)
    lse_col = col_logsumexp_ddp(logit, keepdim=True)        # (G, 1, M)
    # log_A in-place into logit
    logit.sub_(lse_row, alpha=0.5).sub_(lse_col, alpha=0.5)
    P = logit.exp_()  # alias; logit is now P
    return P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _sinkhorn_plan(logit: Tensor, iters: int) -> Tensor:
    """T-step log-domain Sinkhorn, with a final row-renormalize for barycentric use.

    All updates mutate `logit` in-place to keep peak memory at one full-size tensor.
    """
    for _ in range(iters):
        logit.sub_(torch.logsumexp(logit, dim=-1, keepdim=True))
        logit.sub_(col_logsumexp_ddp(logit, keepdim=True))
    logit.sub_(torch.logsumexp(logit, dim=-1, keepdim=True))
    P = logit.exp_()
    return P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _one_sided_plan(logit: Tensor) -> Tensor:
    logit.sub_(torch.logsumexp(logit, dim=-1, keepdim=True))
    return logit.exp_()


def _build_plan(logit: Tensor, plan_type: str, sinkhorn_iters: int) -> Tensor:
    """Dispatch over plan_type. Mutates logit in-place."""
    if plan_type == "two-sided":
        return _two_sided_plan(logit)
    if plan_type == "sinkhorn":
        return _sinkhorn_plan(logit, iters=sinkhorn_iters)
    if plan_type == "one-sided":
        return _one_sided_plan(logit)
    raise ValueError(f"Unknown plan_type: {plan_type}")


def compute_drift_split(
    x_real: Tensor,
    x_fake: Tensor,
    eps: float | list[float],
    plan_type: str = "two-sided",
    sinkhorn_iters: int = 20,
    dist_metric: str = "l2_sq",
    normalize_feature: bool = True,
    normalize_drift: bool = False,
) -> tuple[Tensor, dict]:
    """
    Args
    ----
    x_real:              (G, Nr_local, D) real features (no_grad upstream).
    x_fake:              (G, Nf_local, D) fake features (autograd link
                         upstream to the generator; this function is called
                         under torch.no_grad in train_unc.py and only returns V).
    eps:                 kernel temperature -- either a single float for
                         single-tau drift, or a list of floats for the
                         multi-temperature averaging scheme used by the
                         original Drifting paper (Algorithm 2). When a list
                         is given, V is computed at each tau independently
                         and summed (with optional per-tau normalize_drift).
    plan_type:           'two-sided' / 'sinkhorn' / 'one-sided'.
    sinkhorn_iters:      T (only used when plan_type='sinkhorn').
    dist_metric:         'l2_sq' (Gaussian kernel) or 'l2' (Laplacian).
    normalize_feature:   if True, divide all distances by their batch-global
                         mean, matching drifting-models-pytorch's convention.
    normalize_drift:     if True, each per-tau V is rescaled to unit RMS
                         BEFORE summing across taus (so the three taus
                         contribute equally to V_sum independent of their
                         raw magnitudes). No effect for single-tau.

    Returns
    -------
    V:    (G, Nf_local, D), drift field for the LOCAL fake samples.
          When eps is a list, V = sum_tau V_tau (optionally normalized).
    info: dict with 'data-scale' and per-tau Vnorm2 diagnostics.
    """
    eps_list = [float(eps)] if isinstance(eps, (int, float)) else [float(e) for e in eps]
    n_taus = len(eps_list)

    G, Nf_local, D = x_fake.shape

    # 1. Gather global y_real (positives) and y_neg (= all fakes across ranks).
    y_pos = torch.cat(gather_tensor(x_real), dim=1)  # (G, Nr_global, D)
    y_neg = torch.cat(gather_tensor(x_fake), dim=1)  # (G, Nf_global, D)

    # 2. Pairwise L2 distances. We compute L2 first; the Gaussian kernel then
    #    squares post-normalization. Doing it this way keeps feature
    #    normalization meaningful for both Gaussian and Laplacian.
    d_pos = torch.cdist(x_fake, y_pos)  # (G, Nf_local, Nr_global)
    d_neg = torch.cdist(x_fake, y_neg)  # (G, Nf_local, Nf_global)

    info: dict = {}

    if normalize_feature:
        # Mean of L2 distances across pos+neg WITHOUT a torch.cat (avoids a
        # 4 GB temp at B=2048). Use weighted means then combine.
        n_pos = float(d_pos.numel())
        n_neg = float(d_neg.numel())
        dist_scale = (d_pos.sum() + d_neg.sum()) / (n_pos + n_neg)
        dist_scale = reduce_tensor(dist_scale).item()
        dist_scale = max(dist_scale, 1e-3)
        d_pos.div_(dist_scale)
        d_neg.div_(dist_scale)
        # Match drifting-models-pytorch convention: data_scale = dist_scale / sqrt(D)
        # so that scaled coordinates have unit-order magnitude.
        data_scale = max(dist_scale / math.sqrt(float(D)), 1e-3)
        y_pos = y_pos / data_scale
        y_neg = y_neg / data_scale
        info["data-scale"] = data_scale
    else:
        info["data-scale"] = 1.0

    # 3. Build self-mask once (used for non-Sinkhorn plans). Tiny tensor.
    self_mask = None
    if plan_type != "sinkhorn":
        rank_offset = get_rank() * 1_000_000
        index_x = torch.arange(Nf_local, device=x_fake.device) + rank_offset
        index_neg = torch.cat(gather_tensor(index_x), dim=0)
        self_mask = (index_x[:, None] == index_neg[None, :]).unsqueeze(0)

    # 4. Multi-tau loop. For n_taus > 1 we keep the (normalized) L2 distance
    #    matrices around and clone per iteration so each tau gets a fresh
    #    tensor for in-place kernel ops. For n_taus == 1 we alias to avoid
    #    the clone (so memory profile matches single-tau exactly).
    V_sum: Tensor | None = None
    for eps_val in eps_list:
        d_pos_t = d_pos if n_taus == 1 else d_pos.clone()
        d_neg_t = d_neg if n_taus == 1 else d_neg.clone()

        # POSITIVE branch
        if dist_metric == "l2_sq":
            d_pos_t.pow_(2).neg_().div_(eps_val)
        elif dist_metric == "l2":
            d_pos_t.neg_().div_(eps_val)
        else:
            raise ValueError(f"Unknown dist_metric: {dist_metric}")
        P_xy = _build_plan(d_pos_t, plan_type, sinkhorn_iters)
        drift_pos = P_xy @ y_pos
        del P_xy, d_pos_t

        # NEGATIVE branch
        if dist_metric == "l2_sq":
            d_neg_t.pow_(2).neg_().div_(eps_val)
        else:  # l2
            d_neg_t.neg_().div_(eps_val)
        if self_mask is not None:
            d_neg_t.masked_fill_(self_mask, float("-inf"))
        P_xx = _build_plan(d_neg_t, plan_type, sinkhorn_iters)
        drift_neg = P_xx @ y_neg
        del P_xx, d_neg_t

        V_t = drift_pos - drift_neg
        del drift_pos, drift_neg

        # Per-tau diagnostic (raw magnitude, before optional normalize_drift)
        V2_t = (V_t * V_t).mean()
        V2_t = reduce_tensor(V2_t)
        info[f"Vnorm2-eps{eps_val}"] = V2_t.item()

        # Optional drift normalization (per-tau unit RMS), then accumulate.
        if normalize_drift:
            V_t = V_t / torch.sqrt(V2_t.clamp_min(1e-8))

        V_sum = V_t if V_sum is None else V_sum + V_t

    return V_sum, info
