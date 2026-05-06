import math

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed import get_rank, get_world_size, gather_tensor, reduce_tensor, is_dist_avail_and_initialized


def col_softmax_ddp(logit: Tensor) -> Tensor:
    """Compute the column-wise softmax in distributed mode."""
    if not is_dist_avail_and_initialized():
        return logit.softmax(dim=-2)
    col_max = logit.max(dim=-2).values
    col_max = reduce_tensor(col_max, op="max")
    exp_local = torch.exp(logit - col_max.unsqueeze(-2))
    col_sum = exp_local.sum(dim=-2)
    col_sum = reduce_tensor(col_sum, op="sum")
    return exp_local / col_sum.unsqueeze(-2)


def col_logsumexp_ddp(logit: Tensor) -> Tensor:
    """Compute logsumexp over the row dimension, across DDP ranks."""
    if not is_dist_avail_and_initialized():
        return torch.logsumexp(logit, dim=-2)
    col_max = logit.max(dim=-2).values
    col_max = reduce_tensor(col_max, op="max")
    exp_local = torch.exp(logit - col_max.unsqueeze(-2)).sum(dim=-2)
    exp_sum = reduce_tensor(exp_local, op="sum")
    return torch.log(exp_sum.clamp_min(1e-30)) + col_max


def _pairwise_distance(x: Tensor, y: Tensor, dist_metric: str) -> Tensor:
    dist = torch.cdist(x, y)
    if dist_metric == "l2":
        return dist
    if dist_metric == "l2_sq":
        return dist.square()
    raise ValueError(f"Unknown distance metric: {dist_metric}")


def _row_entropy_stats(plan: Tensor) -> tuple[float, float]:
    row_sum = plan.sum(dim=-1)
    entropy = -(plan.clamp_min(1e-30) * plan.clamp_min(1e-30).log()).sum(dim=-1)
    eff_neighbors = torch.exp(entropy).mean()
    row_mae = (row_sum - 1.0).abs().mean()
    eff_neighbors = reduce_tensor(eff_neighbors)
    row_mae = reduce_tensor(row_mae)
    return eff_neighbors.item(), row_mae.item()


def _col_mae_stats(plan: Tensor) -> float:
    col_sum = plan.sum(dim=-2)
    col_sum = reduce_tensor(col_sum, op="sum")
    target = plan.shape[-2] * get_world_size() / plan.shape[-1]
    return (col_sum - target).abs().mean().item()


def _barycentric_plan(
        logit: Tensor,
        plan: str,
        sinkhorn_iters: int = 30,
) -> Tensor:
    """Return row-stochastic barycentric weights for local rows and global columns."""
    if plan == "one-sided":
        return torch.softmax(logit, dim=-1)

    if plan == "two-sided":
        log_row = logit - torch.logsumexp(logit, dim=-1, keepdim=True)
        log_col = logit - col_logsumexp_ddp(logit).unsqueeze(-2)
        log_plan = 0.5 * (log_row + log_col)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
        return torch.exp(log_plan)

    if plan == "sinkhorn":
        if sinkhorn_iters <= 0:
            raise ValueError(f"sinkhorn_iters must be positive, got {sinkhorn_iters}")
        num_rows = logit.shape[-2] * get_world_size()
        num_cols = logit.shape[-1]
        log_r = -math.log(float(num_rows))
        log_c = -math.log(float(num_cols))
        log_u = torch.zeros_like(logit[..., :, 0])
        log_v = torch.zeros_like(logit[..., 0, :])
        for _ in range(int(sinkhorn_iters)):
            log_u = log_r - torch.logsumexp(logit + log_v.unsqueeze(-2), dim=-1)
            log_v = log_c - col_logsumexp_ddp(logit + log_u.unsqueeze(-1))
        log_plan = logit + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
        return torch.exp(log_plan)

    raise ValueError(f"Unknown barycentric plan: {plan}")


def _fake_self_mask(x: Tensor, y_neg: Tensor) -> Tensor:
    index_x = torch.arange(x.shape[1], device=x.device) + get_rank() * 1000000
    index_neg = torch.cat(gather_tensor(index_x), dim=0)
    return torch.eq(index_x[:, None], index_neg[None, :]).expand(x.shape[0], -1, -1)


def compute_barycentric_drift(
        x_real: Tensor,
        x_fake: Tensor,
        tau: float = 0.01,
        plan: str = "sinkhorn",
        dist_metric: str = "l2_sq",
        sinkhorn_iters: int = 30,
        normalize_feature: bool = False,
        normalize_drift: bool = False,
        self_mask: str = "non_sinkhorn",
) -> tuple[Tensor, dict]:
    """Compute FFHQ/toy-style minibatch barycentric drift.

    V = P(fake, real) @ real - P(fake, fake) @ fake, using global minibatches in DDP.
    Sinkhorn follows the professor FFHQ code by default: no self-distance masking.
    """
    x = x_fake.float()
    y_pos = torch.cat(gather_tensor(x_real.float()), dim=1)
    y_neg = torch.cat(gather_tensor(x_fake.float()), dim=1)
    _, _, dim = x.shape

    if normalize_feature:
        scale_dist = torch.cat([
            torch.cdist(x, y_pos),
            torch.cdist(x, y_neg),
        ], dim=-1)
        dist_scale = reduce_tensor(scale_dist.mean()).item()
        data_scale = dist_scale / (dim ** 0.5)
        x = x / max(data_scale, 1e-3)
        y_pos = y_pos / max(data_scale, 1e-3)
        y_neg = y_neg / max(data_scale, 1e-3)
    else:
        data_scale = 1.0

    dist_pos = _pairwise_distance(x, y_pos, dist_metric)
    dist_neg = _pairwise_distance(x, y_neg, dist_metric)

    if self_mask == "always" or (self_mask == "non_sinkhorn" and plan != "sinkhorn"):
        dist_neg = dist_neg.masked_fill(_fake_self_mask(x, y_neg), torch.inf)
    elif self_mask not in {"none", "non_sinkhorn", "always"}:
        raise ValueError(f"Unknown self_mask mode: {self_mask}")

    logit_pos = -dist_pos / float(tau)
    logit_neg = -dist_neg / float(tau)
    P_pos = _barycentric_plan(logit_pos, plan=plan, sinkhorn_iters=sinkhorn_iters)
    P_neg = _barycentric_plan(logit_neg, plan=plan, sinkhorn_iters=sinkhorn_iters)
    V = P_pos @ y_pos - P_neg @ y_neg

    Vnorm2 = reduce_tensor((V ** 2).mean())
    if normalize_drift:
        V = V / torch.sqrt(Vnorm2.clamp(min=1e-8))

    pos_eff, pos_row_mae = _row_entropy_stats(P_pos)
    neg_eff, neg_row_mae = _row_entropy_stats(P_neg)
    info = {
        "data-scale": data_scale,
        f"Vnorm2-tau{tau}": Vnorm2.item(),
        "pos-eff-neighbors": pos_eff,
        "neg-eff-neighbors": neg_eff,
        "pos-row-mae": pos_row_mae,
        "neg-row-mae": neg_row_mae,
    }
    if plan in {"two-sided", "sinkhorn"}:
        info["pos-col-mae"] = _col_mae_stats(P_pos)
        info["neg-col-mae"] = _col_mae_stats(P_neg)
    return V, info


def compute_drift(
        x_real: Tensor,
        x_fake: Tensor,
        kernel_temp: float | list[float] = 0.05,
        kernel_norm: str = "mutual-softmax",
        normalize_feature: bool = False,
        normalize_drift: bool = False,
) -> tuple[Tensor, dict]:
    """Compute the drifting field in a group-wise manner.

    Args:
        x_real: Groups of real samples, shape (G, Nr, D).
        x_fake: Groups of fake samples, shape (G, Nf, D).
        kernel_temp: Temperature of the kernel.
        kernel_norm: How to compute the drifting field.
        normalize_feature: Whether to normalize the feature.
        normalize_drift: Whether to normalize the drifting field.

    Returns:
        V: The drifting field for each fake sample, shape (G, Nf, D).
        info: Any additional information.

    References:
        1. "Generative Modeling via Drifting". https://arxiv.org/abs/2602.04770
    """
    # get x, y_pos, y_neg
    x = x_fake                                       # (G, N, D)
    y_pos = torch.cat(gather_tensor(x_real), dim=1)  # (G, N_pos, D)
    y_neg = torch.cat(gather_tensor(x_fake), dim=1)  # (G, N_neg, D)
    G, N, D = x.shape
    N_pos = y_pos.shape[1]
    N_neg = y_neg.shape[1]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)               # (G, N, N_pos)
    dist_neg = torch.cdist(x, y_neg)               # (G, N, N_neg)
    dist = torch.cat([dist_pos, dist_neg], dim=2)  # (G, N, N_pos + N_neg)

    # feature normalization
    if normalize_feature:
        dist_scale = dist.mean()
        dist_scale = reduce_tensor(dist_scale).item()
        dist = dist / max(dist_scale, 1e-3)
        data_scale = dist_scale / (D ** 0.5)
        x = x / max(data_scale, 1e-3)
        y_pos = y_pos / max(data_scale, 1e-3)
        y_neg = y_neg / max(data_scale, 1e-3)
    else:
        data_scale = 1.

    # self-masking
    index_x = torch.arange(N, device=x.device) + get_rank() * 1000000   # (N, )
    index_neg = torch.cat(gather_tensor(index_x), dim=0)                # (N_neg, )
    mask = torch.eq(index_x[:, None], index_neg[None, :])               # (N, N_neg)
    mask = F.pad(mask, pad=(N_pos, 0), value=False)                     # (N, N_pos + N_neg)
    dist.masked_fill_(mask.unsqueeze(0), torch.inf)                     # (G, N, N_pos + N_neg)

    # compute drifting fields for each temperature
    info = {"data-scale": data_scale}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]

    for temp in kernel_temp:
        # compute logits
        logit = -dist / temp  # (G, N, N_pos + N_neg)

        # compute the drifting field
        if kernel_norm == "mutual-softmax":
            # follow the Algorithm 2 in the paper
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A_pos, A_neg = A.split([N_pos, N_neg], dim=-1)   # (G, N, N_pos), (G, N, N_neg)
            W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # (G, N, N_pos)
            W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # (G, N, N_neg)
            drift_pos = W_pos @ y_pos  # (G, N, D)
            drift_neg = W_neg @ y_neg  # (G, N, D)
            V = drift_pos - drift_neg  # (G, N, D)
        elif kernel_norm == "softmax":
            logit_pos, logit_neg = logit.split([N_pos, N_neg], dim=-1)
            W_pos = logit_pos.softmax(dim=-1)  # (G, N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (G, N, N_neg)
            drift_pos = W_pos @ y_pos  # (G, N, D)
            drift_neg = W_neg @ y_neg  # (G, N, D)
            V = drift_pos - drift_neg  # (G, N, D)
        else:
            raise ValueError(f"Unknown kernel normalization: {kernel_norm}")

        # collect ||V||^2
        Vnorm2 = (V ** 2).mean()
        Vnorm2 = reduce_tensor(Vnorm2)
        info[f"Vnorm2-temp{temp}"] = Vnorm2.item()

        # drift normalization
        if normalize_drift:
            V_scale = torch.sqrt(Vnorm2.clamp(min=1e-8))
            V = V / V_scale

        # sum all drifting fields
        V_sum = V_sum + V
    return V_sum, info


def compute_drift_c2i(
        x_real: Tensor,
        x_fake: Tensor,
        x_unc: Tensor,
        alpha: Tensor,
        kernel_temp: float | list[float] = 0.05,
        kernel_norm: str = "mutual-softmax",
        normalize_feature: bool = False,
        normalize_drift: bool = False,
) -> tuple[Tensor, dict]:
    """Compute the class-conditional drifting field in a group-wise manner.

    Args:
        x_real: Groups of real samples, shape (G, Nr, D). Samples in the same group should belong to the same class.
        x_fake: Groups of fake samples, shape (G, Nf, D). Samples in the same group should belong to the same class.
        x_unc: Groups of unconditional samples, shape (G, Nu, D).
        alpha: Classifier-free guidance (CFG) scale, shape (G, Nf).
        kernel_temp: Temperature of the kernel.
        kernel_norm: How to compute the drifting field.
        normalize_feature: Whether to normalize the feature.
        normalize_drift: Whether to normalize the drifting field.

    Returns:
        V: The drifting field for each fake sample, shape (B, Nf, D).
        info: Any additional information.

    References:
        1. "Generative Modeling via Drifting". https://arxiv.org/abs/2602.04770
    """
    # get x, y_pos, y_neg
    x = x_fake                                         # (G, N, D)
    y_pos = torch.cat(gather_tensor(x_real), dim=1)    # (G, N_pos, D)
    y_neg_f = torch.cat(gather_tensor(x_fake), dim=1)  # (G, N_neg_f, D)
    y_neg_u = torch.cat(gather_tensor(x_unc), dim=1)   # (G, N_neg_u, D)
    B, N, D = x.shape
    N_pos = y_pos.shape[1]
    N_neg_f = y_neg_f.shape[1]
    N_neg_u = y_neg_u.shape[1]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)                             # (G, N, N_pos)
    dist_neg_f = torch.cdist(x, y_neg_f)                         # (G, N, N_neg_f)
    dist_neg_u = torch.cdist(x, y_neg_u)                         # (G, N, N_neg_u)
    dist = torch.cat([dist_pos, dist_neg_f, dist_neg_u], dim=2)  # (G, N, N_pos + N_neg_f + N_neg_u)

    # compute distance weight
    w_unc = (alpha - 1) * (N_neg_f - 1) / N_neg_u                # (G, N)
    weight = w_unc.unsqueeze(-1).repeat(1, 1, N_neg_u)           # (G, N, N_neg_u)
    weight = F.pad(weight, pad=(N_pos + N_neg_f, 0), value=1.0)  # (G, N, N_pos + N_neg_f + N_neg_u)
    weighted_dist = dist * weight                                # (G, N, N_pos + N_neg_f + N_neg_u)

    # feature normalization
    if normalize_feature:
        dist_scale = weighted_dist.mean() / weight.mean()
        dist_scale = reduce_tensor(dist_scale).item()
        dist = dist / max(dist_scale, 1e-3)
        data_scale = dist_scale / (D ** 0.5)
        x = x / max(data_scale, 1e-3)
        y_pos = y_pos / max(data_scale, 1e-3)
        y_neg_f = y_neg_f / max(data_scale, 1e-3)
        y_neg_u = y_neg_u / max(data_scale, 1e-3)
    else:
        data_scale = 1.

    # self-masking
    index_x = torch.arange(N, device=x.device) + get_rank() * 1000000  # (N, )
    index_neg_f = torch.cat(gather_tensor(index_x), dim=0)             # (N_neg_f, )
    mask = torch.eq(index_x[:, None], index_neg_f[None, :])            # (N, N_neg_f)
    mask = F.pad(mask, pad=(N_pos, N_neg_u), value=False)              # (N, N_pos + N_neg_f + N_neg_u)
    dist.masked_fill_(mask.unsqueeze(0), torch.inf)                    # (G, N, N_pos + N_neg_f + N_neg_u)

    # combine negative samples
    N_neg = N_neg_f + N_neg_u
    y_neg = torch.cat([y_neg_f, y_neg_u], dim=1)  # (G, N_neg, D)

    # compute drifting fields for each temperature
    info = {"data-scale": data_scale}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]

    for temp in kernel_temp:
        # compute logits
        logit = -dist / temp  # (G, N, N_pos + N_neg_f + N_neg_u)

        # compute the drifting field
        if kernel_norm == "mutual-softmax":
            # follow the Algorithm 2 in the paper
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A = A * weight
            A_pos, A_neg = A.split([N_pos, N_neg], dim=-1)   # (G, N, N_pos), (G, N, N_neg)
            W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # (G, N, N_pos)
            W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # (G, N, N_neg)
            drift_pos = W_pos @ y_pos  # (G, N, D)
            drift_neg = W_neg @ y_neg  # (G, N, D)
            V = drift_pos - drift_neg  # (G, N, D)
        elif kernel_norm == "softmax":
            logit = logit + torch.log(weight.clamp(min=1e-8))
            logit_pos, logit_neg = logit.split([N_pos, N_neg], dim=-1)
            W_pos = logit_pos.softmax(dim=-1)  # (G, N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (G, N, N_neg)
            drift_pos = W_pos @ y_pos  # (G, N, D)
            drift_neg = W_neg @ y_neg  # (G, N, D)
            V = drift_pos - drift_neg  # (G, N, D)
        else:
            raise ValueError(f"Unknown kernel normalization: {kernel_norm}")

        # collect ||V||^2
        Vnorm2 = (V ** 2).mean()
        Vnorm2 = reduce_tensor(Vnorm2)
        info[f"Vnorm2-temp{temp}"] = Vnorm2.item()

        # drift normalization
        if normalize_drift:
            V_scale = torch.sqrt(Vnorm2.clamp(min=1e-8))
            V = V / V_scale

        # sum all drifting fields
        V_sum = V_sum + V
    return V_sum, info
