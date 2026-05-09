import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed import get_rank, gather_tensor, reduce_tensor, is_dist_avail_and_initialized


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


def col_logsumexp_ddp(logit: Tensor, keepdim: bool = False, chunk_size: int = 1024) -> Tensor:
    """Compute the column-wise logsumexp in distributed mode."""
    if not is_dist_avail_and_initialized():
        return torch.logsumexp(logit, dim=-2, keepdim=keepdim)
    col_max = logit.max(dim=-2, keepdim=True).values
    col_max = reduce_tensor(col_max, op="max")
    col_sum = torch.empty_like(col_max)
    for start in range(0, logit.shape[-1], chunk_size):
        end = min(start + chunk_size, logit.shape[-1])
        exp_local = logit[..., start:end] - col_max[..., start:end]
        exp_local.exp_()
        exp_sum = exp_local.sum(dim=-2, keepdim=True)
        col_sum[..., start:end] = reduce_tensor(exp_sum, op="sum")
    out = col_max + torch.log(col_sum.clamp_min(1e-30))
    return out if keepdim else out.squeeze(-2)


def mutual_softmax_inplace_(logit: Tensor) -> Tensor:
    """A = sqrt(row_softmax(logit) * col_softmax(logit)), written into logit."""
    row_lse = torch.logsumexp(logit, dim=-1, keepdim=True)
    col_lse = col_logsumexp_ddp(logit, keepdim=True)
    logit.sub_(row_lse, alpha=0.5).sub_(col_lse, alpha=0.5)
    return logit.exp_()


def drift_from_coupling(A: Tensor, y_pos: Tensor, y_neg: Tensor, N_pos: int, N_neg: int) -> Tensor:
    """Compute drift from A without materializing W_pos/W_neg."""
    A_pos, A_neg = A.split([N_pos, N_neg], dim=-1)
    pos_mass = A_pos.sum(dim=-1, keepdim=True)
    neg_mass = A_neg.sum(dim=-1, keepdim=True)
    drift_pos = A_pos @ y_pos
    drift_neg = A_neg @ y_neg
    drift_pos.mul_(neg_mass)
    drift_neg.mul_(pos_mass)
    return drift_pos - drift_neg


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
    dist_pos = torch.cdist(x, y_pos)  # (G, N, N_pos)
    dist = dist_pos.new_empty(G, N, N_pos + N_neg)
    dist[..., :N_pos].copy_(dist_pos)
    del dist_pos
    dist_neg = torch.cdist(x, y_neg)  # (G, N, N_neg)
    dist[..., N_pos:].copy_(dist_neg)
    del dist_neg

    # feature normalization
    if normalize_feature:
        dist_scale = dist.mean()
        dist_scale = reduce_tensor(dist_scale).item()
        dist.div_(max(dist_scale, 1e-3))
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
    dist[..., N_pos:].masked_fill_(mask.unsqueeze(0), torch.inf)        # (G, N, N_neg)
    del index_x, index_neg, mask

    # compute drifting fields for each temperature
    info = {"data-scale": data_scale}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]
    single_temp = len(kernel_temp) == 1

    for temp in kernel_temp:
        # compute logits
        logit = dist if single_temp else dist.clone()
        logit.neg_().div_(temp)  # (G, N, N_pos + N_neg)

        # compute the drifting field
        if kernel_norm == "mutual-softmax":
            # follow the Algorithm 2 in the paper
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            V = drift_from_coupling(A, y_pos, y_neg, N_pos, N_neg)
            del A_row, A_col, A
        elif kernel_norm == "mutual-softmax2x":
            # follow the Algorithm 2 in the paper
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            #round 2
            A2_row = torch.softmax(A, dim=-1)
            A2_col = col_softmax_ddp(A)
            A2 = torch.sqrt(A2_row * A2_col)
            #continue like earlier
            A2_pos, A2_neg = A2.split([N_pos, N_neg], dim=-1)   # (G, N, N_pos), (G, N, N_neg)
            W_pos = A2_pos * A2_neg.sum(dim=-1, keepdim=True)  # (G, N, N_pos)
            W_neg = A2_neg * A2_pos.sum(dim=-1, keepdim=True)  # (G, N, N_neg)
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
            del logit_pos, logit_neg, W_pos, W_neg, drift_pos, drift_neg
        elif kernel_norm == "softmax-extra":
            logit_pos, logit_neg = logit.split([N_pos, N_neg], dim=-1)

            # first row softmax, same as normal softmax branch
            W_pos = logit_pos.softmax(dim=-1)
            W_neg = logit_neg.softmax(dim=-1)

            # extra column softmax
            W_pos = col_softmax_ddp(W_pos)
            W_neg = col_softmax_ddp(W_neg)

            # extra row softmax
            W_pos = W_pos.softmax(dim=-1)
            W_neg = W_neg.softmax(dim=-1)

            drift_pos = W_pos @ y_pos
            drift_neg = W_neg @ y_neg
            V = drift_pos - drift_neg
            del logit_pos, logit_neg, W_pos, W_neg, drift_pos, drift_neg

        else:
            raise ValueError(f"Unknown kernel normalization: {kernel_norm}")
        del logit
        if single_temp:
            del dist

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
    dist_pos = torch.cdist(x, y_pos)  # (G, N, N_pos)
    dist = dist_pos.new_empty(B, N, N_pos + N_neg_f + N_neg_u)
    dist[..., :N_pos].copy_(dist_pos)
    del dist_pos
    dist_neg_f = torch.cdist(x, y_neg_f)  # (G, N, N_neg_f)
    dist[..., N_pos:N_pos + N_neg_f].copy_(dist_neg_f)
    del dist_neg_f
    dist_neg_u = torch.cdist(x, y_neg_u)  # (G, N, N_neg_u)
    dist[..., N_pos + N_neg_f:].copy_(dist_neg_u)
    del dist_neg_u

    # compute distance weight
    w_unc = (alpha - 1) * (N_neg_f - 1) / N_neg_u                # (G, N)
    weight = w_unc.unsqueeze(-1).repeat(1, 1, N_neg_u)           # (G, N, N_neg_u)
    weight = F.pad(weight, pad=(N_pos + N_neg_f, 0), value=1.0)  # (G, N, N_pos + N_neg_f + N_neg_u)

    # feature normalization
    if normalize_feature:
        weighted_dist = dist * weight                            # (G, N, N_pos + N_neg_f + N_neg_u)
        dist_scale = weighted_dist.mean() / weight.mean()
        dist_scale = reduce_tensor(dist_scale).item()
        del weighted_dist
        dist.div_(max(dist_scale, 1e-3))
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
    dist[..., N_pos:N_pos + N_neg_f].masked_fill_(mask.unsqueeze(0), torch.inf)  # (G, N, N_neg_f)
    del index_x, index_neg_f, mask

    # combine negative samples
    N_neg = N_neg_f + N_neg_u
    y_neg = torch.cat([y_neg_f, y_neg_u], dim=1)  # (G, N_neg, D)

    # compute drifting fields for each temperature
    info = {"data-scale": data_scale}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]
    single_temp = len(kernel_temp) == 1

    for temp in kernel_temp:
        # compute logits
        logit = dist if single_temp else dist.clone()
        logit.neg_().div_(temp)  # (G, N, N_pos + N_neg_f + N_neg_u)

        # compute the drifting field
        if kernel_norm == "mutual-softmax":
            # follow the Algorithm 2 in the paper
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A.mul_(weight)
            V = drift_from_coupling(A, y_pos, y_neg, N_pos, N_neg)
            del A_row, A_col, A
        elif kernel_norm == "softmax":
            logit.add_(torch.log(weight.clamp(min=1e-8)))
            logit_pos, logit_neg = logit.split([N_pos, N_neg], dim=-1)
            W_pos = logit_pos.softmax(dim=-1)  # (G, N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (G, N, N_neg)
            drift_pos = W_pos @ y_pos  # (G, N, D)
            drift_neg = W_neg @ y_neg  # (G, N, D)
            V = drift_pos - drift_neg  # (G, N, D)
            del logit_pos, logit_neg, W_pos, W_neg, drift_pos, drift_neg
        else:
            raise ValueError(f"Unknown kernel normalization: {kernel_norm}")
        del logit
        if single_temp:
            del dist

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
