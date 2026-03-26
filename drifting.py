import torch
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


def compute_drift(
        x_real: Tensor,
        x_fake: Tensor,
        kernel_temp: float | list[float] = 0.05,
        implementation: str = "paper-algorithm2",
        normalize_feature: bool = False,
        normalize_drift: bool = False,
) -> tuple[Tensor, dict]:
    """Compute the drifting field in a batch-wise manner.

    Args:
        x_real: Batches of real samples, shape (B, Nr, D).
        x_fake: Batches of fake samples, shape (B, Nf, D).
        kernel_temp: Temperature of the kernel.
        implementation: Algorithm 2 or Eq.(11) in the paper.
        normalize_feature: Whether to normalize the feature.
        normalize_drift: Whether to normalize the drifting field.

    Returns:
        V: The drifting field for each fake sample, shape (B, Nf, D).
        info: Any additional information.

    References:
        1. "Generative Modeling via Drifting". https://arxiv.org/abs/2602.04770
    """
    # get x, y_pos, y_neg
    x = x_fake                                       # (B, N, D)
    y_pos = torch.cat(gather_tensor(x_real), dim=1)  # (B, N_pos, D)
    y_neg = torch.cat(gather_tensor(x_fake), dim=1)  # (B, N_neg, D)
    B, N, D = x.shape
    N_pos = y_pos.shape[1]
    N_neg = y_neg.shape[1]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # (B, N, N_pos)
    dist_neg = torch.cdist(x, y_neg)  # (B, N, N_neg)

    # feature normalization
    if normalize_feature:
        dist_scale = torch.cat([dist_pos, dist_neg], dim=2).mean()
        dist_scale = reduce_tensor(dist_scale)
        dist_pos = dist_pos / dist_scale.clamp(min=1e-3)
        dist_neg = dist_neg / dist_scale.clamp(min=1e-3)
        data_scale = dist_scale / (D ** 0.5)
        x = x / data_scale.clamp(min=1e-3)
        y_pos = y_pos / data_scale.clamp(min=1e-3)
        y_neg = y_neg / data_scale.clamp(min=1e-3)

    # self-masking
    index_x = torch.arange(N, device=x.device) + get_rank() * 1000000  # (N, )
    index_neg = torch.cat(gather_tensor(index_x), dim=0)               # (N_neg, )
    mask = torch.eq(index_x[:, None], index_neg[None, :]).unsqueeze(0)
    dist_neg.masked_fill_(mask, 1e6)

    # compute drifting fields for each temperature
    info = {}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]

    for temp in kernel_temp:
        # compute logits
        logit_pos = -dist_pos / temp  # (B, N, N_pos)
        logit_neg = -dist_neg / temp  # (B, N, N_neg)

        # compute the drifting field
        if implementation == "paper-algorithm2":
            # follow the Algorithm 2 in the paper
            logit = torch.cat([logit_pos, logit_neg], dim=-1)  # (B, N, N_pos + N_neg)
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A_pos, A_neg = A.split([N_pos, N_neg], dim=-1)
            W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # (B, N, N_pos)
            W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # (B, N, N_neg)
            drift_pos = W_pos @ y_pos  # (B, N, D)
            drift_neg = W_neg @ y_neg  # (B, N, D)
            V = drift_pos - drift_neg  # (B, N, D)
        elif implementation == "paper-equation11":
            # follow the Eq.(11) in the paper
            W_pos = logit_pos.softmax(dim=-1)  # (B, N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (B, N, N_neg)
            drift_pos = W_pos @ y_pos  # (B, N, D)
            drift_neg = W_neg @ y_neg  # (B, N, D)
            V = drift_pos - drift_neg  # (B, N, D)
        else:
            raise ValueError(f"Unknown drift implementation: {implementation}")

        # collect ||V||^2
        Vnorm2 = (V ** 2).sum(dim=-1).mean()
        Vnorm2 = reduce_tensor(Vnorm2)
        info[f"Vnorm2-temp{temp}"] = Vnorm2.item()

        # drift normalization
        if normalize_drift:
            V_scale = torch.sqrt((Vnorm2 / D).clamp(min=1e-8))
            V = V / V_scale

        # sum all drifting fields
        V_sum = V_sum + V
    return V_sum, info


def compute_drift_c2i(
        x_real: Tensor,
        x_fake: Tensor,
        x_unc: Tensor,
        alpha: float = 1.0,
        kernel_temp: float | list[float] = 0.05,
        implementation: str = "paper-algorithm2",
        normalize_feature: bool = False,
        normalize_drift: bool = False,
) -> tuple[Tensor, dict]:
    """Compute the class-conditional drifting field in a batch-wise manner.

    Args:
        x_real: Batches of real samples, shape (B, Nr, D). Samples in the same batch should belong to the same class.
        x_fake: Batches of fake samples, shape (B, Nf, D). Samples in the same batch should belong to the same class.
        x_unc: Batches of unconditional samples, shape (B, Nu, D).
        alpha: Classifier-free guidance (CFG) scale.
        kernel_temp: Temperature of the kernel.
        implementation: Algorithm 2 or Eq.(11) in the paper.
        normalize_feature: Whether to normalize the feature.
        normalize_drift: Whether to normalize the drifting field.

    Returns:
        V: The drifting field for each fake sample, shape (B, Nf, D).
        info: Any additional information.

    References:
        1. "Generative Modeling via Drifting". https://arxiv.org/abs/2602.04770
    """
    # compute unc weight
    N_fake = x_fake.shape[0]
    N_unc = x_unc.shape[0]
    weight_unc = (alpha - 1) * (N_fake - 1) / N_unc

    # get x, y_pos, y_neg
    x = x_fake                                         # (B, N, D)
    y_pos = torch.cat(gather_tensor(x_real), dim=1)    # (B, N_pos, D)
    y_neg_f = torch.cat(gather_tensor(x_fake), dim=1)  # (B, N_neg_f, D)
    y_neg_u = torch.cat(gather_tensor(x_unc), dim=1)   # (B, N_neg_u, D)
    B, N, D = x.shape
    N_pos = y_pos.shape[1]
    N_neg_f = y_neg_f.shape[1]
    N_neg_u = y_neg_u.shape[1]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)      # (B, N, N_pos)
    dist_neg_f = torch.cdist(x, y_neg_f)  # (B, N, N_neg_f)
    dist_neg_u = torch.cdist(x, y_neg_u)  # (B, N, N_neg_u)
    dist_neg_u = dist_neg_u * weight_unc  # TODO

    # feature normalization
    if normalize_feature:
        dist_scale = torch.cat([dist_pos, dist_neg_f, dist_neg_u], dim=2).mean()
        dist_scale = reduce_tensor(dist_scale)
        dist_pos = dist_pos / dist_scale.clamp(min=1e-3)
        dist_neg_f = dist_neg_f / dist_scale.clamp(min=1e-3)
        dist_neg_u = dist_neg_u / dist_scale.clamp(min=1e-3)
        data_scale = dist_scale / (D ** 0.5)
        x = x / data_scale.clamp(min=1e-3)
        y_pos = y_pos / data_scale.clamp(min=1e-3)
        y_neg_f = y_neg_f / data_scale.clamp(min=1e-3)
        y_neg_u = y_neg_u / data_scale.clamp(min=1e-3)

    # self-masking
    index_x = torch.arange(N, device=x.device) + get_rank() * 1000000  # (N, )
    index_neg_f = torch.cat(gather_tensor(index_x), dim=0)             # (N_neg_f, )
    mask = torch.eq(index_x[:, None], index_neg_f[None, :]).unsqueeze(0)
    dist_neg_f.masked_fill_(mask, 1e6)

    # combine negative samples
    N_neg = N_neg_f + N_neg_u
    y_neg = torch.cat([y_neg_f, y_neg_u], dim=1)           # (B, N_neg, D)
    dist_neg = torch.cat([dist_neg_f, dist_neg_u], dim=2)  # (B, N, N_neg)

    # compute drifting fields for each temperature
    info = {}
    V_sum = torch.zeros_like(x)
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]

    for temp in kernel_temp:
        # compute logits
        logit_pos = -dist_pos / temp  # (B, N, N_pos)
        logit_neg = -dist_neg / temp  # (B, N, N_neg)

        # compute the drifting field
        if implementation == "paper-algorithm2":
            # follow the Algorithm 2 in the paper
            logit = torch.cat([logit_pos, logit_neg], dim=-1)  # (B, N, N_pos + N_neg)
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A_pos, A_neg = A.split([N_pos, N_neg], dim=-1)
            W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # (B, N, N_pos)
            W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # (B, N, N_neg)
            drift_pos = W_pos @ y_pos  # (B, N, D)
            drift_neg = W_neg @ y_neg  # (B, N, D)
            V = drift_pos - drift_neg  # (B, N, D)
        elif implementation == "paper-equation11":
            # follow the Eq.(11) in the paper
            W_pos = logit_pos.softmax(dim=-1)  # (B, N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (B, N, N_neg)
            drift_pos = W_pos @ y_pos  # (B, N, D)
            drift_neg = W_neg @ y_neg  # (B, N, D)
            V = drift_pos - drift_neg  # (B, N, D)
        else:
            raise ValueError(f"Unknown drift implementation: {implementation}")

        # collect ||V||^2
        Vnorm2 = (V ** 2).sum(dim=-1).mean()
        Vnorm2 = reduce_tensor(Vnorm2)
        info[f"Vnorm2-temp{temp}"] = Vnorm2.item()

        # drift normalization
        if normalize_drift:
            V_scale = torch.sqrt((Vnorm2 / D).clamp(min=1e-8))
            V = V / V_scale

        # sum all drifting fields
        V_sum = V_sum + V
    return V_sum, info
