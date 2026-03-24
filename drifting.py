import torch
from torch import Tensor

from utils.distributed import get_rank, gather_tensor, reduce_tensor, is_dist_avail_and_initialized


def col_softmax_ddp(logit: Tensor) -> Tensor:
    if not is_dist_avail_and_initialized():
        return logit.softmax(dim=-2)
    col_max = logit.max(dim=-2).values
    col_max = reduce_tensor(col_max, op="max")
    exp_local = torch.exp(logit - col_max.unsqueeze(0))
    col_sum = exp_local.sum(dim=-2)
    col_sum = reduce_tensor(col_sum, op="sum")
    return exp_local / col_sum.unsqueeze(0)


def compute_drift(
        x_real: Tensor,
        x_fake: Tensor,
        kernel_temp: float | list[float] = 0.05,
        implementation: str = "paper-algorithm2",
        normalize_feature: bool = False,
        normalize_drift: bool = False,
):
    shape = x_fake.shape
    if isinstance(kernel_temp, float):
        kernel_temp = [kernel_temp]

    # assign a unique index for each fake sample (for self-masking)
    index_fake = torch.arange(x_fake.shape[0], device=x_fake.device) + get_rank() * 1000000

    # gather real and fake samples
    x_fake_global = torch.cat(gather_tensor(x_fake.clone().detach()), dim=0)
    x_real_global = torch.cat(gather_tensor(x_real.clone().detach()), dim=0)
    index_fake_global = torch.cat(gather_tensor(index_fake.clone().detach()), dim=0)

    # rename and reshape variables
    x = x_fake.flatten(1)
    y_pos = x_real_global.flatten(1)
    y_neg = x_fake_global.flatten(1)
    index_x = index_fake
    index_neg = index_fake_global

    N, D = x.shape
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # (N, N_pos)
    dist_neg = torch.cdist(x, y_neg)  # (N, N_neg)

    # feature normalization
    if normalize_feature:
        dist_scale = torch.cat([dist_pos, dist_neg], dim=1).mean()
        dist_pos = dist_pos / dist_scale.clamp(min=1e-3)
        dist_neg = dist_neg / dist_scale.clamp(min=1e-3)
        data_scale = dist_scale / (D ** 0.5)
        x = x / data_scale.clamp(min=1e-3)
        y_pos = y_pos / data_scale.clamp(min=1e-3)
        y_neg = y_neg / data_scale.clamp(min=1e-3)

    # self-masking
    mask = index_x[:, None] == index_neg[None, :]
    dist_neg.masked_fill_(mask, 1e6)

    # compute drifting fields for each temperature
    info = {}
    V_sum = torch.zeros_like(x)

    for temp in kernel_temp:
        # compute logits
        logit_pos = -dist_pos / temp  # (N, N_pos)
        logit_neg = -dist_neg / temp  # (N, N_neg)

        # compute the drifting field
        if implementation == "paper-algorithm2":
            # follow the Algorithm 2 in the paper
            logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)
            A_row = torch.softmax(logit, dim=-1)
            A_col = col_softmax_ddp(logit)
            A = torch.sqrt(A_row * A_col)
            A_pos, A_neg = A.split([N_pos, N_neg], dim=1)
            W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
            W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)
            drift_pos = W_pos @ y_pos  # (N, D)
            drift_neg = W_neg @ y_neg  # (N, D)
            V = drift_pos - drift_neg  # (N, D)

        elif implementation == "paper-equation11":
            # follow the Eq.(11) in the paper
            W_pos = logit_pos.softmax(dim=-1)  # (N, N_pos)
            W_neg = logit_neg.softmax(dim=-1)  # (N, N_neg)
            drift_pos = W_pos @ y_pos  # (N, D)
            drift_neg = W_neg @ y_neg  # (N, D)
            V = drift_pos - drift_neg  # (N, D)

        else:
            raise ValueError(f"Unknown drift implementation: {implementation}")

        # collect ||V||^2
        V_norm2 = (V ** 2).sum(dim=1).mean()
        V_norm2 = reduce_tensor(V_norm2)
        info[f"V-norm2-temp{temp}"] = V_norm2.item()

        # drift normalization
        if normalize_drift:
            V_scale = torch.sqrt((V_norm2 / D).clamp(min=1e-8))
            V = V / V_scale

        # sum all drifting fields
        V_sum = V_sum + V

    V_sum = V_sum.reshape(*shape)
    return V_sum, info
