import math
import torch
from torch import Tensor


def rotary_embedding(idx: Tensor, dim: int, base: float = 10000) -> Tensor:
    """idx (*) -> freqs_cis (*, dim/2, 2)"""
    assert dim % 2 == 0
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(-math.log(base) * freqs).to(device=idx.device)
    theta = idx.float()[..., None] * freqs
    freqs_cis = torch.stack([theta.cos(), theta.sin()], dim=-1)
    return freqs_cis


def apply_rotary_embedding(x: Tensor, freqs_cis: Tensor, style: str = "gpt-j") -> Tensor:
    """x (B, H, L, D), freqs_cis (L, D/2, 2) -> (B, H, L, D)"""
    freqs_cos, freqs_sin = torch.unbind(freqs_cis, dim=-1)
    if style == "gpt-j":
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_cos = x_ * freqs_cos.unsqueeze(-1)
        x_sin = torch.stack([-x_[..., 1], x_[..., 0]], dim=-1) * freqs_sin.unsqueeze(-1)
        x_out = (x_cos + x_sin).reshape(*x.shape)
    elif style == "gpt-neox":
        x1, x2 = x.float().chunk(2, dim=-1)
        x_cos = torch.cat([x1 * freqs_cos, x2 * freqs_cos], dim=-1)
        x_sin = torch.cat([-x2 * freqs_sin, x1 * freqs_sin], dim=-1)
        x_out = (x_cos + x_sin).reshape(*x.shape)
    else:
        raise ValueError(f"Unknown rotary embedding style: {style}")
    return x_out.type_as(x)


def get_1d_rotary_positional_embedding(
        seq_len: int,
        dim: int,
        base: float = 10000,
        max_seq_len: float = None,
) -> Tensor:
    """return freqs_cis (seq_len, dim/2, 2)"""
    max_seq_len = max_seq_len or seq_len
    idx = torch.arange(seq_len, dtype=torch.float32) / seq_len * max_seq_len
    freqs_cis = rotary_embedding(idx, dim, base=base)
    return freqs_cis


def get_2d_rotary_positional_embedding(
        height: int,
        width: int,
        dim: int,
        base: float = 10000,
        max_height: float = None,
        max_width: float = None,
) -> Tensor:
    """return freqs_cis (height, width, dim/2, 2)"""
    max_height = max_height or height
    max_width = max_width or width
    grid_h = torch.arange(height, dtype=torch.float32) / height * max_height
    grid_w = torch.arange(width, dtype=torch.float32) / width * max_width
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    freqs_cis_h = rotary_embedding(grid[0], dim // 2, base=base)
    freqs_cis_w = rotary_embedding(grid[1], dim // 2, base=base)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-2)
    return freqs_cis


def get_nd_rotary_positional_embedding(
        shape: list[int],
        axes_dim: list[int],
        base: float = 10000,
        max_shape: list[float] = None,
) -> Tensor:
    """return freqs_cis (*shape, sum(axes_dim)/2, 2)"""
    assert len(shape) == len(axes_dim)
    max_shape = [None] * len(shape) if max_shape is None else max_shape
    max_shape = [max_shape[i] or shape[i] for i in range(len(shape))]
    grid_axes = [torch.arange(shape[i], dtype=torch.float32) / shape[i] * max_shape[i] for i in range(len(shape))]
    grid = torch.meshgrid(*grid_axes, indexing="ij")
    freqs_cis_axes = [rotary_embedding(grid[i], axes_dim[i], base=base) for i in range(len(shape))]
    freqs_cis = torch.cat(freqs_cis_axes, dim=-2)
    return freqs_cis
