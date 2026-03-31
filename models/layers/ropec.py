import math
import torch
from torch import Tensor


def rotary_embedding(idx: Tensor, dim: int, base: float = 10000) -> Tensor:
    """idx (*) -> freqs_cis (*, dim/2) [complex tensor]"""
    assert dim % 2 == 0
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(-math.log(base) * freqs).to(device=idx.device)
    theta = idx.float()[..., None] * freqs
    freqs_cis = torch.polar(torch.ones_like(theta), theta)
    return freqs_cis


def apply_rotary_embedding(x: Tensor, freqs_cis: Tensor, style: str = "gpt-j") -> Tensor:
    """x (B, H, L, D), freqs_cis (L, D/2) -> (B, H, L, D)"""
    if style == "gpt-j":
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_ = torch.view_as_complex(x_)
        x_out = torch.view_as_real(x_ * freqs_cis).reshape(*x.shape)
    elif style == "gpt-neox":
        x_ = torch.stack(x.float().chunk(2, dim=-1), dim=-1)
        x_ = torch.view_as_complex(x_)
        x_out = torch.view_as_real(x_ * freqs_cis).transpose(-2, -1).reshape(*x.shape)
    else:
        raise ValueError(f"Unknown rotary embedding style: {style}")
    return x_out.type_as(x)


def get_1d_rotary_positional_embedding(
        seq_len: int,
        dim: int,
        base: float = 10000,
        max_seq_len: float = None,
) -> Tensor:
    """return freqs_cis (seq_len, dim/2) [complex tensor]"""
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
    """return freqs_cis (height, width, dim/2) [complex tensor]"""
    max_height = max_height or height
    max_width = max_width or width
    grid_h = torch.arange(height, dtype=torch.float32) / height * max_height
    grid_w = torch.arange(width, dtype=torch.float32) / width * max_width
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    freqs_cis_h = rotary_embedding(grid[0], dim // 2, base=base)
    freqs_cis_w = rotary_embedding(grid[1], dim // 2, base=base)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1)
    return freqs_cis


def get_nd_rotary_positional_embedding(
        shape: list[int],
        axes_dim: list[int],
        base: float = 10000,
        max_shape: list[float] = None,
) -> Tensor:
    """return freqs_cis (*shape, sum(axes_dim)/2) [complex tensor]"""
    assert len(shape) == len(axes_dim)
    max_shape = [None] * len(shape) if max_shape is None else max_shape
    max_shape = [max_shape[i] or shape[i] for i in range(len(shape))]
    grid_axes = [torch.arange(shape[i], dtype=torch.float32) / shape[i] * max_shape[i] for i in range(len(shape))]
    grid = torch.meshgrid(*grid_axes, indexing="ij")
    freqs_cis_axes = [rotary_embedding(grid[i], axes_dim[i], base=base) for i in range(len(shape))]
    freqs_cis = torch.cat(freqs_cis_axes, dim=-1)
    return freqs_cis
