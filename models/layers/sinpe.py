import math
import torch
from torch import Tensor


def sinusoidal_embedding(idx: Tensor, dim: int, base: float = 10000, style: str = "cos-first") -> Tensor:
    """idx (*) -> embed (*, dim)"""
    assert dim % 2 == 0
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(-math.log(base) * freqs).to(device=idx.device)
    embed = idx.float()[..., None] * freqs
    if style == "cos-first":
        embed = torch.cat([torch.cos(embed), torch.sin(embed)], dim=-1)
    elif style == "sin-first":
        embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
    else:
        raise ValueError(f"Unknown sinusoidal embedding style: {style}")
    return embed


def get_1d_sinusoidal_positional_embedding(
        seq_len: int,
        dim: int,
        base: float = 10000,
        max_seq_len: float = None,
        style: str = "cos-first",
) -> Tensor:
    """return embed (seq_len, dim)"""
    max_seq_len = max_seq_len or seq_len
    idx = torch.arange(seq_len, dtype=torch.float32) / seq_len * max_seq_len
    embed = sinusoidal_embedding(idx, dim, base=base, style=style)
    return embed


def get_2d_sinusoidal_positional_embedding(
        height: int,
        width: int,
        dim: int,
        base: float = 10000,
        max_height: float = None,
        max_width: float = None,
        style: str = "cos-first",
        width_first: bool = False,
) -> Tensor:
    """return embed (height, width, dim)"""
    max_height = max_height or height
    max_width = max_width or width
    grid_h = torch.arange(height, dtype=torch.float32) / height * max_height
    grid_w = torch.arange(width, dtype=torch.float32) / width * max_width
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    embed_h = sinusoidal_embedding(grid[0], dim // 2, base=base, style=style)
    embed_w = sinusoidal_embedding(grid[1], dim // 2, base=base, style=style)
    embed = torch.cat([embed_w, embed_h], dim=-1) if width_first else torch.cat([embed_h, embed_w], dim=-1)
    return embed


def get_nd_sinusoidal_positional_embedding(
        shape: list[int],
        axes_dim: list[int],
        base: float = 10000,
        max_shape: list[float] = None,
        style: str = "cos-first",
) -> Tensor:
    """return embed (*shape, sum(axes_dim))"""
    assert len(shape) == len(axes_dim)
    max_shape = [None] * len(shape) if max_shape is None else max_shape
    max_shape = [max_shape[i] or shape[i] for i in range(len(shape))]
    grid_axes = [torch.arange(shape[i], dtype=torch.float32) / shape[i] * max_shape[i] for i in range(len(shape))]
    grid = torch.meshgrid(*grid_axes, indexing="ij")
    embed_axes = [sinusoidal_embedding(grid[i], axes_dim[i], base=base, style=style) for i in range(len(shape))]
    embed = torch.cat(embed_axes, dim=-1)
    return embed
