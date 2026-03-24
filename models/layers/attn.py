import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from models.layers.norm import RMSNorm


def attention(q: Tensor, k: Tensor, v: Tensor, dropout_p: float = 0.0, backend: str = "sdpa") -> Tensor:
    """q (B, H, L, D), k (B, H, S, D), v (B, H, S, D) -> (B, H, L, D)"""
    if backend == "naive":
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(q.size(0), 1, L, S, dtype=q.dtype).cuda()
        with torch.autocast("cuda", enabled=False):
            attn_weight = q.float() @ k.float().transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ v

    elif backend == "sdpa":
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    else:
        raise ValueError(f"Unknown attention backend: {backend}")
    return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_backend: str = "sdpa",
            apply_rope: Callable = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_backend = attn_backend
        self.apply_rope = apply_rope

        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, rope: Tensor = None) -> Tensor:
        """x (B, L, D), rope (L, d/2) -> (B, L, D)"""
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = self.apply_rope(q, rope)
            k = self.apply_rope(k, rope)

        x = attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., backend=self.attn_backend)
        x = rearrange(x, "B H L D -> B L (H D)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_backend: str = "sdpa",
            apply_rope: Callable = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_backend = attn_backend
        self.apply_rope = apply_rope

        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, y: Tensor, rope_x: Tensor = None, rope_y: Tensor = None) -> Tensor:
        """x (B, L, D), y (B, S, D), rope_x (L, d/2), rope_y (S, d/2) -> (B, L, D)"""
        q = self.q(x)
        kv = self.kv(y)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k, v = rearrange(kv, "B S (K H D) -> K B H S D", K=2, H=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_x is not None:
            q = self.apply_rope(q, rope_x)
        if rope_y is not None:
            k = self.apply_rope(k, rope_y)

        x = attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., backend=self.attn_backend)
        x = rearrange(x, "B H L D -> B L (H D)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
