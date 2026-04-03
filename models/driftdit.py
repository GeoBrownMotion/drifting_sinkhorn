import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from models.layers.ffn import SwiGLUFFN
from models.layers.attn import SelfAttention
from models.layers.norm import modulate, RMSNorm
from models.layers.embed import PatchEmbedder, LabelEmbedder, TimestepEmbedder
from models.layers.sinpe import (
    get_1d_sinusoidal_positional_embedding,
    get_2d_sinusoidal_positional_embedding,
)
from models.layers.ropem import (
    get_1d_rotary_positional_embedding,
    get_2d_rotary_positional_embedding,
    apply_rotary_embedding,
)

torch.set_float32_matmul_precision("high")

if os.environ.get("USE_TORCH_COMPILE", "0") == "1":
    maybe_compile = torch.compile
else:
    maybe_compile = lambda x: x


class DriftDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = RMSNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_backend="sdpa",
            apply_rope=apply_rotary_embedding,
        )
        self.mlp = SwiGLUFFN(hidden_dim, int(hidden_dim * mlp_ratio))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True))

    @maybe_compile
    def forward(self, x: Tensor, c: Tensor, rope: Tensor) -> Tensor:
        """x (B, L, D), c (B, [L], D), rope -> (B, L, D)"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=-1)
        gate_msa = gate_msa.unsqueeze(1) if gate_msa.ndim == 2 else gate_msa
        gate_mlp = gate_mlp.unsqueeze(1) if gate_mlp.ndim == 2 else gate_mlp
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))

    @maybe_compile
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """x (B, L, D), c (B, [L], D) -> (B, L, P*P*C)"""
        shift, scale = self.adaln(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DriftDiT(nn.Module):
    def __init__(
            self,
            input_size: int,
            patch_size: int,
            in_channels: int,
            hidden_dim: int,
            depth: int,
            num_heads: int,
            num_classes: int,
            num_registers: int,
            num_style_tokens: int,
            mlp_ratio: float = 4.0,
            checkpointing: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.num_style_tokens = num_style_tokens
        self.checkpointing = checkpointing

        self.head_dim = hidden_dim // num_heads
        self.grid_size = input_size // patch_size

        # patch embedding
        self.x_embedder = PatchEmbedder(in_channels, hidden_dim, patch_size, bias=True)

        # rotary positional embedding
        rope_img = get_2d_rotary_positional_embedding(self.grid_size, self.grid_size, self.head_dim)
        rope_img = rope_img.reshape(self.grid_size * self.grid_size, *rope_img.shape[2:])
        rope_reg = get_1d_rotary_positional_embedding(self.num_registers, self.head_dim)
        self.register_buffer("rope", torch.cat([rope_reg, rope_img], dim=0), persistent=False)

        # sinusoidal positional embedding
        sinpe_img = get_2d_sinusoidal_positional_embedding(self.grid_size, self.grid_size, hidden_dim)
        sinpe_img = sinpe_img.reshape(self.grid_size * self.grid_size, hidden_dim)
        sinpe_reg = get_1d_sinusoidal_positional_embedding(self.num_registers, hidden_dim)
        self.register_buffer("sinpe", torch.cat([sinpe_reg, sinpe_img], dim=0), persistent=False)

        # class embedding
        self.y_embedder = LabelEmbedder(hidden_dim, num_classes) if num_classes > 0 else None
        self.alpha_embedder = TimestepEmbedder(hidden_dim) if num_classes > 0 else None

        # style embedding
        self.style_embedder = nn.Embedding(64, hidden_dim) if num_style_tokens > 0 else None

        # registers
        self.reg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # transformer blocks
        self.blocks = nn.ModuleList([
            DriftDiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])

        # final layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)

        # initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # apply basic init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # class embedding
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # style embedding
        if self.style_embedder is not None:
            nn.init.normal_(self.style_embedder.weight, std=0.02)

        # adaln modulation
        for block in self.blocks:
            nn.init.constant_(block.adaln[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaln[-1].bias, 0)    # type: ignore
        nn.init.constant_(self.final_layer.adaln[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln[-1].bias, 0)

        # final layer
        nn.init.normal_(self.final_layer.linear.weight, 0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """x (B, L, D=P*P*C) -> (B, H, W, C)"""
        c = self.in_channels
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape((x.shape[0], c, h * p, h * p))
        return x

    def forward(self, x: Tensor, y: Tensor = None, alpha: Tensor = None) -> Tensor:
        """x (B, C, H, W), [y (B, )], [alpha (B, )] -> (B, C, H, W)"""
        # patch embedding
        x = self.x_embedder(x)
        B, L, D = x.shape

        # conditioning (class + alpha + style)
        c = torch.zeros((B, D), device=x.device)
        if self.y_embedder is not None:
            alpha = torch.ones_like(y, dtype=torch.float32) if alpha is None else alpha
            c = c + self.y_embedder(y) + self.alpha_embedder(alpha)
        if self.style_embedder is not None:
            style_index = torch.randperm(64, device=x.device)[:self.num_style_tokens]
            style_embedding = self.style_embedder(style_index).sum(dim=0)
            c = c + style_embedding.unsqueeze(0)

        # prepend register tokens
        registers = self.reg_proj(c).unsqueeze(1).repeat(1, self.num_registers, 1)
        x = torch.cat([registers, x], dim=1)
        x = x + self.sinpe.unsqueeze(0)

        # transformer blocks
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = checkpoint(block, x, c, self.rope, use_reentrant=False)
            else:
                x = block(x, c, self.rope)

        # remove register tokens
        x = x[:, self.num_registers:, :]

        # final layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


def DriftDiT_S_1(**kwargs):
    return DriftDiT(patch_size=1, hidden_dim=384, depth=12, num_heads=6, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_S_2(**kwargs):
    return DriftDiT(patch_size=2, hidden_dim=384, depth=12, num_heads=6, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_S_16(**kwargs):
    return DriftDiT(patch_size=16, hidden_dim=384, depth=12, num_heads=6, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_B_1(**kwargs):
    return DriftDiT(patch_size=1, hidden_dim=768, depth=12, num_heads=12, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_B_2(**kwargs):
    return DriftDiT(patch_size=2, hidden_dim=768, depth=12, num_heads=12, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_B_16(**kwargs):
    return DriftDiT(patch_size=16, hidden_dim=768, depth=12, num_heads=12, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_L_1(**kwargs):
    return DriftDiT(patch_size=1, hidden_dim=1024, depth=24, num_heads=16, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_L_2(**kwargs):
    return DriftDiT(patch_size=2, hidden_dim=1024, depth=24, num_heads=16, num_registers=16, num_style_tokens=32, **kwargs)


def DriftDiT_L_16(**kwargs):
    return DriftDiT(patch_size=16, hidden_dim=1024, depth=24, num_heads=16, num_registers=16, num_style_tokens=32, **kwargs)
