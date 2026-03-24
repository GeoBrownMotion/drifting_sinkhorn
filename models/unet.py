import torch
import torch.nn as nn
from torch import Tensor

from models.layers.attn import SelfAttention


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x (B, Cin, H, W) -> (B, Cout, H*2, W*2)"""
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, Cin, H, W) -> (B, Cout, H/2, W/2)"""
        return self.down(x)


class SelfAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_backend: str = "sdpa"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, attn_backend=attn_backend)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)
        x = x + self.attn(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, C, H, W)"""
        return self.convs(x) + self.shortcut(x)


class UNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            in_channels: int,
            out_channels: int,
            dim: int = 128,
            dim_mults: list[int] = (1, 2, 2, 2),
            use_attn_size: list[int] = (16, ),
            num_res_blocks: int = 2,
            num_heads: int = 1,
            dropout: float = 0.0,
    ):
        super().__init__()

        # first conv
        self.first_conv = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # downsample blocks
        dims = [dim]
        cur_size, cur_dim = input_size, dim
        self.down_blocks = nn.ModuleList([])
        for k, mult in enumerate(dim_mults):
            out_dim = dim * mult
            for i in range(num_res_blocks):
                self.down_blocks.append(ResBlock(cur_dim, out_dim, dropout=dropout))
                if cur_size in use_attn_size:
                    self.down_blocks.append(SelfAttention2D(out_dim, num_heads=num_heads))
                dims.append(out_dim)
                cur_dim = out_dim
            if k < len(dim_mults) - 1:
                self.down_blocks.append(Downsample(out_dim, out_dim))
                dims.append(out_dim)
                cur_size = cur_size // 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(cur_dim, cur_dim, dropout=dropout),
            SelfAttention2D(cur_dim, num_heads=num_heads),
            ResBlock(cur_dim, cur_dim, dropout=dropout),
        )

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for k, mult in enumerate(reversed(dim_mults)):
            out_dim = dim * mult
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(cur_dim + dims.pop(), out_dim, dropout=dropout))
                if cur_size in use_attn_size:
                    self.up_blocks.append(SelfAttention2D(out_dim, num_heads=num_heads))
                cur_dim = out_dim
            if k < len(dim_mults) - 1:
                self.up_blocks.append(Upsample(out_dim, out_dim))
                cur_size = cur_size * 2

        # final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, cur_dim),
            nn.SiLU(),
            nn.Conv2d(cur_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, C, H, W)"""
        # first conv
        x = self.first_conv(x)
        skips = [x]

        # downsample blocks
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                x = block(x)
                skips.append(x)
            elif isinstance(block, SelfAttention2D):
                x = block(x)
                skips[-1] = x
            elif isinstance(block, Downsample):
                x = block(x)
                skips.append(x)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")

        # bottleneck
        x = self.bottleneck(x)

        # upsample blocks
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                skip = skips.pop()
                x = block(torch.cat([x, skip], dim=1))
            elif isinstance(block, SelfAttention2D):
                x = block(x)
            elif isinstance(block, Upsample):
                x = block(x)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")

        # final conv
        x = self.final_conv(x)
        return x
