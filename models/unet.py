import torch
import torch.nn as nn
from torch import Tensor

from models.layers.attn import SelfAttention
from models.layers.embed import LabelEmbedder, TimestepEmbedder


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
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, embed_dim: int = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )
        self.adagn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels * 2),
        ) if embed_dim is not None else None

    def forward(self, x: Tensor, c: Tensor = None) -> Tensor:
        """x (B, C, H, W), [c(B, D)] -> (B, C, H, W)"""
        if self.adagn is not None:
            shift, scale = self.adagn(c).chunk(2, dim=-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
        else:
            shift, scale = 0., 0.
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        return h + self.shortcut(x)


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
            num_classes: int = 0,
    ):
        super().__init__()

        # first conv
        self.first_conv = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # class embedding
        embed_dim = dim * 4 if num_classes > 0 else None
        self.y_embedder = LabelEmbedder(embed_dim, num_classes) if num_classes > 0 else None
        self.alpha_embedder = TimestepEmbedder(embed_dim) if num_classes > 0 else None

        # downsample blocks
        dims = [dim]
        cur_size, cur_dim = input_size, dim
        self.down_blocks = nn.ModuleList([])
        for k, mult in enumerate(dim_mults):
            out_dim = dim * mult
            for i in range(num_res_blocks):
                self.down_blocks.append(ResBlock(cur_dim, out_dim, dropout=dropout, embed_dim=embed_dim))
                if cur_size in use_attn_size:
                    self.down_blocks.append(SelfAttention2D(out_dim, num_heads=num_heads))
                dims.append(out_dim)
                cur_dim = out_dim
            if k < len(dim_mults) - 1:
                self.down_blocks.append(Downsample(out_dim, out_dim))
                dims.append(out_dim)
                cur_size = cur_size // 2

        # bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(cur_dim, cur_dim, dropout=dropout, embed_dim=embed_dim),
            SelfAttention2D(cur_dim, num_heads=num_heads),
            ResBlock(cur_dim, cur_dim, dropout=dropout, embed_dim=embed_dim),
        ])

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for k, mult in enumerate(reversed(dim_mults)):
            out_dim = dim * mult
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(cur_dim + dims.pop(), out_dim, dropout=dropout, embed_dim=embed_dim))
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

    def forward(self, x: Tensor, y: Tensor = None, alpha: Tensor = None) -> Tensor:
        """x (B, C, H, W), [y (B, )], [alpha (B, )] -> (B, C, H, W)"""
        # first conv
        x = self.first_conv(x)
        skips = [x]

        # conditioning
        c = None
        if self.y_embedder is not None:
            alpha = torch.ones_like(y, dtype=torch.float32) if alpha is None else alpha
            c = self.y_embedder(y) + self.alpha_embedder(alpha)

        # downsample blocks
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                x = block(x, c)
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
        x = self.bottleneck[0](x, c)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, c)

        # upsample blocks
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                skip = skips.pop()
                x = block(torch.cat([x, skip], dim=1), c)
            elif isinstance(block, SelfAttention2D):
                x = block(x)
            elif isinstance(block, Upsample):
                x = block(x)
            else:
                raise ValueError(f"Unknown block type: {type(block)}")

        # final conv
        x = self.final_conv(x)
        return x
