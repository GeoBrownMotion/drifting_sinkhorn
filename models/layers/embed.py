import torch.nn as nn
from torch import Tensor
from einops import rearrange

from models.layers.sinpe import sinusoidal_embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t (B, ) -> (B, D)"""
        t_freq = sinusoidal_embedding(t, self.frequency_embedding_dim).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, dim)

    def forward(self, y: Tensor) -> Tensor:
        """y (B, ) -> (B, D)"""
        embeddings = self.embedding_table(y)
        return embeddings


class PatchEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, patch_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, L=(H/P)*(W/P), D)"""
        x = self.proj(x)
        x = rearrange(x, "B D H W -> B (H W) D")
        return x


class BottleneckPatchEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bottleneck_dim: int, patch_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.proj1 = nn.Conv2d(in_dim, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, out_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, C, H, W) -> (B, L=(H/P)*(W/P), D)"""
        x = self.proj2(self.proj1(x))
        x = rearrange(x, "B D H W -> B (H W) D")
        return x
