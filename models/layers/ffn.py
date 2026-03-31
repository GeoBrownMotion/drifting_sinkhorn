import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        return self.ffn(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True, multiple_of: int = None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if multiple_of is not None:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        return self.w2(self.ffn_dropout(F.silu(self.w1(x)) * self.w3(x)))
