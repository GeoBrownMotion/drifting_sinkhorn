import torch
import torch.nn as nn
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """x (B, L, D), shift (B, [L], D), scale (B, [L], D) -> (B, L, D)"""
    shift = shift.unsqueeze(1) if shift.ndim == 2 else shift
    scale = scale.unsqueeze(1) if scale.ndim == 2 else scale
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """x (B, L, D) -> (B, L, D)"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
