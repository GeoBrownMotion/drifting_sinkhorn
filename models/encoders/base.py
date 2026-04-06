import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        pass
