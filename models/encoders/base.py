import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    def __init__(self, autoencoder: nn.Module = None):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        pass
