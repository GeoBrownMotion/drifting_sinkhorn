import torch.nn as nn
from torch import Tensor


class IdentityAutoencoder(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        return x

    def decode(self, z: Tensor) -> Tensor:
        return z
