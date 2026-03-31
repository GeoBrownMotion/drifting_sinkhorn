import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DownsampleEncoder(nn.Module):
    def __init__(self, downratios: list[int], autoencoder: nn.Module = None):
        super().__init__()
        self.downratios = downratios
        self.autoencoder = autoencoder

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if self.autoencoder is not None:
            x = self.autoencoder.decode(x)
        H, W = x.shape[-2:]
        xdown = {}
        for r in self.downratios:
            h, w = H // r, W // r
            x = F.interpolate(x, size=(h, w), mode="bilinear")
            xdown[f"down{r}"] = x.flatten(1).unsqueeze(0)
        return xdown
