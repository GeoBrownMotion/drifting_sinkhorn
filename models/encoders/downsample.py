import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Downsample(nn.Module):
    def __init__(self, downratios: list[int]):
        super().__init__()
        self.downratios = downratios

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        H, W = x.shape[-2:]
        xdown = {"data": x}
        for r in self.downratios:
            h, w = H // r, W // r
            x = F.interpolate(x, size=(h, w), mode="bilinear")
            xdown[f"down{r}"] = x
        return xdown
