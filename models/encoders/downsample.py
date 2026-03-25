import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DownsampleEncoder(nn.Module):
    def __init__(self, downratios: list[int]):
        super().__init__()
        self.downratios = downratios

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        H, W = x.shape[-2:]
        xdown = {}
        for r in self.downratios:
            h, w = H // r, W // r
            x = F.interpolate(x, size=(h, w), mode="bilinear")
            xdown[f"down{r}"] = x.flatten(1).unsqueeze(0)
        return xdown
