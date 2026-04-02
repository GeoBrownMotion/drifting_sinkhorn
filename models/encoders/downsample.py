import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DownsampleEncoder(nn.Module):
    def __init__(self, downratios: list[int], autoencoder: nn.Module = None):
        super().__init__()
        self.downratios = downratios

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        results.update({"xnorm": ((x ** 2).mean(dim=(2, 3)) + 1e-6).sqrt().unsqueeze(0)})
        # downsample
        H, W = x.shape[-2:]
        for r in self.downratios:
            h, w = H // r, W // r
            xd = F.interpolate(x, size=(h, w), mode="bilinear")
            results.update({f"down{r}": xd.flatten(1).unsqueeze(0)})
        return results
