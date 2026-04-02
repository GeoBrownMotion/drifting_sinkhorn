import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


class PatchEncoder(nn.Module):
    def __init__(self, patch_sizes: list[int], autoencoder: nn.Module = None):
        super().__init__()
        self.patch_sizes = patch_sizes

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        results.update({"xnorm": ((x ** 2).mean(dim=(2, 3)) + 1e-6).sqrt().unsqueeze(0)})
        # patch statistics
        for ps in self.patch_sizes:
            xp = rearrange(x, "b (h ph) (w pw) d -> b (h w) (ph pw) d", ph=ps, pw=ps)
            stat = torch.cat([xp.mean(dim=-2), xp.std(dim=-2)], dim=1).permute(1, 0, 2)
            results.update({f"patch{ps}": stat.contiguous()})
        return results
