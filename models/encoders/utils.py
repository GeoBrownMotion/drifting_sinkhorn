import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


def postprocess(z: Tensor, global_stat: bool = True, patch2_stat: bool = True, patch4_stat: bool = True) -> Tensor:
    """Follow Appendix A.5 in the paper."""
    B, H, W, D = z.shape
    z = F.normalize(z, dim=-1)

    # per location vector
    zg = z.reshape(B, H * W, D)
    features = [zg.permute(1, 0, 2)]

    # global mean and std
    if global_stat:
        features.append(torch.stack([zg.mean(dim=1), zg.std(dim=1)], dim=0))

    # patch 2x2 mean and std
    if patch2_stat and H % 2 == 0 and W % 2 == 0:
        z2 = rearrange(z, "b (h ph) (w pw) d -> b (h w) (ph pw) d", ph=2, pw=2)
        features.append(torch.cat([z2.mean(dim=-2), z2.std(dim=-2)], dim=1).permute(1, 0, 2))

    # patch 4x4 mean and std
    if patch4_stat and H % 4 == 0 and W % 4 == 0:
        z4 = rearrange(z, "b (h ph) (w pw) d -> b (h w) (ph pw) d", ph=4, pw=4)
        features.append(torch.cat([z4.mean(dim=-2), z4.std(dim=-2)], dim=1).permute(1, 0, 2))

    features = torch.cat(features, dim=0)
    return features


class FeatureExtractor(nn.Module):
    """Extract specified features from a model by forward hooks."""

    def __init__(self, model: nn.Module, layers: list[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        _ = self.model(*args, **kwargs)
        return self._features
