import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.functional import normalize
from einops import rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", bf16: bool = True):
        super().__init__()
        self.model_name = model_name
        self.bf16 = bf16

        self.encoder = self.load_encoder()

    def load_encoder(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="XFormers is not available*")
            encoder = torch.hub.load("facebookresearch/dinov2", self.model_name, verbose=False)
        encoder.eval()
        return encoder

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        x = torch.nn.functional.interpolate(x, (224, 224), mode="bicubic")
        return x

    @staticmethod
    def process(z: Tensor) -> Tensor:
        B, H, W, D = z.shape
        zg = z.reshape(B, H * W, D)
        z2 = rearrange(z, "b (h ph) (w pw) d -> b (h w) (ph pw) d", ph=2, pw=2)
        z4 = rearrange(z, "b (h ph) (w pw) d -> b (h w) (ph pw) d", ph=4, pw=4)
        features = torch.cat([
            zg.permute(1, 0, 2),                                                    # per location vector
            torch.stack([zg.mean(dim=1), zg.std(dim=1)], dim=0),                    # global mean and std
            torch.cat([z2.mean(dim=-2), z2.std(dim=-2)], dim=1).permute(1, 0, 2),   # patch 2x2 mean and std
            torch.cat([z4.mean(dim=-2), z4.std(dim=-2)], dim=1).permute(1, 0, 2),   # patch 4x4 mean and std
        ], dim=0)
        return features

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = {
            "x": x.flatten(1).unsqueeze(0),
            "xnorm": ((x ** 2).mean(dim=(2, 3)) + 1e-6).sqrt().unsqueeze(0)
        }
        # extract features
        z = self.preprocess(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            z = self.encoder.forward_features(z)
        z = z["x_norm_patchtokens"].float()
        # process features
        B, L, D = z.shape
        z = z.reshape(B, 16, 16, D)
        features.update({"feat": self.process(z)})
        return features
