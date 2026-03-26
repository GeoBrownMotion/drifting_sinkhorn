import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.functional import normalize
from einops import rearrange

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14", resolution: int = 256, bf16: bool = True):
        super().__init__()
        self.model_name = model_name
        self.resolution = resolution
        self.bf16 = bf16
        assert resolution in [64, 128, 256, 512]

        self.encoder = self.load_encoder()

    def load_encoder(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="XFormers is not available*")
            encoder = torch.hub.load("facebookresearch/dinov2", self.model_name, verbose=False)
        del encoder.head
        patch_resolution = int(16 * self.resolution / 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder.eval()
        return encoder

    def preprocess(self, x: Tensor) -> Tensor:
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        x = torch.nn.functional.interpolate(x, int(224 * self.resolution / 256), mode="bicubic")
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
        H = W = self.resolution // 16
        z = z.reshape(B, H, W, D)
        features.update({"feat": self.process(z)})
        return features
