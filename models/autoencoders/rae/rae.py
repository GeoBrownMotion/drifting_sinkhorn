import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoImageProcessor

from .encoders import Dinov2Encoder, MAEEncoder, SigLIP2Encoder
from .decoders import GeneralDecoder


CONFIGS = {
    "dinov2-base-vitxl-n08": {
        "model-cls": Dinov2Encoder,
        "model-name": "facebook/dinov2-with-registers-base",
        "norm-stat": ("nyu-visionx/RAE-collections", "stats/dinov2/wReg_base/imagenet1k/stat.pt"),
        "decoder-config": os.path.join(os.path.dirname(__file__), "decoders/configs/ViTXL"),
        "decoder-weights": ("nyu-visionx/RAE-collections", "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
    },
    "dinov2-base-vitxl": {
        "model-cls": Dinov2Encoder,
        "model-name": "facebook/dinov2-with-registers-base",
        "norm-stat": ("nyu-visionx/RAE-collections", "stats/dinov2/wReg_base/imagenet1k/stat.pt"),
        "decoder-config": os.path.join(os.path.dirname(__file__), "decoders/configs/ViTXL"),
        "decoder-weights": ("nyu-visionx/RAE-collections", "decoders/dinov2/wReg_base/ViTXL/dinov2_decoder.pt"),
    },
    "mae-base-vitxl-n08": {
        "model-cls": MAEEncoder,
        "model-name": "facebook/vit-mae-base",
        "norm-stat": ("nyu-visionx/RAE-collections", "stats/mae/base_p16/ImageNet1k/stat.pt"),
        "decoder-config": os.path.join(os.path.dirname(__file__), "decoders/configs/ViTXL"),
        "decoder-weights": ("nyu-visionx/RAE-collections", "decoders/mae/base_p16/ViTXL_n08/model.pt"),
    },
    "siglip2-base-vitxl-n08": {
        "model-cls": SigLIP2Encoder,
        "model-name": "google/siglip2-base-patch16-256",
        "norm-stat": ("nyu-visionx/RAE-collections", "stats/siglip2/base_p16_i256/ImageNet1k/stat.pt"),
        "decoder-config": os.path.join(os.path.dirname(__file__), "decoders/configs/ViTXL"),
        "decoder-weights": ("nyu-visionx/RAE-collections", "decoders/siglip2/base_p16_i256/ViTXL_n08/model.pt"),
    },
}


class RAE(nn.Module):
    def __init__(self, variant: str = "dinov2-base-vitxl-n08", normalize: bool = True):
        super().__init__()
        if variant not in CONFIGS:
            raise ValueError(f"Unknown RAE variant: {variant}")
        self.conf = CONFIGS[variant]
        self.normalize = normalize

        # load encoder
        encoder_cls = self.conf["model-cls"]
        self.encoder = encoder_cls(self.conf["model-name"])

        # load image processor
        proc = AutoImageProcessor.from_pretrained(self.conf["model-name"], use_fast=True)
        self.register_buffer("encoder_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("encoder_std", torch.tensor(proc.image_std).view(1, 3, 1, 1))

        # load normalization statistics
        if self.normalize:
            stat_path = hf_hub_download(*self.conf["norm-stat"])
            stats = torch.load(stat_path, map_location="cpu", weights_only=True)
            latent_mean = torch.tensor(0.) if stats.get("mean") is None else stats.get("mean")
            latent_var = torch.tensor(1.) if stats.get("var") is None else stats.get("var")
            self.register_buffer("latent_mean", latent_mean)
            self.register_buffer("latent_var", latent_var)

        # load decoder config
        decoder_config = AutoConfig.from_pretrained(self.conf["decoder-config"])
        decoder_config.hidden_size = self.encoder.hidden_size
        decoder_config.patch_size = 16
        decoder_config.image_size = int(decoder_config.patch_size * math.sqrt(self.encoder.base_patches))

        # load decoder
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.encoder.base_patches)
        state_dict_path = hf_hub_download(*self.conf["decoder-weights"])
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        keys = self.decoder.load_state_dict(state_dict, strict=False)
        assert len(keys.missing_keys) == 0

        # freeze parameters
        self.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 3, H, W) -> z (B, C, P, P)"""
        # resize input
        _, _, h, w = x.shape
        if h != self.encoder.input_size or w != self.encoder.input_size:
            x = F.interpolate(x, size=(self.encoder.input_size, self.encoder.input_size), mode="bicubic", align_corners=False)
        # normalize input
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.encoder_mean) / self.encoder_std
        # encode input
        z = self.encoder(x)
        # reshape latent
        b, n, c = z.shape
        p = self.encoder.patch_num
        z = z.transpose(1, 2).view(b, c, p, p)
        # normalize latent
        if self.normalize:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + 1e-5)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, C, P, P) -> x (B, 3, H, W)"""
        # denormalize latent
        if self.normalize:
            z = z * torch.sqrt(self.latent_var + 1e-5) + self.latent_mean
        # reshape latent
        b, c, h, w = z.shape
        n = h * w
        z = z.view(b, c, n).transpose(1, 2)
        # decode latent
        output = self.decoder(z, drop_cls_token=False).logits
        x = self.decoder.unpatchify(output)
        # denormalize output
        x = x * self.encoder_std + self.encoder_mean
        x = x * 2 - 1  # [0, 1] -> [-1, 1]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
