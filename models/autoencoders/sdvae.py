import torch.nn as nn
from torch import Tensor
from diffusers import AutoencoderKL


class SDVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").eval()

    def encode(self, x: Tensor) -> Tensor:
        z = self.vae.encode(x).latent_dist.sample()
        z = z * 0.18215
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / 0.18215
        x = self.vae.decode(z).sample
        return x
