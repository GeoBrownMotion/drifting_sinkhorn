import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel


class Dinov2Encoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = Dinov2WithRegistersModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        self.encoder.layernorm.elementwise_affine = False  # noqa
        self.encoder.layernorm.weight = None  # noqa
        self.encoder.layernorm.bias = None  # noqa

    @property
    def input_size(self) -> int:
        return 224

    @property
    def patch_size(self):
        return self.encoder.config.patch_size

    @property
    def patch_num(self) -> int:
        assert self.input_size % self.patch_size == 0
        return self.input_size // self.patch_size

    @property
    def base_patches(self) -> int:
        return self.patch_num * self.patch_num

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features
