import torch
import torch.nn as nn
from transformers import SiglipModel


class SigLIP2Encoder(nn.Module):
    def __init__(self, model_name: str, num_tokens=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.model = SiglipModel.from_pretrained(model_name).vision_model
        self.model.post_layernorm.elementwise_affine = False
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None

    @property
    def input_size(self) -> int:
        return 256

    @property
    def patch_size(self):
        return self.model.config.patch_size

    @property
    def patch_num(self) -> int:
        assert self.input_size % self.patch_size == 0
        return self.input_size // self.patch_size

    @property
    def base_patches(self) -> int:
        return self.patch_num * self.patch_num

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """
        outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state
        return image_features
