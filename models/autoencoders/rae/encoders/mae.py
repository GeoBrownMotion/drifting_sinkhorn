import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining


class MAEEncoder(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.model = ViTMAEForPreTraining.from_pretrained(model_name).vit
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.model.config.mask_ratio = 0. # no masking

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

    def forward(self, images):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and width of the image
        """
        h,w = images.shape[2], images.shape[3]
        patch_num = int(h * w  // self.patch_size ** 2)
        assert patch_num * self.patch_size ** 2 == h * w, 'image size should be divisible by patch size'
        noise = torch.arange(patch_num).unsqueeze(0).expand(images.shape[0],-1).to(images.device).to(images.dtype)
        outputs = self.model(images, noise, interpolate_pos_encoding = True)
        image_features = outputs.last_hidden_state[:, 1:] # remove cls token
        return image_features
