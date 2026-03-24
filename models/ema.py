from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay

        self.ema_model = deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float = None):
        decay = self.decay if decay is None else decay
        model_params = OrderedDict(model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        for name, param in model_params.items():
            if param.requires_grad:
                ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)
            else:
                ema_params[name].copy_(param.data)
