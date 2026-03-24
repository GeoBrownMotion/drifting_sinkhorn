import math
import torch.nn as nn


def get_param_groups(model: nn.Module, weight_decay: float = 0, skip_list: list = ()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.},
        {"params": decay, "weight_decay": weight_decay}
    ]


def get_actual_lr(lr: float, batch_size: int, scale_lr: str = "linear"):
    if scale_lr == "linear":
        scale = batch_size / 256
        actual_lr = lr * scale
    elif scale_lr == "sqrt":
        scale = math.sqrt(batch_size / 256)
        actual_lr = lr * scale
    elif scale_lr == "none":
        actual_lr = lr
    else:
        raise ValueError(f"Invalid lr scaling rule: {scale_lr}")
    return actual_lr
