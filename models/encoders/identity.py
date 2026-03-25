import torch.nn as nn
from torch import Tensor


class IdentityEncoder(nn.Module):
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {"identity": x.flatten(1).unsqueeze(0)}
