from torch import Tensor
from models.encoders.base import BaseEncoder


class IdentityEncoder(BaseEncoder):
    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        return {"x": x.flatten(1).unsqueeze(0)}
