import torch
import torch.nn.functional as F
from torch import Tensor

from models.encoders.base import BaseEncoder


class DownsampleEncoder(BaseEncoder):
    def __init__(self, down_ratios: list[int]):
        super().__init__()
        self.down_ratios = down_ratios

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        # downsample
        H, W = x.shape[-2:]
        for r in self.down_ratios:
            h, w = H // r, W // r
            x_down = F.interpolate(x, size=(h, w), mode="bilinear")
            results.update({f"down{r}": x_down.flatten(1).unsqueeze(0)})
        return results


class BlurEncoder(BaseEncoder):
    def __init__(self, kernel_sizes: list[int] = (5, )):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.kernels = {ks: torch.ones(3, 1, ks, ks) / (ks * ks) for ks in kernel_sizes}

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        # blurring
        for ks in self.kernel_sizes:
            kernel = self.kernels[ks].to(x.device)
            x_blurred = F.conv2d(x, kernel, padding=ks // 2, groups=x.shape[1])
            results.update({f"blur{ks}": x_blurred.flatten(1).unsqueeze(0)})
        return results


class RandomDroppingEncoder(BaseEncoder):
    def __init__(self, drop_ratios: list[float]):
        super().__init__()
        self.drop_ratios = drop_ratios

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # get seed
        seed = kwargs["seed"]
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        # random dropping
        for dr in self.drop_ratios:
            mask = torch.rand(x.shape[1:], generator=generator, device=x.device) > dr
            results.update({f"drop{dr}": x[:, mask].unsqueeze(0)})
        return results


class RandomSlicingEncoder(BaseEncoder):
    def __init__(self, num_slices: int = 1000, store_input: bool = True):
        super().__init__()
        self.num_slices = num_slices
        self.store_input = store_input

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # get seed
        seed = kwargs["seed"]
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)} if self.store_input else {}
        # random slicing
        v = torch.randn((self.num_slices, *x.shape[1:]), generator=generator, device=x.device).flatten(1)  # (M, D)
        x_sliced = torch.sum(v.unsqueeze(1) * x.flatten(1).unsqueeze(0), dim=-1, keepdim=True)             # (M, N, 1)
        results.update({"sliced": x_sliced})
        return results


class RandomProjectionEncoder(BaseEncoder):
    def __init__(self, num: int, dim: int, store_input: bool = True):
        super().__init__()
        self.num = num
        self.dim = dim
        self.store_input = store_input

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # get seed
        seed = kwargs["seed"]
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)} if self.store_input else {}
        # random projection
        x_flat = x.flatten(1)                                                                         # (N, D)
        v = torch.randn((self.num, x_flat.shape[1], self.dim), generator=generator, device=x.device)  # (M, D, d)
        v, _ = torch.linalg.qr(v)                                                                     # (M, D, d)
        x_proj = x_flat.unsqueeze(0) @ v                                                              # (M, N, d)
        results.update({"proj": x_proj})
        return results
