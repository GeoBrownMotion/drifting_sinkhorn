import math
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST as _MNIST
from torchvision.datasets import CIFAR10 as _CIFAR10

from utils.image import center_crop_arr


class Toy2D(Dataset):
    def __init__(
            self,
            name: str,
            shift: tuple[float, float] = (0.0, 0.0),
            scale: tuple[float, float] = (1.0, 1.0),
            embed: int = None,
            seed: int = None,
    ):
        n = 65536
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        # create 2D points
        if name == "checkerboard":
            b = torch.randint(0, 2, (n,), generator=g)
            i = torch.randint(0, 2, (n,), generator=g) * 2 + b
            j = torch.randint(0, 2, (n,), generator=g) * 2 + b
            u = torch.rand(n, generator=g)
            v = torch.rand(n, generator=g)
            pts = torch.stack([i + u, j + v], dim=1) - 2.0
            pts = pts / 2.0
            pts = pts + 0.05 * torch.randn(pts.shape, generator=g)

        elif name == "swiss-roll":
            u = torch.rand(n, generator=g)
            t = 0.5 * math.pi + 4.0 * math.pi * u
            pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
            pts = pts / (pts.abs().max() + 1e-8)
            pts = pts + 0.03 * torch.randn(pts.shape, generator=g)

        elif name == "ring8":
            x = torch.randn((n // 8, 8)) * 0.04
            y = torch.randn((n // 8, 8)) * 0.04
            angles = torch.linspace(0, 1.75, 8) * math.pi
            x = (x + torch.cos(angles)).view(-1)
            y = (y + torch.sin(angles)).view(-1)
            pts = torch.stack([x, y], dim=1)

        else:
            raise ValueError(f"Unknown dataset: {name}")

        self.pts = pts * torch.tensor(scale) + torch.tensor(shift)

        # random projection
        self.Q = None
        if embed is not None:
            self.Q = torch.randn((embed, 2), generator=g)
            self.Q, _ = torch.linalg.qr(self.Q)

        # for plotting
        self.xmin, self.ymin = self.pts.min(dim=0).values
        self.xmax, self.ymax = self.pts.max(dim=0).values
        self.xmin = min(self.xmin, -1)
        self.ymin = min(self.ymin, -1)
        self.xmax = max(self.xmax, +1)
        self.ymax = max(self.ymax, +1)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, index: int):
        points = self.pts[index]
        if self.Q is not None:
            points = points @ self.Q.T
        return points

    def visualize(self, samples: Tensor, savepath: str, title: str = None):
        pts = self.pts[:5000]
        if self.Q is not None:
            samples = samples @ self.Q
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.3, c="C0")
        ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.3, c="C1")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.xmin - 0.5, self.xmax + 0.5)
        ax.set_ylim(self.ymin - 0.5, self.ymax + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)


class MNIST(Dataset):
    def __init__(self, root: str, image_size: int):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.dataset = _MNIST(root, train=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class CIFAR10(Dataset):
    def __init__(self, root: str, image_size: int):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.dataset = _CIFAR10(root, train=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class ImageNet(Dataset):
    def __init__(self, root: str, image_size: int):
        transform = T.Compose([
            T.Lambda(lambda image: center_crop_arr(image, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.dataset = ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}
