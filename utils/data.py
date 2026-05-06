import os
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.datasets as dset
from torch import Tensor
from torch.utils.data import Dataset

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

        elif name == "swissroll":
            u = torch.rand(n, generator=g)
            t = 0.5 * math.pi + 4.0 * math.pi * u
            pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
            pts = pts / (pts.abs().max() + 1e-8)
            pts = pts + 0.03 * torch.randn(pts.shape, generator=g)

        elif name == "moons":
            u = torch.rand(n, generator=g)
            t = math.pi * u
            pts1 = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
            pts2 = torch.stack([1 - torch.cos(t), 1 - torch.sin(t) - 0.5], dim=1)
            pts = torch.cat([pts1, pts2], dim=0)
            pts = pts[torch.randperm(pts.shape[0], generator=g)]
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
        self.dataset = dset.MNIST(root, train=True, transform=transform)
        self.labels = self.dataset.targets.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class CIFAR10(Dataset):
    def __init__(self, root: str, image_size: int, subset_size: int = None):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        full = dset.CIFAR10(root, train=True, transform=transform)
        if subset_size is not None and subset_size < len(full):
            self.dataset = torch.utils.data.Subset(full, list(range(subset_size)))
            self.labels = list(full.targets[:subset_size])
        else:
            self.dataset = full
            self.labels = full.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class CelebA(Dataset):
    def __init__(self, root: str, image_size: int, pflip: float = 0.5):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(pflip),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.dataset = dset.CelebA(root, split="train", transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image}


class FFHQ(Dataset):
    def __init__(self, root: str, image_size: int, pflip: float = 0.5):
        self.root = os.path.expanduser(root)
        self.image_paths = list(sorted(glob.glob(os.path.join(self.root, "*.png"))))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(pflip),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {"index": index, "image": image}


class AFHQ(Dataset):
    def __init__(self, root: str, image_size: int, pflip: float = 0.5):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(pflip),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.dataset = dset.ImageFolder(root=root, transform=transform)
        self.labels = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class ImageNet(Dataset):
    def __init__(self, root: str, image_size: int, pflip: float = 0.5):
        transform = T.Compose([
            T.Lambda(lambda image: center_crop_arr(image, image_size)),
            T.RandomHorizontalFlip(pflip),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.dataset = dset.ImageFolder(root=root, transform=transform)
        self.labels = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return {"index": index, "image": image, "label": label}


class LatentDataset(Dataset):
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        with open(os.path.join(self.root, "metadata.jsonl"), "r") as f:
            self.metadata = [json.loads(line) for line in f]
        self.labels = [int(item["label"]) for item in self.metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        metadata = self.metadata[index]
        file = np.load(metadata["file"])
        latent = torch.from_numpy(file["latent"]).float()
        label = metadata["label"]
        return {"index": index, "image": latent, "label": label}


class C2IDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.labels
        self.indices_unc = torch.randperm(len(dataset)).tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        index_unc = self.indices_unc[index]
        data = self.dataset[index]
        data_unc = self.dataset[index_unc]
        return {**data, "image_unc": data_unc["image"]}
