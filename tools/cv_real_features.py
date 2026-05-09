"""
Measure CV (std/mean) of pairwise distances in REAL DINOv2 feature space on
actual CIFAR-10 images.

Compares against the random-gaussian baseline used in highdim_collapse_diag.py.
If real CIFAR DINOv2 features have higher CV than random gaussian (because of
semantic clustering: same-class images are close, cross-class are far), then
the paper's "sinkhorn-beats-two-sided" mechanism may still activate in our
setup despite high dim. If CV is similar to gaussian (~2.5%), the mechanism is
truly killed by concentration of measure and we need lower-dim features.

For each feature stream output by our DINOv2Encoder, reports:
  - mean, std, CV of pairwise L2 distances
  - logit spread at eps=0.02 (= -CV / eps in normalized units)
"""
import math
import torch
from omegaconf import OmegaConf

from utils.misc import set_seed, instantiate_from_config


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = OmegaConf.load("configs/cifar10-unc-split-screening.yaml")
    encoder = instantiate_from_config(conf.encoder).to(device).eval()

    # Lazy-build a small CIFAR loader (1024 samples, no DDP).
    dataset = instantiate_from_config(conf.data)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[:1024]
    images = torch.stack([dataset[i.item()]["image"] for i in indices], dim=0).to(device)
    print(f"loaded {images.shape[0]} CIFAR-10 images, shape={tuple(images.shape)}")

    with torch.no_grad():
        from models.autoencoders.identity import IdentityAutoencoder
        ae = IdentityAutoencoder().to(device).eval()
        feat = encoder(images, autoencoder=ae)  # dict, each (F, N, D)

    print(f"\n{'stream':<25} {'F':>3} {'N':>5} {'D':>5} {'d.mean':>10} {'d.std':>10} {'CV(d)':>8} {'logit_std/eps':>14}")
    print("-" * 95)

    for name, f in feat.items():
        F, N, D = f.shape
        # Use the first feature index (F=1 usually for our streams)
        x = f[0].float()  # (N, D)
        d = torch.cdist(x, x)  # (N, N)
        # Exclude diagonal (= 0)
        mask = ~torch.eye(N, dtype=torch.bool, device=d.device)
        d_off = d[mask]
        d_mean = d_off.mean().item()
        d_std = d_off.std().item()
        cv = d_std / max(d_mean, 1e-12)
        eps = 0.02
        # logit = -d^2/eps; std of logit ≈ |d^2|.std/eps  ≈  2*d_mean*d_std/eps
        logit_std_at_eps_normalized = (2.0 * 1.0 * cv / eps)  # in normalized units (d_mean→1)
        print(f"{name:<25} {F:>3} {N:>5} {D:>5} {d_mean:>10.4f} {d_std:>10.4f} {cv*100:>7.2f}% {logit_std_at_eps_normalized:>14.2f}")

    # Random gaussian baseline at the same shapes for comparison
    print(f"\n{'random_gaussian_768D':<25} {'-':>3} {'-':>5} {'-':>5}", end="")
    x = torch.randn(1024, 768, device=device)
    d = torch.cdist(x, x)
    d_off = d[~torch.eye(1024, dtype=torch.bool, device=d.device)]
    d_mean = d_off.mean().item(); d_std = d_off.std().item()
    cv = d_std / d_mean
    print(f" {d_mean:>10.4f} {d_std:>10.4f} {cv*100:>7.2f}% {(2.0 * cv / 0.02):>14.2f}")

    print("\nReading:")
    print("  CV ~ 2.5% (random gaussian) → logit_std/eps ≈ 2.5  → plan stays soft (no collapse)")
    print("  CV ~ 15-20% (paper mnist 32D) → logit_std/eps ≈ 15-20  → two-sided collapses, sinkhorn wins")
    print("  Real DINOv2 CIFAR: see above. If significantly > gaussian, mechanism may still activate.")


if __name__ == "__main__":
    main()
