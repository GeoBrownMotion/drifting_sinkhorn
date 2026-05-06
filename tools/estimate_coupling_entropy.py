import argparse
import json
import os
import sys

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from drifting import compute_barycentric_drift
from utils.misc import instantiate_from_config, set_seed


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--plans", type=str, default="two-sided,sinkhorn")
    parser.add_argument("--taus", type=str, default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--feature-key", type=str, default="dinov2-norm")
    parser.add_argument("--num-real", type=int, default=64)
    parser.add_argument("--num-fake", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--self-mask", type=str, default=None, choices=["none", "non_sinkhorn", "always"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true", default=False)
    return parser


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    set_seed(args.seed)

    conf_base = OmegaConf.load(args.config)
    conf_overrides = OmegaConf.from_dotlist(args.overrides)
    conf = OmegaConf.merge(conf_base, conf_overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = instantiate_from_config(conf.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.num_real,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = instantiate_from_config(conf.model).to(device).eval()
    encoder = instantiate_from_config(conf.encoder).to(device).eval()
    for module in (model, encoder):
        for param in module.parameters():
            param.requires_grad = False

    batch = next(iter(dataloader))
    x_real = batch["image"].float().to(device)
    z = torch.randn(args.num_fake, conf.model.params.in_channels, conf.model.params.input_size, conf.model.params.input_size, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
        x_fake = model(z)
        feat_real = encoder(x_real)
        feat_fake = encoder(x_fake)

    if args.feature_key not in feat_real:
        keys = ", ".join(feat_real.keys())
        raise KeyError(f"Unknown feature key {args.feature_key!r}; available keys: {keys}")

    f_real = rearrange(feat_real[args.feature_key].float(), "f n d -> f n d")
    f_fake = rearrange(feat_fake[args.feature_key].float(), "f n d -> f n d")
    plans = _split_csv(args.plans)
    taus = [float(item) for item in _split_csv(args.taus)]

    self_mask = args.self_mask or conf.drifting.get("self_mask", "none")
    normalize_feature = conf.drifting.get("normalize_feature", True)
    normalize_drift = conf.drifting.get("normalize_drift", True)
    dist_metric = conf.drifting.get("dist_metric", "l2_sq")

    print(json.dumps({
        "feature_key": args.feature_key,
        "num_real": args.num_real,
        "num_fake": args.num_fake,
        "feature_groups": f_real.shape[0],
        "feature_dim": f_real.shape[-1],
        "dist_metric": dist_metric,
        "normalize_feature": bool(normalize_feature),
        "normalize_drift": bool(normalize_drift),
        "self_mask": self_mask,
    }))

    for plan in plans:
        for tau in taus:
            _, info = compute_barycentric_drift(
                x_real=f_real,
                x_fake=f_fake,
                tau=tau,
                plan=plan,
                dist_metric=dist_metric,
                sinkhorn_iters=args.sinkhorn_iters,
                normalize_feature=bool(normalize_feature),
                normalize_drift=bool(normalize_drift),
                self_mask=self_mask,
            )
            print(json.dumps({
                "plan": plan,
                "tau": tau,
                "pos_eff_neighbors": info["pos-eff-neighbors"],
                "neg_eff_neighbors": info["neg-eff-neighbors"],
                "pos_row_mae": info["pos-row-mae"],
                "neg_row_mae": info["neg-row-mae"],
                "pos_col_mae": info.get("pos-col-mae"),
                "neg_col_mae": info.get("neg-col-mae"),
                "vnorm2": info[f"Vnorm2-tau{tau}"],
            }))


if __name__ == "__main__":
    main()
