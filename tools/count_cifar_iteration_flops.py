import argparse
import json
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.complexity import estimate_drift_flops, make_flops_report, split_csv


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate per-iteration CIFAR drifting FLOPs from config shapes."
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--plans", type=str, default="two-sided,sinkhorn")
    parser.add_argument("--num-real", type=int, default=None)
    parser.add_argument("--num-fake", type=int, default=None)
    parser.add_argument("--sinkhorn-iters", type=int, default=None)
    parser.add_argument("--dinov2-dim", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    conf = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    results = estimate_drift_flops(
        conf,
        plans=split_csv(args.plans),
        num_real=args.num_real,
        num_fake=args.num_fake,
        sinkhorn_iters=args.sinkhorn_iters,
        dinov2_dim=args.dinov2_dim,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(make_flops_report(results))


if __name__ == "__main__":
    main()
