"""FLOPs profiling for one CIFAR-10 drift training step.

Reports forward + backward FLOPs per step, broken down into:
  - Encoder fwd (no_grad on real, with grad on fake)
  - Generator fwd + bwd (UNet)
  - Drift compute (mutual-softmax / two-sided split / Sinkhorn iterations)

For Sinkhorn T-ablation, override sinkhorn_iters on the command line:
    python tools/flop_count.py -c configs/cifar10-unc-split-sinkhorn.yaml --batch 256 \
        --set drifting.sinkhorn_iters=20

For an apples-to-apples baseline number, point at the original config
(mutual-softmax / kernel_temp list).

Notes
-----
* Single-GPU profile by default. Set --batch to the per-rank batch size that
  the actual DDP training uses (e.g. B=512 for 4-rank DDP at total B=2048).
  At too-large per-rank batches the encoder fwd+bwd OOMs on a single A6000;
  in that case, profile at a smaller batch and the per-step number scales
  cleanly (encoder is ~linear, drift is ~quadratic in batch).
* Drift compute is wrapped in torch.no_grad in train_unc.py, so its backward
  graph is not built; we report drift FLOPs as forward-only by design.
* Backward FLOPs of UNet/encoder are counted automatically by FlopCounterMode
  during loss.backward().
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

# Make the project root importable when running this file from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drifting import compute_drift  # noqa: E402
from drifting_split import compute_drift_split  # noqa: E402
from models.autoencoders.identity import IdentityAutoencoder  # noqa: E402
from utils.misc import instantiate_from_config  # noqa: E402


def _format_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MFLOPs"
    return f"{n} FLOPs"


def call_drift(conf, f_real, f_fake):
    method = conf.drifting.get("method", "joint")
    if method == "split":
        return compute_drift_split(
            x_real=f_real.detach(),
            x_fake=f_fake.detach(),
            eps=conf.drifting.eps,
            plan_type=conf.drifting.plan_type,
            sinkhorn_iters=conf.drifting.get("sinkhorn_iters", 20),
            dist_metric=conf.drifting.get("dist_metric", "l2_sq"),
            normalize_feature=conf.drifting.normalize_feature,
            normalize_drift=conf.drifting.get("normalize_drift", False),
        )
    return compute_drift(
        x_real=f_real.detach(),
        x_fake=f_fake.detach(),
        kernel_temp=conf.drifting.kernel_temp,
        kernel_norm=conf.drifting.kernel_norm,
        normalize_feature=conf.drifting.normalize_feature,
        normalize_drift=conf.drifting.normalize_drift,
    )


def run_step(conf, model, autoencoder, encoder, z, x_real, *, drift: bool):
    """One forward (+ optional drift) + backward step. Returns the loss tensor."""
    x_fake = model(z)
    with torch.no_grad():
        feat_real = encoder(x_real, autoencoder=autoencoder)
    feat_fake = encoder(x_fake, autoencoder=autoencoder)

    Ng = conf.train.num_groups
    Nr = conf.train.num_real_samples
    Nf = conf.train.num_fake_samples
    loss = torch.tensor(0.0, device=x_fake.device)
    n_streams = 0
    for name in feat_real.keys():
        f_real = feat_real[name].float()
        f_fake = feat_fake[name].float()
        f_real = rearrange(f_real, "f (ng nr) d -> (f ng) nr d", ng=Ng, nr=Nr)
        f_fake = rearrange(f_fake, "f (ng nf) d -> (f ng) nf d", ng=Ng, nf=Nf)
        if drift:
            with torch.no_grad():
                V, info = call_drift(conf, f_real, f_fake)
            f_fake = f_fake / max(info["data-scale"], 1e-3)
            loss = loss + F.mse_loss(f_fake, (f_fake + V).detach())
        else:
            loss = loss + f_fake.float().pow(2).mean()
        n_streams += 1
    if n_streams > 0:
        loss = loss / n_streams
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Per-rank batch (overrides num_real_samples and num_fake_samples). "
             "For 4-GPU DDP at total B=2048, pass --batch 512.",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="OmegaConf overrides, e.g. drifting.sinkhorn_iters=20")
    parser.add_argument("--out", default=None, help="Optional path to dump JSON results")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 autocast")
    parser.add_argument("--time-iters", type=int, default=10,
                        help="Number of full-step iterations to time for wall-clock measurement")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))
    if args.batch is not None:
        conf.train.num_real_samples = args.batch
        conf.train.num_fake_samples = args.batch

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16)

    method = conf.drifting.get("method", "joint")
    print(f"[setup] config: {args.config}")
    print(f"[setup] drift method: {method}")
    if method == "split":
        print(f"[setup] plan_type: {conf.drifting.plan_type}, eps: {conf.drifting.eps}, "
              f"sinkhorn_iters: {conf.drifting.get('sinkhorn_iters', 20)}")
    else:
        print(f"[setup] kernel_norm: {conf.drifting.kernel_norm}, "
              f"kernel_temp: {conf.drifting.kernel_temp}")
    print(f"[setup] num_real_samples: {conf.train.num_real_samples}, "
          f"num_fake_samples: {conf.train.num_fake_samples}")

    # Build modules ----------------------------------------------------------
    model = instantiate_from_config(conf.model).to(device)
    if hasattr(conf, "autoencoder"):
        autoencoder = instantiate_from_config(conf.autoencoder).to(device).eval()
    else:
        autoencoder = IdentityAutoencoder().to(device).eval()
    encoder = instantiate_from_config(conf.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    n_unet = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"[setup] UNet params: {n_unet/1e6:.1f}M, Encoder params: {n_enc/1e6:.1f}M")

    # Inputs -----------------------------------------------------------------
    in_channels = conf.model.params.in_channels
    input_size = conf.model.params.input_size
    input_shape = (in_channels, input_size, input_size)
    Ng = conf.train.num_groups
    Nr = conf.train.num_real_samples
    Nf = conf.train.num_fake_samples
    z = torch.randn(Ng * Nf, *input_shape, device=device)
    x_real = torch.randn(Ng * Nr, *input_shape, device=device)

    # Warmup so cuDNN benchmarks settle before the FLOPs measurement
    print("[warmup] Running 1 step (no FlopCounter) to settle cuDNN ...")
    with autocast_ctx:
        loss = run_step(conf, model, autoencoder, encoder, z, x_real, drift=True)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure: drift-only (forward, no_grad context)
    print("[measure] drift-only forward FLOPs ...")
    with torch.no_grad(), autocast_ctx:
        x_fake = model(z)
        feat_real = encoder(x_real, autoencoder=autoencoder)
        feat_fake = encoder(x_fake, autoencoder=autoencoder)

    drift_counter = FlopCounterMode(display=False)
    with drift_counter, torch.no_grad():
        for name in feat_real.keys():
            f_real = feat_real[name].float()
            f_fake = feat_fake[name].float()
            f_real = rearrange(f_real, "f (ng nr) d -> (f ng) nr d", ng=Ng, nr=Nr)
            f_fake = rearrange(f_fake, "f (ng nf) d -> (f ng) nf d", ng=Ng, nf=Nf)
            _ = call_drift(conf, f_real, f_fake)
    drift_flops = drift_counter.get_total_flops()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Measure: full step (encoder + generator + drift + backward)
    print("[measure] full step (fwd + bwd) FLOPs ...")
    full_counter = FlopCounterMode(display=False)
    with full_counter:
        with autocast_ctx:
            loss = run_step(conf, model, autoencoder, encoder, z, x_real, drift=True)
        loss.backward()
    full_flops = full_counter.get_total_flops()
    torch.cuda.synchronize()

    per_module = {k: sum(v.values()) for k, v in full_counter.flop_counts.items()}
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Wall-clock timing (the FlopCounter does not count the logsumexp / exp /
    # softmax that dominate Sinkhorn iterations, so wall-clock is the metric
    # that actually answers reviewers about cost).
    print(f"[measure] Wall-clock timing over {args.time_iters} full steps ...")
    full_step_ms: list[float] = []
    drift_ms: list[float] = []
    # Re-run the encoder fwd outside the timed region for the drift-only timing.
    for _ in range(args.time_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast_ctx:
            loss = run_step(conf, model, autoencoder, encoder, z, x_real, drift=True)
        loss.backward()
        torch.cuda.synchronize()
        full_step_ms.append((time.perf_counter() - t0) * 1000.0)
        model.zero_grad(set_to_none=True)

    # Drift-only timing: features prebuilt, just call drift on each stream.
    with torch.no_grad(), autocast_ctx:
        x_fake = model(z)
        feat_real = encoder(x_real, autoencoder=autoencoder)
        feat_fake = encoder(x_fake, autoencoder=autoencoder)
    for _ in range(args.time_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for name in feat_real.keys():
                f_real = feat_real[name].float()
                f_fake = feat_fake[name].float()
                f_real = rearrange(f_real, "f (ng nr) d -> (f ng) nr d", ng=Ng, nr=Nr)
                f_fake = rearrange(f_fake, "f (ng nf) d -> (f ng) nf d", ng=Ng, nf=Nf)
                _ = call_drift(conf, f_real, f_fake)
        torch.cuda.synchronize()
        drift_ms.append((time.perf_counter() - t0) * 1000.0)

    full_med = statistics.median(full_step_ms)
    drift_med = statistics.median(drift_ms)

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Per-step cost (per-rank, batch={})".format(Nf))
    print("=" * 60)
    print("Wall-clock (median of {} runs):".format(args.time_iters))
    print(f"  Full step (fwd+bwd):    {full_med:7.1f} ms")
    print(f"  Drift compute (fwd):    {drift_med:7.1f} ms  "
          f"({100 * drift_med / max(full_med, 1e-3):.1f}% of full step)")
    print(f"  Non-drift:              {full_med - drift_med:7.1f} ms")
    print()
    print("FLOPs (matmul+conv only; logsumexp/softmax/exp NOT counted by FlopCounterMode):")
    print(f"  Total (fwd+bwd):        {_format_flops(full_flops)}")
    print(f"  Drift (fwd):            {_format_flops(drift_flops)}")
    print(f"Peak GPU mem (this run):  {peak_mem:.2f} GB")

    print("\nPer top-level module (forward+backward attribution):")
    for k, v in sorted(per_module.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {k:<50s} {_format_flops(v)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": str(args.config),
            "overrides": list(args.overrides),
            "batch_per_rank": Nf,
            "drift_method": conf.drifting.get("method", "joint"),
            "drift_plan_type": conf.drifting.get("plan_type"),
            "drift_eps": conf.drifting.get("eps"),
            "drift_sinkhorn_iters": conf.drifting.get("sinkhorn_iters"),
            "drift_kernel_norm": conf.drifting.get("kernel_norm"),
            "drift_kernel_temp": list(conf.drifting.get("kernel_temp", [])) or None,
            "full_step_ms_median": full_med,
            "drift_ms_median": drift_med,
            "drift_pct_of_step_walltime": 100.0 * drift_med / max(full_med, 1e-3),
            "full_step_ms_all": full_step_ms,
            "drift_ms_all": drift_ms,
            "full_step_flops": full_flops,
            "drift_flops": drift_flops,
            "non_drift_flops": full_flops - drift_flops,
            "drift_overhead_pct_flops": 100.0 * drift_flops / max(full_flops, 1),
            "peak_gpu_mem_gb": peak_mem,
            "per_module_flops": per_module,
            "_caveat": (
                "FlopCounterMode counts matmul/conv only. Sinkhorn iter ops "
                "(logsumexp/exp/sub) are NOT counted here. Use full_step_ms_median "
                "and drift_ms_median for actual cost comparisons across T values."
            ),
        }, indent=2))
        print(f"\n[json] saved to {out_path}")


if __name__ == "__main__":
    main()
