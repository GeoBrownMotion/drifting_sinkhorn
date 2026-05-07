"""Auto FID curve: watches an exp dir, computes FID/IS/KID for each new ckpt.

Usage:
    python tools/fid_watch.py \
        --exp-dir runs/full_sinkhorn_b2048_tau0p5 \
        --config configs/cifar10-unc-split-sinkhorn.yaml \
        --num-samples 10000 \
        --poll-secs 300

The script:
    1. Loads existing entries from <exp-dir>/fid_curve.json (if any).
    2. Lists all ckpt dirs under <exp-dir>/ckpt/ matching step*.
    3. For each ckpt not yet in fid_curve.json:
         - Generates --num-samples samples (CPU) into
           <exp-dir>/samples_step{step:07d}_{N}k/
         - Runs `fidelity --fid --isc --kid` against cifar10-train.
         - Appends a row to fid_curve.json (sorted by step).
    4. Repeats every --poll-secs (use --once to disable polling).

CPU-only by default (CUDA_VISIBLE_DEVICES="") so it does not contend with
an active training run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_fidelity_output(text: str) -> dict:
    patterns = {
        "fid": r"frechet_inception_distance:\s*([\d.eE+\-]+)",
        "is_mean": r"inception_score_mean:\s*([\d.eE+\-]+)",
        "is_std": r"inception_score_std:\s*([\d.eE+\-]+)",
        "kid_mean": r"kernel_inception_distance_mean:\s*([\d.eE+\-]+)",
        "kid_std": r"kernel_inception_distance_std:\s*([\d.eE+\-]+)",
    }
    out: dict = {}
    for k, p in patterns.items():
        m = re.search(p, text)
        out[k] = float(m.group(1)) if m else None
    return out


def step_from_ckpt_dir(p: Path) -> int | None:
    m = re.search(r"step(\d+)", p.name)
    return int(m.group(1)) if m else None


def count_pngs(sample_dir: Path) -> int:
    if not sample_dir.is_dir():
        return 0
    return len(list(sample_dir.glob("*.png")))


def already_sampled(sample_dir: Path, n: int) -> bool:
    return count_pngs(sample_dir) >= n


def ensure_sampled(
    *,
    config: str,
    weights: Path,
    save_dir: Path,
    num_samples: int,
    bspp: int,
    max_attempts: int = 2,
) -> tuple[bool, str]:
    """Try up to max_attempts times to populate save_dir with num_samples PNGs.
    Wipes partial dirs between attempts. Returns (ok, last_err)."""
    for attempt in range(1, max_attempts + 1):
        if save_dir.is_dir():
            n = count_pngs(save_dir)
            if n >= num_samples:
                return True, ""
            print(f"  attempt {attempt}: existing dir has only {n}/{num_samples} PNGs, wiping...", flush=True)
            shutil.rmtree(save_dir, ignore_errors=True)
        ok, err = run_sampling(
            config=config, weights=weights, save_dir=save_dir,
            num_samples=num_samples, bspp=bspp,
        )
        if not ok:
            return False, f"attempt {attempt}: subprocess failed: {err}"
        n = count_pngs(save_dir)
        if n >= num_samples:
            return True, ""
        print(f"  attempt {attempt}: sample_unc.py reported success but only "
              f"{n}/{num_samples} PNGs in {save_dir.name}, retrying...", flush=True)
    return False, f"exhausted {max_attempts} attempts; last count={count_pngs(save_dir)}"


def run_sampling(
    *,
    config: str,
    weights: Path,
    save_dir: Path,
    num_samples: int,
    bspp: int,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "sample_unc.py",
        "-c", config,
        "-w", str(weights),
        "--save-dir", str(save_dir),
        "--num-samples", str(num_samples),
        "--bspp", str(bspp),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or "")[-3000:]
    return True, ""


def run_fidelity(*, sample_dir: Path, batch_size: int) -> tuple[bool, str, str]:
    cmd = [
        "fidelity",
        "--gpu", "",
        "--fid", "--isc", "--kid", "--kid-subset-size", "1000",
        "--input1", str(sample_dir),
        "--input2", "cifar10-train",
        "--batch-size", str(batch_size),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return False, "", (r.stderr or "")[-3000:]
    return True, r.stdout + "\n" + r.stderr, ""


def write_curve(curve: list, fid_json: Path) -> None:
    curve.sort(key=lambda x: x["step"])
    fid_json.write_text(json.dumps(curve, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True,
                    help="Experiment dir; reads ckpt/, writes fid_curve.json")
    ap.add_argument("--config", type=str, required=True,
                    help="Config used during training (for sample_unc.py)")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--bspp", type=int, default=64)
    ap.add_argument("--fidelity-batch", type=int, default=64,
                    help="Inception forward batch size for fidelity")
    ap.add_argument("--poll-secs", type=int, default=300,
                    help="Poll interval; 0 or --once to do a single pass")
    ap.add_argument("--once", action="store_true",
                    help="Run a single pass and exit")
    args = ap.parse_args()

    exp_dir: Path = args.exp_dir.resolve()
    ckpt_root = exp_dir / "ckpt"
    fid_json = exp_dir / "fid_curve.json"

    if not ckpt_root.is_dir():
        print(f"[fid_watch] ckpt root not found: {ckpt_root}", flush=True)
        sys.exit(1)

    curve: list = []
    if fid_json.is_file():
        try:
            curve = json.loads(fid_json.read_text())
        except json.JSONDecodeError:
            print(f"[fid_watch] WARN: failed to parse {fid_json}, starting fresh", flush=True)
    seen_steps = {row["step"] for row in curve}
    print(f"[fid_watch] existing entries: {sorted(seen_steps) or 'none'}", flush=True)
    print(f"[fid_watch] exp_dir: {exp_dir}", flush=True)
    print(f"[fid_watch] num_samples per ckpt: {args.num_samples}", flush=True)

    while True:
        ckpt_dirs = sorted(p for p in ckpt_root.glob("step*") if p.is_dir())
        todo = []
        for c in ckpt_dirs:
            step = step_from_ckpt_dir(c)
            if step is None or step in seen_steps:
                continue
            ema = c / "model_ema.pt"
            if not ema.is_file():
                print(f"[fid_watch] step {step}: model_ema.pt missing, skip", flush=True)
                continue
            todo.append((step, c, ema))

        if not todo:
            print(f"[fid_watch] no new ckpts; latest seen step = "
                  f"{max(seen_steps) if seen_steps else 'none'}", flush=True)
        for step, _ckpt, ema in todo:
            sample_dir = exp_dir / f"samples_step{step:07d}_{args.num_samples // 1000}k"
            t_total = time.time()

            # 1. Sampling (with completeness check + retry).
            print(f"[fid_watch] step {step}: ensure {args.num_samples} samples in {sample_dir.name}", flush=True)
            t0 = time.time()
            ok, err = ensure_sampled(
                config=args.config,
                weights=ema,
                save_dir=sample_dir,
                num_samples=args.num_samples,
                bspp=args.bspp,
                max_attempts=2,
            )
            if not ok:
                print(f"[fid_watch] step {step}: sampling FAILED: {err}", flush=True)
                continue
            print(f"[fid_watch] step {step}: sampling ok ({count_pngs(sample_dir)} PNGs) in {time.time() - t0:.1f}s", flush=True)

            # 2. Fidelity (FID + IS + KID)
            print(f"[fid_watch] step {step}: computing FID/IS/KID ...", flush=True)
            t0 = time.time()
            ok, out, err = run_fidelity(sample_dir=sample_dir, batch_size=args.fidelity_batch)
            if not ok:
                print(f"[fid_watch] step {step}: fidelity FAILED:\n{err}", flush=True)
                continue
            metrics = parse_fidelity_output(out)
            print(f"[fid_watch] step {step}: fidelity done in {time.time() - t0:.1f}s "
                  f"-> FID={metrics.get('fid')}, IS={metrics.get('is_mean')}, "
                  f"KID={metrics.get('kid_mean')}", flush=True)

            row = {
                "step": step,
                "num_samples": args.num_samples,
                "fid": metrics.get("fid"),
                "is_mean": metrics.get("is_mean"),
                "is_std": metrics.get("is_std"),
                "kid_mean": metrics.get("kid_mean"),
                "kid_std": metrics.get("kid_std"),
                "wall_secs": time.time() - t_total,
            }
            curve.append(row)
            seen_steps.add(step)
            write_curve(curve, fid_json)
            print(f"[fid_watch] step {step}: appended -> {fid_json}", flush=True)

        if args.once or args.poll_secs <= 0:
            break
        print(f"[fid_watch] sleeping {args.poll_secs}s ...", flush=True)
        time.sleep(args.poll_secs)

    print(f"[fid_watch] done. {len(curve)} entries in {fid_json}", flush=True)


if __name__ == "__main__":
    main()
