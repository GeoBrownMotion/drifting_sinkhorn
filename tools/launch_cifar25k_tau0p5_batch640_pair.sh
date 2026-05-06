#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/hep3/rebuttal/runs}"
CONFIG="configs/cifar10-unc-pixel-dinov2-unet-barycentric.yaml"

cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  -c "${CONFIG}"
  --bf16
  --set train.num_steps=25000
  --set train.print_freq=100
  --set train.save_freq=2500
  --set train.sample_freq=2500
  --set dataloader.num_workers=4
  --set train.num_real_samples=640
  --set train.num_fake_samples=640
  --set drifting.tau=0.5
  --set drifting.sinkhorn_iters=30
)

run_train() {
  local plan="$1"
  local exp_dir="$2"
  conda run -n rebuttal-cifar-baseline --no-capture-output \
    torchrun --standalone --nproc-per-node 4 train_unc.py \
    "${COMMON_ARGS[@]}" \
    -e "${exp_dir}" \
    --set drifting.plan="${plan}"
}

run_train "sinkhorn" "${RUN_ROOT}/cifar10_bary_sinkhorn_tau0p5_batch640_25k_${STAMP}"
run_train "two-sided" "${RUN_ROOT}/cifar10_bary_two_sided_tau0p5_batch640_25k_${STAMP}"
