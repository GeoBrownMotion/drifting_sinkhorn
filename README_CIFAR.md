# CIFAR-10 Rebuttal Experiments

This branch contains the CIFAR-10 minibatch drifting experiments used for the rebuttal.
The main comparison is:

- **Sinkhorn drift**: `drifting.mode=barycentric`, `drifting.plan=sinkhorn`
- **Baseline**: `drifting.mode=barycentric`, `drifting.plan=two-sided`

The baseline here is the two-sided barycentric control for the Sinkhorn experiment. It is
not the original multi-temperature mutual-softmax baseline from `drifting-models-pytorch`.

## Clone

```bash
git clone https://github.com/GeoBrownMotion/drifting_sinkhorn.git
cd drifting_sinkhorn
git checkout cifar_v1
```

## Environment

The experiments were tested with Python 3.12 and PyTorch 2.6.0.

```bash
conda create -n rebuttal-cifar-baseline python=3.12 -y
conda activate rebuttal-cifar-baseline

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The DINOv2 encoder is loaded through `torch.hub`, so the first run needs either internet
access or a populated local torch hub cache.

## Data

By default, the config points to:

```text
/home/hep3/rebuttal/data/cifar10
```

On another machine, either edit `configs/cifar10-unc-pixel-dinov2-unet-barycentric.yaml`
or override the path at launch time:

```bash
--set data.params.root=/path/to/cifar10
```

The dataset class uses `torchvision.datasets.CIFAR10(download=True)`, so it can download
CIFAR-10 automatically if the root path is writable.

## Recommended 4-GPU Sinkhorn Run

This script runs Sinkhorn only. Run the two-sided baseline separately with the matched
command below, ideally on another machine.

```bash
RUN_ROOT=/path/to/runs \
STAMP=$(date +%Y%m%d_%H%M%S) \
tools/launch_cifar25k_tau0p5_batch640_sinkhorn.sh
```

Default settings in the launcher:

- `num_steps=25000`
- global `num_real_samples=640`
- global `num_fake_samples=640`
- `tau=0.5`
- `sinkhorn_iters=30`
- `save_freq=2500`
- `sample_freq=2500`
- `bf16`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`

The script also sets:

```bash
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

These flags were needed on our 4-GPU A5000 machine to avoid NCCL/PyTorch memory issues.

## Run Sinkhorn Only

```bash
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run -n rebuttal-cifar-baseline --no-capture-output \
torchrun --standalone --nproc-per-node 4 train_unc.py \
  -c configs/cifar10-unc-pixel-dinov2-unet-barycentric.yaml \
  --bf16 \
  --set train.num_steps=25000 \
  --set train.print_freq=100 \
  --set train.save_freq=2500 \
  --set train.sample_freq=2500 \
  --set dataloader.num_workers=4 \
  --set train.num_real_samples=640 \
  --set train.num_fake_samples=640 \
  --set drifting.tau=0.5 \
  --set drifting.sinkhorn_iters=30 \
  --set drifting.plan=sinkhorn \
  -e /path/to/runs/cifar10_bary_sinkhorn_tau0p5_batch640_25k
```

## Run Two-Sided Baseline Only

```bash
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run -n rebuttal-cifar-baseline --no-capture-output \
torchrun --standalone --nproc-per-node 4 train_unc.py \
  -c configs/cifar10-unc-pixel-dinov2-unet-barycentric.yaml \
  --bf16 \
  --set train.num_steps=25000 \
  --set train.print_freq=100 \
  --set train.save_freq=2500 \
  --set train.sample_freq=2500 \
  --set dataloader.num_workers=4 \
  --set train.num_real_samples=640 \
  --set train.num_fake_samples=640 \
  --set drifting.tau=0.5 \
  --set drifting.sinkhorn_iters=30 \
  --set drifting.plan=two-sided \
  -e /path/to/runs/cifar10_bary_two_sided_tau0p5_batch640_25k
```

## Outputs

Each run writes:

- `output.log`: training logs and coupling statistics
- `samples/step*.jpg`: generated image grids every 2.5K steps
- `ckpt/step*/`: checkpoints every 2.5K steps
- `complexity.json`: per-iteration drift FLOPs estimate
- `complexity.md`: readable FLOPs and asymptotic complexity report

For the default batch-640 setting, the drift FLOPs estimate is:

- two-sided: `2.134 TFLOPs / iteration`
- Sinkhorn, 30 iterations: `2.334 TFLOPs / iteration`
- Sinkhorn drift overhead vs two-sided: `1.094x`

This estimate covers the drifting-field computation only. It does not include UNet or DINOv2
forward/backward FLOPs.

## Monitoring

```bash
tail -f /path/to/runs/<run_name>/output.log
ls /path/to/runs/<run_name>/samples
watch -n 10 nvidia-smi
```

The first sample and checkpoint appear at step `2499`.

## Quick Smoke Test

Use this to verify that the code path, DINOv2 loading, FLOPs report, and DDP launch work.

```bash
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
conda run -n rebuttal-cifar-baseline --no-capture-output \
torchrun --standalone --nproc-per-node 1 train_unc.py \
  -c configs/cifar10-unc-pixel-dinov2-unet-barycentric.yaml \
  --bf16 \
  --set data.params.subset_size=16 \
  --set train.num_steps=1 \
  --set train.print_freq=1 \
  --set train.save_freq=100000 \
  --set train.sample_freq=100000 \
  --set train.num_real_samples=4 \
  --set train.num_fake_samples=4 \
  --set dataloader.num_workers=1 \
  --set train.sched.params.warmup_steps=1 \
  --set train.sched.params.training_steps=1 \
  --set drifting.tau=0.5 \
  --set drifting.sinkhorn_iters=30 \
  --set drifting.plan=sinkhorn \
  -e /path/to/runs/cifar10_smoke
```

## Notes

- Larger minibatches are preferred because the method effectively solves a minibatch OT
  problem. On our current 4xA5000 setup, global batch 640 is the largest clean setting with
  DINOv2 features.
- The 64-image overfit runs are only sanity checks for the code path. They are not reliable
  indicators of final CIFAR-10 sample quality.
