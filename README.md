# Drifting Models

## Setup

```shell
# clone the repo
git clone https://github.com/xyfJASON/progressive-drifting.git
cd progressive-drifting

# create conda environment
conda create -n drift python=3.12 -y
conda activate drift

# install dependencies
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Configs

<table>
<tr>
    <th align="left">Dataset</th>
    <th align="left">Cond.</th>
    <th align="left">Autoencoder</th>
    <th align="left">Encoder</th>
    <th align="left">Network</th>
    <th align="left">Config.</th>
</tr>
<tr>
    <td rowspan="2">MNIST 32x32</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>UNet (8.2M)</td>
    <td><a href="./configs/mnist-unc.yaml">mnist-unc</a></td>
</tr>
<tr>
    <td>class</td>
    <td>-</td>
    <td>-</td>
    <td>UNet (9.6M)</td>
    <td><a href="./configs/mnist-c2i.yaml">mnist-c2i</a></td>
</tr>
<tr>
    <td rowspan="2">CIFAR-10 32x32</td>
    <td>-</td>
    <td>-</td>
    <td>DINOv2</td>
    <td>UNet (32.9M)</td>
    <td><a href="./configs/cifar10-unc-dinov2.yaml">cifar10-unc-dinov2</a></td>
</tr>
<tr>
    <td>class</td>
    <td>-</td>
    <td>DINOv2</td>
    <td>UNet (38.4M)</td>
    <td><a href="./configs/cifar10-c2i-dinov2.yaml">cifar10-c2i-dinov2</a></td>
</tr>
<tr>
    <td rowspan="2">FFHQ 256x256</td>
    <td rowspan="2">-</td>
    <td>-</td>
    <td>DINOv2</td>
    <td>DriftDiT-S/16 (33.0M)</td>
    <td><a href="./configs/ffhq-dinov2-dits16.yaml">ffhq-dinov2-dits16</a></td>
</tr>
<tr>
    <td>SDVAE</td>
    <td>DINOv2</td>
    <td>DriftDiT-S/2 (32.4M)</td>
    <td><a href="./configs/ffhq-sdvae-dinov2-dits2.yaml">ffhq-sdvae-dinov2-dits2</a></td>
</tr>
</table>

## Training

```shell
# Unconditional Generation
torchrun --nproc-per-node 8 train_unc.py -c CONFIG [-e EXPDIR] [--bf16]

# Class-to-Image Generation
torchrun --nproc-per-node 8 train_c2i.py -c CONFIG [-e EXPDIR] [--bf16]
```

- `-c CONFIG`: path to the configuration file.
- `-e EXPDIR`: path to the experiment directory. Default: `./runs/exp-<timestamp>`.
- `--bf16`: use bf16 mixed-precision.
- For DriftDiT-based models, set `USE_TORCH_COMPILE=1` to enable `torch.compile`.

## Sampling

```shell
# Unconditional Generation
torchrun --nproc-per-node 8 sample_unc.py -c CONFIG -w WEIGHTS --save-dir SAVE_DIR --num-samples NUM_SAMPLES --bspp BSPP [--bf16] [--make-npz]

# Class-to-Image Generation
torchrun --nproc-per-node 8 sample_c2i.py -c CONFIG -w WEIGHTS --save-dir SAVE_DIR --num-samples NUM_SAMPLES --num-classes NUM_CLASSES --cfg-scale CFG_SCALE --bspp BSPP [--bf16] [--make-npz]
```

- `-c CONFIG`: path to the configuration file.
- `-w WEIGHTS`: path to the trained model weights.
- `--save-dir SAVE_DIR`: directory to save the generated samples.
- `--num-samples NUM_SAMPLES`: total number of samples to generate.
- `--num-classes NUM_CLASSES`: number of classes to generate (only for class-to-image generation).
- `--cfg-scale CFG_SCALE`: classifier-free guidance scale (only for class-to-image generation).
- `--bspp BSPP`: batch size per process.
- `--bf16`: use bf16 mixed-precision.
- `--make-npz`: save the generated samples in `.npz` format for ImageNet FID evaluation.

## Results

### CIFAR-10 (unc)

|     Encoder     | Ng | Nr  | Nf  | B=Ng×Nf | Iters. | FID ↓ |
|:---------------:|:--:|:---:|:---:|:-------:|:------:|:-----:|
|  DINOv2 (norm)  | 64 | 10  | 10  |   640   |  100K  |   ?   |
|  DINOv2 (norm)  | 10 | 64  | 64  |   640   |  100K  |   ?   |
|  DINOv2 (norm)  | 5  | 128 | 128 |   640   |  100K  |   ?   |
|  DINOv2 (norm)  | 1  | 640 | 640 |   640   |  100K  |   ?   |
| MoCov2 (layer4) | 1  | 640 | 640 |   640   |  100K  |   ?   |
