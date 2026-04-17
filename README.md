# Drifting Models

Unofficial PyTorch implementation of ["Generative Modeling via Drifting"](http://arxiv.org/abs/2602.04770) by Deng et al.

## Setup

```shell
# clone the repo
git clone https://github.com/xyfJASON/drifting-models-pytorch.git
cd drifting-models-pytorch

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
    <th align="left">AE</th>
    <th align="left">Encoder</th>
    <th align="left">Network</th>
    <th align="left">Config.</th>
</tr>
<tr>
    <td rowspan="2">MNIST 32×32</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>UNet (8.2M)</td>
    <td><a href="./configs/mnist-unc-pixel-noenc-unet.yaml">mnist-unc-pixel-noenc-unet</a></td>
</tr>
<tr>
    <td>class</td>
    <td>-</td>
    <td>-</td>
    <td>UNet (9.6M)</td>
    <td><a href="./configs/mnist-c2i-pixel-noenc-unet.yaml">mnist-c2i-pixel-noenc-unet</a></td>
</tr>
<tr>
    <td rowspan="2">CIFAR-10 32×32</td>
    <td>-</td>
    <td>-</td>
    <td>DINOv2</td>
    <td>UNet (32.9M)</td>
    <td><a href="./configs/cifar10-unc-pixel-dinov2-unet.yaml">cifar10-unc-pixel-dinov2-unet</a></td>
</tr>
<tr>
    <td>class</td>
    <td>-</td>
    <td>DINOv2</td>
    <td>UNet (38.4M)</td>
    <td><a href="./configs/cifar10-c2i-pixel-dinov2-unet.yaml">cifar10-c2i-pixel-dinov2-unet</a></td>
</tr>
<tr>
    <td rowspan="2">FFHQ 256×256</td>
    <td rowspan="2">-</td>
    <td>SDVAE</td>
    <td>DINOv2</td>
    <td>UNet (32.9M)</td>
    <td><a href="configs/ffhq-unc-sdvae-dinov2-unet.yaml">ffhq-unc-sdvae-dinov2-unet</a></td>
</tr>
<tr>
    <td>SDVAE</td>
    <td>DINOv2</td>
    <td>DiT-S/2 (33.2M)</td>
    <td><a href="./configs/ffhq-unc-sdvae-dinov2-dits2.yaml">ffhq-unc-sdvae-dinov2-dits2</a></td>
</tr>
</table>

## Preprocessing

Extract and save autoencoder latents. This is optional but can speed up training.

```shell
torchrun --nproc-per-node 8 preprocess.py --dataname DATANAME --dataroot DATAROOT --image-size IMAGESIZE --save-dir SAVEDIR [--autoencoder AUTOENCODER]
```

- `--dataname DATANAME`: name of the dataset, e.g., `ffhq`.
- `--dataroot DATAROOT`: root directory of the dataset.
- `--image-size IMAGESIZE`: image size, e.g., `256`.
- `--save-dir SAVEDIR`: directory to save the extracted latents, e.g., `./data/ffhq256-latents`.
- `--autoencoder AUTOENCODER`: autoencoder to use, e.g., `sdvae`.

## Training

```shell
# Unconditional Generation
torchrun --nproc-per-node 8 train_unc.py -c CONFIG [-e EXPDIR] [--bf16] [--use-latent-dataset]

# Class-to-Image Generation
torchrun --nproc-per-node 8 train_c2i.py -c CONFIG [-e EXPDIR] [--bf16] [--use-latent-dataset]
```

- `-c CONFIG`: path to the configuration file.
- `-e EXPDIR`: path to the experiment directory. Default: `./runs/exp-<timestamp>`.
- `--bf16`: use bf16 mixed-precision.
- `--use-latent-dataset`: use the preprocessed latent dataset for training.
- `USE_TORCH_COMPILE=1`: set this environment variable to enable `torch.compile` for DiT-based models.

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

### CIFAR-10 (unconditional)

|     |  Encoder (layer)  | Ng | Nr  | Nf  | B=Ng×Nf | Iters. |  FID ↓   |
|:---:|:-----------------:|:--:|:---:|:---:|:-------:|:------:|:--------:|
| (a) |   DINOv2 (norm)   | 1  | 640 | 640 |   640   |  100K  | **6.74** |
| (b) |   DINOv2 (norm)   | 5  | 128 | 128 |   640   |  100K  |   8.75   |
| (c) |   DINOv2 (norm)   | 10 | 64  | 64  |   640   |  100K  |  10.23   |
| (d) |   DINOv2 (norm)   | 1  | 320 | 640 |   640   |  100K  |   7.45   |
| (e) |   DINOv2 (norm)   | 1  | 640 | 320 |   320   |  100K  |   7.69   |
| (f) |  MoCov2 (layer4)  | 1  | 640 | 640 |   640   |  100K  |   8.02   |
| (g) | ResNet18 (layer4) | 1  | 640 | 640 |   640   |  100K  |  13.24   |

**Number of groups \[(a),(b),(c)\]**: The cost of computing distance matrix is negligible compared to the cost
of model forward and backward passes, thus the training budget is dominated by the effective batch size B=Ng×Nf.
Given a fixed budget of B=640, reducing the number of groups Ng leads to better performance.

**Number of samples \[(a),(d),(e)\]**: As the drifting field is computed by Monte Carlo estimation, increasing
the number of real samples Nr or the number of fake samples Nf can reduce the variance of the estimation, thus
improving the performance.

**Encoder choice \[(a),(f),(g)\]**: Since Euclidean distance becomes less meaningful in high-dimensional pixel
space, image drifting models rely on pretrained image encoders to extract features for distance computation.
DINOv2 features outperform MoCov2 and ResNet18 features in our experiments.


## References

```bibtex
@article{deng2026generative,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```
