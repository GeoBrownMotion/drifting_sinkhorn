# Drifting Models

|     Dataset      |          Task           |   Network    |  Data Space   |          Drift Space          |
|:----------------:|:-----------------------:|:------------:|:-------------:|:-----------------------------:|
|   MNIST 32x32    |   Unconditional (unc)   | UNet (8.2M)  |     Pixel     |             Pixel             |
|   MNIST 32x32    | Class-conditional (c2i) | UNet (9.6M)  |     Pixel     |             Pixel             |
|   CelebA 64x64   |   Unconditional (unc)   | UNet (32.9M) |     Pixel     | Pixel + DINOv2 (single-scale) |

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

## Training

```shell
# MNIST 32x32, unconditional generation
torchrun --nproc-per-node 8 train_unc.py -c ./configs/mnist-unc.yaml

# MNIST 32x32, class-conditional generation
torchrun --nproc-per-node 8 train_c2i.py -c ./configs/mnist-c2i.yaml

# CelebA 64x64
torchrun --nproc-per-node 8 train_unc.py -c ./configs/celeba-unc.yaml --bf16
```
