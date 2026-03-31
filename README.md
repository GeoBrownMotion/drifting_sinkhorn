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

## Training

<table>
<tr>
    <th style="text-align: left">Dataset</th>
    <th style="text-align: left">Task</th>
    <th style="text-align: left">Network</th>
    <th style="text-align: left">Autoencoder</th>
    <th style="text-align: left">Feature Encoder</th>
    <th style="text-align: left">Config</th>
</tr>
<tr>
    <td rowspan="2">MNIST 32x32</td>
    <td>Uncond.</td>
    <td>UNet (8.2M)</td>
    <td>-</td>
    <td>-</td>
    <td><a href="./configs/mnist-unc.yaml">config</a></td>
</tr>
<tr>
    <td>Class-to-Image</td>
    <td>UNet (9.6M)</td>
    <td>-</td>
    <td>-</td>
    <td><a href="./configs/mnist-c2i.yaml">config</a></td>
</tr>
<tr>
    <td rowspan="2">CIFAR-10 32x32</td>
    <td>Uncond.</td>
    <td>UNet (32.9M)</td>
    <td>-</td>
    <td>DINOv2</td>
    <td><a href="./configs/cifar10-unc.yaml">config</a></td>
</tr>
<tr>
    <td>Class-to-Image</td>
    <td>UNet (38.4M)</td>
    <td>-</td>
    <td>DINOv2</td>
    <td><a href="./configs/cifar10-c2i.yaml">config</a></td>
</tr>
<tr>
    <td>FFHQ 256x256</td>
    <td>Uncond.</td>
    <td>DriftDiT-S/2 (32.4M)</td>
    <td>SDVAE</td>
    <td>DINOv2</td>
    <td><a href="./configs/ffhq-unc.yaml">config</a></td>
</tr>
</table>


```shell
# MNIST 32x32, unconditional generation
torchrun --nproc-per-node 8 train_unc.py -c ./configs/mnist-unc.yaml

# MNIST 32x32, class-to-image generation
torchrun --nproc-per-node 8 train_c2i.py -c ./configs/mnist-c2i.yaml

# CIFAR-10 32x32, unconditional generation
torchrun --nproc-per-node 8 train_unc.py -c ./configs/cifar10-unc.yaml --bf16

# CIFAR-10 32x32, class-to-image generation
torchrun --nproc-per-node 8 train_c2i.py -c ./configs/cifar10-c2i.yaml --bf16

# FFHQ 256x256, unconditional generation
torchrun --nproc-per-node 8 train_unc.py -c ./configs/ffhq-unc.yaml --bf16
```
