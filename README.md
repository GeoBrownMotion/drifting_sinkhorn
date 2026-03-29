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
    <th>Dataset</th>
    <th>Task</th>
    <th>Network</th>
    <th>Data Space</th>
    <th>Feature Encoder</th>
    <th>Config</th>
</tr>
<tr>
    <td rowspan="2">MNIST 32x32</td>
    <td>Uncond.</td>
    <td>UNet (8.2M)</td>
    <td>Pixel</td>
    <td>-</td>
    <td><a href="./configs/mnist-unc.yaml">config</a></td>
</tr>
<tr>
    <td>Class-to-Image</td>
    <td>UNet (9.6M)</td>
    <td>Pixel</td>
    <td>-</td>
    <td><a href="./configs/mnist-c2i.yaml">config</a></td>
</tr>
<tr>
    <td rowspan="2">CIFAR-10 32x32</td>
    <td>Uncond.</td>
    <td>UNet (32.9M)</td>
    <td>Pixel</td>
    <td>DINOv2</td>
    <td><a href="./configs/cifar10-unc.yaml">config</a></td>
</tr>
<tr>
    <td>Class-to-Image</td>
    <td>UNet (38.4M)</td>
    <td>Pixel</td>
    <td>DINOv2</td>
    <td><a href="./configs/cifar10-c2i.yaml">config</a></td>
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
```
