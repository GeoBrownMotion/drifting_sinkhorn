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
    <td>FFHQ 256x256</td>
    <td>-</td>
    <td>SDVAE</td>
    <td>DINOv2</td>
    <td>DriftDiT-S/2 (32.4M)</td>
    <td><a href="./configs/ffhq-sdvae-dinov2-dits2.yaml">ffhq-sdvae-dinov2-dits2</a></td>
</tr>
</table>


```shell
# Unconditional Generation
torchrun --nproc-per-node 8 train_unc.py -c CONFIG [-e EXPDIR] [--bf16]

# Class-to-Image Generation
torchrun --nproc-per-node 8 train_c2i.py -c CONFIG [-e EXPDIR] [--bf16]
```

- `-c CONFIG`: path to the configuration file.
- `-e EXPDIR`: path to the experiment directory. Default: `./runs/exp-<timestamp>`.
- `--bf16`: use bf16 mixed-precision training.
