import re
import numpy as np
from typing import Any
from pathlib import Path
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flax.serialization import msgpack_restore
from huggingface_hub import hf_hub_download

from models.encoders.base import BaseEncoder
from models.encoders.utils import FeatureExtractor, postprocess


class LatentMAEEncoder(BaseEncoder):
    def __init__(
            self,
            model_name: str = "mae_latent_640",
            layers: list[str] = ("norms.0", "norms.1", "norms.2", "norms.3"),
            global_stat: bool = True,
            patch2_stat: bool = True,
            patch4_stat: bool = True,
    ):
        super().__init__()
        self.global_stat = global_stat
        self.patch2_stat = patch2_stat
        self.patch4_stat = patch4_stat

        # load pretrained latent mae
        self.latent_mae = self.load_latent_mae(model_name).eval()

        # wrap with feature extractor
        self.latent_mae = FeatureExtractor(self.latent_mae, layers=layers)

    @staticmethod
    def load_latent_mae(model_name: str) -> nn.Module:
        base_channels = {"mae_latent_256": 256, "mae_latent_640": 640}[model_name]
        latent_mae = ResNetEncoder(in_channels=4, base_channels=base_channels, layers=(3, 4, 6, 3), dropout_prob=0.0)
        ckpt_file = hf_hub_download("Goodeat/drifting", filename=f"models/mae/jax/{model_name}/ema_params.msgpack")
        state_dict_jax = msgpack_restore(Path(ckpt_file).read_bytes())["encoder"]
        state_dict = convert_jax_state_dict_to_torch(state_dict_jax)
        # replace "layer{i}_norm" with "norms.{i-1}"
        for k in list(state_dict.keys()):
            nk = re.sub(r"layer(\d+)_norm", lambda m: f"norms.{int(m[1]) - 1}", k)
            if nk != k:
                state_dict[nk] = state_dict.pop(k)
        latent_mae.load_state_dict(state_dict, strict=True)
        del state_dict_jax
        del state_dict
        return latent_mae

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        # extract features
        features = self.latent_mae(x)
        # postprocess features
        for k, v in features.items():
            results.update({f"mae-{k}": postprocess(
                v.permute(0, 2, 3, 1),
                global_stat=self.global_stat,
                patch2_stat=self.patch2_stat,
                patch4_stat=self.patch4_stat,
            )})
        return results


# ==========================================================================================================
# Self-contained PyTorch MAE-ResNet implementation that is compatible with the pretrained JAX checkpoints.
#
# Developed with the help of GitHub Copilot (GPT-5.3-Codex).
# Reference: https://github.com/lambertae/drifting/blob/main/models/mae_model.py
# ==========================================================================================================

def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        stride: int = 1,
        gn_max_groups: int = 32,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.drop = nn.Dropout(p=dropout_prob)
        if stride > 1:
            self.proj_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)
            self.proj_gn = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)

    def forward(self, x: Tensor, train: bool = False) -> Tensor:
        residual = x
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y, inplace=False)
        y = F.dropout(y, p=self.drop.p, training=train)
        y = self.conv2(y)
        y = self.gn2(y)

        if residual.shape != y.shape:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        return F.relu(residual + y, inplace=False)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.layers = layers
        self.gn_max_groups = gn_max_groups
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(base_channels, gn_max_groups), base_channels)

        stages: list[nn.Module] = []
        norms: list[nn.Module] = []
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = base_channels * (2**stage_idx)
            in_ch = base_channels if stage_idx == 0 else base_channels * (2 ** (stage_idx - 1))
            blocks = [
                BasicBlock(
                    in_channels=in_ch,
                    filters=out_ch,
                    stride=stride,
                    gn_max_groups=gn_max_groups,
                    dropout_prob=dropout_prob,
                )
            ]
            for _ in range(1, num_blocks):
                blocks.append(
                    BasicBlock(
                        in_channels=out_ch,
                        filters=out_ch,
                        stride=1,
                        gn_max_groups=gn_max_groups,
                        dropout_prob=dropout_prob,
                    )
                )
            stages.append(nn.ModuleList(blocks))
            norms.append(nn.GroupNorm(_choose_gn_groups(out_ch, gn_max_groups), out_ch))
        self.stages = nn.ModuleList(stages)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: Tensor, train: bool = False) -> Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x, inplace=False)
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, train=train)
            x = self.norms[i](x)
        return x


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(_flatten_tree(v, key))
        return out
    out[prefix] = tree
    return out


def _normalize_jax_key(key: str) -> str:
    if key.startswith("params/"):
        key = key[len("params/") :]

    key = re.sub(r"stages_(\d+)/layers_(\d+)", r"stages.\1.\2", key)
    key = key.replace("/", ".")
    key = key.replace(".scale", ".weight")
    key = key.replace(".kernel", ".weight")
    return key


def _jax_tensor_to_torch(name: str, value: Any) -> Tensor:
    arr = np.asarray(value)
    t = torch.from_numpy(arr.copy())
    if name.endswith(".weight"):
        if t.ndim == 4:
            t = t.permute(3, 2, 0, 1)  # HWIO -> OIHW
        elif t.ndim == 2:
            t = t.permute(1, 0)  # IO -> OI
    return t


def convert_jax_state_dict_to_torch(raw_state: Mapping[str, Any]) -> dict[str, Tensor]:
    flat = _flatten_tree(raw_state)
    out: dict[str, Tensor] = {}
    for k, v in flat.items():
        name = _normalize_jax_key(str(k))
        if not name:
            continue
        out[name] = _jax_tensor_to_torch(name, v)
    return out
