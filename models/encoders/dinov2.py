import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.encoders.base import BaseEncoder
from models.encoders.utils import FeatureExtractor, postprocess


class DINOv2Encoder(BaseEncoder):
    def __init__(
            self,
            model_name: str = "dinov2_vitb14",
            layers: list[str] = ("norm", ),
            global_stat: bool = True,
            patch2_stat: bool = True,
            patch4_stat: bool = True,
    ):
        super().__init__()
        self.global_stat = global_stat
        self.patch2_stat = patch2_stat
        self.patch4_stat = patch4_stat

        # load pretrained dinov2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="XFormers is not available*")
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
            self.dinov2.eval()
        self.num_register_tokens = self.dinov2.num_register_tokens

        # wrap with feature extractor
        self.dinov2 = FeatureExtractor(self.dinov2, layers=layers)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        x = F.interpolate(x, size=(224, 224), mode="bicubic")
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> dict[str, Tensor]:
        # store input
        results = {"x": x.flatten(1).unsqueeze(0)}
        # decode to pixel
        autoencoder = kwargs.get("autoencoder", None)
        if autoencoder is not None:
            x = autoencoder.decode(x)
        # extract features
        z = self.preprocess(x)
        features = self.dinov2(z)
        # postprocess features
        for k, v in features.items():
            v = v[:, self.num_register_tokens+1:]
            B, L, D = v.shape
            results.update({f"dinov2-{k}": postprocess(
                v.reshape(B, 16, 16, D),
                global_stat=self.global_stat,
                patch2_stat=self.patch2_stat,
                patch4_stat=self.patch4_stat,
            )})
        return results
