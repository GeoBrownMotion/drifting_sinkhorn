import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import normalize
from torchvision.models import resnet18, ResNet18_Weights
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.encoders.base import BaseEncoder
from models.encoders.utils import FeatureExtractor, postprocess


class ResNet18Encoder(BaseEncoder):
    def __init__(
            self,
            layers: list[str] = ("layer1", "layer2", "layer3", "layer4"),
            global_stat: bool = True,
            patch2_stat: bool = True,
            patch4_stat: bool = True,
    ):
        super().__init__()
        self.global_stat = global_stat
        self.patch2_stat = patch2_stat
        self.patch4_stat = patch4_stat

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.resnet = FeatureExtractor(self.resnet, layers=layers)

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
        features = self.resnet(z)
        # postprocess features
        for k, v in features.items():
            results.update({f"resnet18-{k}": postprocess(
                v.permute(0, 2, 3, 1),
                global_stat=self.global_stat,
                patch2_stat=self.patch2_stat,
                patch4_stat=self.patch4_stat,
            )})
        return results
