import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50
from torchvision.transforms.functional import normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.encoders.base import BaseEncoder
from models.encoders.utils import FeatureExtractor, postprocess


class MoCov2Encoder(BaseEncoder):
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

        # load pretrained mocov2
        self.mocov2 = resnet50().eval()
        self.load_pretrained()

        # wrap with feature extractor
        self.mocov2 = FeatureExtractor(self.mocov2, layers=layers)

    def load_pretrained(self):
        url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, progress=True, map_location="cpu", weights_only=True)
        state_dict = {
            k.removeprefix("module.encoder_q."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc")
        }
        msg = self.mocov2.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        assert len(msg.unexpected_keys) == 0

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
        features = self.mocov2(z)
        # postprocess features
        for k, v in features.items():
            results.update({f"mocov2-{k}": postprocess(
                v.permute(0, 2, 3, 1),
                global_stat=self.global_stat,
                patch2_stat=self.patch2_stat,
                patch4_stat=self.patch4_stat,
            )})
        return results
