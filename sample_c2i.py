import os
import glob
import tqdm
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from torchvision.utils import save_image

from utils.logger import get_logger
from utils.misc import set_seed, instantiate_from_config
from utils.distributed import (
    init_distributed_mode, get_world_size, get_rank, is_dist_avail_and_initialized,
    is_main_process, main_process_first, wait_for_everyone, cleanup,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to pretrained model weights")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save generated samples")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--bspp", type=int, default=64, help="Batch size for each process")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 mixed precision")
    parser.add_argument("--make-npz", action="store_true", help="Make npz file after sampling")
    parser.add_argument("--label-strategy", type=str, default="balanced", help="Label sampling strategy")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for RNG (adjusted per rank)")
    return parser


class LabelDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int, strategy: str = "balanced"):
        if strategy == "balanced":
            assert num_samples % num_classes == 0
            num_samples_per_class = num_samples // num_classes
            self.labels = torch.arange(num_classes).repeat_interleave(num_samples_per_class).tolist()
        elif strategy == "random":
            self.labels = torch.randint(0, num_classes, (num_samples,)).tolist()
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return index, self.labels[index]


def main():
    # PARSE ARGS AND CONFIGS
    args = get_parser().parse_args()
    conf = OmegaConf.load(args.config)

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(is_main_process=is_main_process())

    # SET SEED
    set_seed(args.seed + get_rank())
    logger.info("=" * 19 + " System Info " + "=" * 18)
    logger.info(f"Using device: {device}")
    logger.info(f"Number of processes: {get_world_size()}")
    logger.info(f"Distributed mode: {is_dist_avail_and_initialized()}")
    logger.info(f"Mixed precision (bf16): {args.bf16}")
    wait_for_everyone()

    # BUILD DATASET
    dataset = LabelDataset(args.num_samples, args.num_classes, args.label_strategy)

    # BUILD DATALOADER
    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp, sampler=datasampler, drop_last=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    logger.info("=" * 19 + " Data Info " + "=" * 20)
    logger.info(f"Number of samples: {len(dataset)}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Label sampling strategy: {args.label_strategy}")
    logger.info(f"Batch size per process: {args.bspp}")

    # BUILD MODEL AND LOAD WEIGHTS
    model = instantiate_from_config(conf.model).eval().to(device)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    logger.info("=" * 19 + " Model Info " + "=" * 19)
    logger.info(f"Built model: {model.__class__.__name__}")
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Successfully load model from {args.weights}")

    # BUILD AUTOENCODER
    if hasattr(conf, "autoencoder"):
        with main_process_first():
            autoencoder = instantiate_from_config(conf.autoencoder).to(device).eval()
    else:
        from models.autoencoders.identity import IdentityAutoencoder
        autoencoder = IdentityAutoencoder().to(device).eval()
    logger.info(f"Loaded frozen autoencoder: {autoencoder.__class__.__name__}")
    logger.info(f"Number of autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    logger.info("=" * 50)
    wait_for_everyone()

    # CREATE SAVE DIR
    if is_main_process():
        os.makedirs(args.save_dir)
    logger.info(f"Samples will be saved to {args.save_dir}")
    wait_for_everyone()

    # START SAMPLING
    logger.info("Start sampling...")
    in_channels = conf.model.params.in_channels
    input_size = conf.model.params.input_size
    input_shape = (in_channels, input_size, input_size)

    for indices, y in tqdm.tqdm(dataloader, desc="Sampling", disable=not is_main_process()):
        y = y.long().to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            z = torch.randn((indices.shape[0], *input_shape), device=device)
            alpha = torch.full((indices.shape[0], ), fill_value=args.cfg_scale, dtype=torch.float32, device=device)
            samples = model(z, y=y, alpha=alpha)
            samples = autoencoder.decode(samples)
        samples = (samples.clamp(-1, 1).cpu() + 1) / 2
        for idx, sample in zip(indices, samples):
            save_image(sample, os.path.join(args.save_dir, f"{idx:06d}.png"))
        wait_for_everyone()
    logger.info(f"Sampled images are saved to {args.save_dir}")
    wait_for_everyone()

    # MAKE NPZ FILE
    if is_main_process() and args.make_npz:
        logger.info("Start making npz file...")

        # find images
        image_paths = sorted(list(glob.glob(os.path.join(args.save_dir, "*.png"))))
        logger.info(f"Found {len(image_paths)} images in {args.save_dir}")

        # shuffle images
        random.seed(args.seed)
        random.shuffle(image_paths)

        # read images
        images = []
        for path in tqdm.tqdm(image_paths, desc="Reading images"):
            images.append(np.asarray(Image.open(path)).astype(np.uint8))
        images = np.stack(images)

        # save npz file
        save_file = f"{args.save_dir}.npz"
        np.savez(save_file, arr_0=images)
        logger.info(f"Saved npz file to {save_file} [shape={images.shape}].")
    wait_for_everyone()

    # CLEANUP
    logger.info("End of sampling.")
    cleanup()


if __name__ == "__main__":
    main()
