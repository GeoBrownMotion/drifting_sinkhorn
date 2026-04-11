import os
import json
import tqdm
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import hflip

from utils.logger import get_logger
from utils.misc import set_seed
from utils.distributed import (
    init_distributed_mode, get_world_size, get_rank, cleanup,
    gather_object, is_dist_avail_and_initialized,
    is_main_process, main_process_first, wait_for_everyone,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--image-size", type=int, required=True, help="The size of the images")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the latents")
    parser.add_argument("--autoencoder", type=str, default="sdvae", help="Autoencoder type")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per process")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser


def main():
    # PARSE ARGS
    args = get_parser().parse_args()

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f"Process {get_rank()} using device: {device}", flush=True)
    wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(is_main_process=is_main_process())

    # SET SEED
    set_seed(args.seed + get_rank())
    logger.info(f"Number of processes: {get_world_size()}")
    logger.info(f"Distributed mode: {is_dist_avail_and_initialized()}")
    wait_for_everyone()

    # BUILD DATASET
    if args.dataname == "imagenet":
        from utils.data import ImageNet
        dataset = ImageNet(root=args.dataroot, image_size=args.image_size, pflip=0.0)
    elif args.dataname == "ffhq":
        from utils.data import FFHQ
        dataset = FFHQ(root=args.dataroot, image_size=args.image_size, pflip=0.0)
    else:
        raise ValueError(f"Unknown dataset: {args.dataname}")

    # BUILD DATALOADER
    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=datasampler, drop_last=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    logger.info(f"Loaded dataset: {args.dataname}")
    logger.info(f"Size of dataset: {len(dataset)}")
    logger.info(f"Batch size per process: {args.batch_size}")

    # LOAD AUTOENCODER
    if args.autoencoder == "sdvae":
        from models.autoencoders.sdvae import SDVAE
        with main_process_first():
            autoencoder = SDVAE().to(device)
    else:
        raise ValueError(f"Unknown autoencoder: {args.autoencoder}")
    logger.info(f"Loaded autoencoder: {autoencoder.__class__.__name__}")

    # CREATE SAVE DIR
    args.save_dir = os.path.expanduser(args.save_dir)
    if is_main_process():
        os.makedirs(args.save_dir)
    logger.info(f"Latents will be saved to: {args.save_dir}")
    wait_for_everyone()

    # START PREPROCESSING
    logger.info(f"Start preprocessing...")
    metadata = {}
    for batch in tqdm.tqdm(dataloader, desc="Preprocessing", disable=not is_main_process()):
        # get data
        indices = batch["index"]
        images = batch["image"].float().to(device)
        labels = (batch["label"] if "label" in batch else torch.zeros_like(indices)).long()

        # encode
        with torch.no_grad():
            latents = autoencoder.encode(images)
            latents_hflip = autoencoder.encode(hflip(images))

        # save to disk
        for i in range(len(indices)):
            # prepare data
            idx = indices[i].cpu().item()
            label = labels[i].cpu().item()
            latent = latents[i].cpu().numpy()
            latent_hflip = latents_hflip[i].cpu().numpy()
            # save latent
            save_dir = os.path.join(args.save_dir, f"{(2*idx)//1000:05d}")
            os.makedirs(save_dir, exist_ok=True)
            file = os.path.join(save_dir, f"{(2*idx):08d}.npz")
            np.savez(file, label=label, latent=latent)
            metadata[2*idx] = {"file": str(file), "label": int(label)}
            # save flipped latent
            save_dir = os.path.join(args.save_dir, f"{(2*idx+1)//1000:05d}")
            os.makedirs(save_dir, exist_ok=True)
            file = os.path.join(save_dir, f"{(2*idx+1):08d}.npz")
            np.savez(file, label=label, latent=latent_hflip)
            metadata[2*idx+1] = {"file": str(file), "label": int(label)}
        wait_for_everyone()

    # save metadata
    metadata = {k: v for m in gather_object(metadata) for k, v in m.items()}
    metadata = dict(sorted(metadata.items()))
    if is_main_process():
        with open(os.path.join(args.save_dir, "metadata.jsonl"), "w") as f:
            for item in metadata.values():
                f.write(json.dumps(item) + "\n")
    wait_for_everyone()

    # END OF PREPROCESSING
    logger.info(f"End of preprocessing.")
    cleanup()


if __name__ == "__main__":
    main()
