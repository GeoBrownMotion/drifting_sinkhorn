import os
import math
import json
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from models.ema import EMA
from drifting import compute_drift
from utils.logger import get_logger, StatusTracker
from utils.optimizer import get_param_groups, get_actual_lr
from utils.misc import check_freq, instantiate_from_config, set_seed, get_time_str
from utils.distributed import (
    cleanup, gather_tensor, reduce_tensor, get_local_rank, get_rank, get_world_size, init_distributed_mode,
    is_dist_avail_and_initialized, is_main_process, on_main_process, wait_for_everyone,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Configuration yaml file")
    parser.add_argument("-e", "--exp-dir", type=str, default=f"runs/exp-{get_time_str()}", help="Experiment directory")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override or add config entries")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--resume", type=str, help="Path to the resume checkpoint directory")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 mixed precision training")
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args = get_parser().parse_args()
    conf_base = OmegaConf.load(args.config)
    conf_overrides = OmegaConf.from_dotlist(args.overrides)
    conf = OmegaConf.merge(conf_base, conf_overrides)

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if is_main_process():
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, "ckpt"))
        os.makedirs(os.path.join(exp_dir, "samples"))
        with open(os.path.join(exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            OmegaConf.save(conf, f)

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, "output.log"),
        is_main_process=is_main_process(),
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, "tensorboard"),
        is_main_process=is_main_process(),
    )

    # SET SEED
    set_seed(args.seed + get_rank())
    logger.info("=" * 19 + " System Info " + "=" * 18)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Number of processes: {get_world_size()}")
    logger.info(f"Distributed mode: {is_dist_avail_and_initialized()}")
    logger.info(f"Mixed precision (bf16): {args.bf16}")
    if len(args.overrides) > 0:
        logger.info("=" * 19 + " Config Info " + "=" * 18)
        for item in args.overrides:
            key, _ = item.split("=", 1)
            old_value = OmegaConf.select(conf_base, key)
            new_value = OmegaConf.select(conf, key)
            if old_value is None:
                logger.info(f"Override: {key}: <NEW> {new_value}")
            else:
                logger.info(f"Override: {key}: {old_value} -> {new_value}")
    wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % get_world_size() == 0
    bspp = conf.train.batch_size // get_world_size()
    assert conf.train.gen_batch_size % get_world_size() == 0
    gen_bspp = conf.train.gen_batch_size // get_world_size()

    train_set = instantiate_from_config(conf.data)
    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    train_loader = DataLoader(train_set, batch_size=bspp, sampler=train_sampler, drop_last=True, **conf.dataloader)
    logger.info("=" * 19 + " Data Info " + "=" * 20)
    logger.info(f"Size of training set: {len(train_set)}")
    logger.info(f"Batch size per process: {bspp}")
    logger.info(f"Total batch size: {conf.train.batch_size}")
    logger.info(f"Generator batch size per process: {gen_bspp}")
    logger.info(f"Total generator batch size: {conf.train.gen_batch_size}")

    # BUILD MODEL
    model = instantiate_from_config(conf.model).to(device)
    ema = EMA(model, decay=conf.train.ema_decay)
    logger.info("=" * 19 + " Model Info " + "=" * 19)
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # BUILD FEATURE ENCODER
    encoder = instantiate_from_config(conf.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info(f"Built frozen feature encoder: {conf.encoder.target}")
    logger.info(f"Number of encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # BUILD OPTIMIZER AND SCHEDULER
    actual_lr = get_actual_lr(conf.train.optim.params.lr, conf.train.batch_size, conf.train.optim.scale_lr)
    param_groups = get_param_groups(model, weight_decay=conf.train.optim.params.weight_decay)
    optimizer = instantiate_from_config(conf.train.optim, params=param_groups, lr=actual_lr)
    scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)
    logger.info("=" * 17 + " Optimizer Info " + "=" * 17)
    logger.info(f"Learning rate scaling rule: {conf.train.optim.scale_lr}")
    logger.info(f"Base learning rate: {conf.train.optim.params.lr}")
    logger.info(f"Actual learning rate: {actual_lr}")
    logger.info("=" * 50)

    # RESUME TRAINING
    step, epoch = 0, 0
    if args.resume is not None:
        logger.info(f"Resume from {args.resume}")
        # load model
        ckpt = torch.load(os.path.join(args.resume, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Successfully load model from {args.resume}")
        # load ema model
        ckpt = torch.load(os.path.join(args.resume, "model_ema.pt"), map_location="cpu", weights_only=True)
        ema.ema_model.load_state_dict(ckpt["model"])
        logger.info(f"Successfully load EMA model from {args.resume}")
        # load training states (optimizer, scheduler, step, epoch)
        ckpt = torch.load(os.path.join(args.resume, "training_states.pt"), map_location="cpu", weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"] + 1
        epoch = ckpt["epoch"]
        logger.info(f"Successfully load training states from {args.resume}")
        logger.info(f"Restart training at step {step}")
        del ckpt

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())
    model_wo_ddp = model.module if is_dist_avail_and_initialized() else model
    if args.resume is None:
        ema.update(model_wo_ddp, decay=0)  # ensure ema is initialized with synced weights
    wait_for_everyone()

    # TRAINING FUNCTIONS
    in_channels = conf.model.params.in_channels
    input_size = conf.model.params.input_size
    input_shape = (in_channels, input_size, input_size)

    @on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model and ema model
        torch.save(dict(model=model_wo_ddp.state_dict()), os.path.join(save_path, "model.pt"))
        torch.save(dict(model=ema.ema_model.state_dict()), os.path.join(save_path, "model_ema.pt"))
        # save training states (optimizer, scheduler, step, epoch)
        torch.save(dict(
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=step,
            epoch=epoch,
        ), os.path.join(save_path, "training_states.pt"))

    def train_step(batch):
        # get data
        x_real = batch["image"].float().to(device)
        # zero gradients
        optimizer.zero_grad()
        # forward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            # generate fake samples
            z = torch.randn(gen_bspp, *input_shape, device=device)
            x_fake = model(z)
            # extract features
            feat_real = encoder(x_real)
            feat_fake = encoder(x_fake)
            if not isinstance(feat_real, dict):
                feat_real = {"scale0": feat_real}
                feat_fake = {"scale0": feat_fake}
            # compute drifting field for each feature
            loss_sum = torch.tensor(0.0, device=device)
            info_sum = {}
            for feat_name in feat_real.keys():
                f_real = feat_real[feat_name]
                f_fake = feat_fake[feat_name]
                V, info = compute_drift(
                    x_real=f_real,
                    x_fake=f_fake,
                    kernel_temp=conf.drifting.kernel_temp,
                    implementation=conf.drifting.implementation,
                    normalize_feature=conf.drifting.normalize_feature,
                    normalize_drift=conf.drifting.normalize_drift,
                )
                # regression loss
                loss = F.mse_loss(f_fake, (f_fake + V).detach())
                loss_sum = loss_sum + loss
                info_sum = {**info_sum, **{f"{feat_name}-{k}": v for k, v in info.items()}}
        # backward
        loss_sum.backward()
        # clip gradients
        if conf.train.get("clip_grad_norm", None):
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.train.clip_grad_norm)
        else:
            grad_norm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
        # update params
        optimizer.step()
        ema.update(model_wo_ddp)
        # update lr
        scheduler.step()
        # reduce stats for logging
        loss_sum = reduce_tensor(loss_sum.detach())
        grad_norm = reduce_tensor(grad_norm)
        return dict(
            loss=loss_sum.item(),
            grad_norm=grad_norm.item(),
            lr=optimizer.param_groups[0]["lr"],
            **info_sum,
        )

    @torch.no_grad()
    def sample(savepath: str):
        num_samples = math.ceil(conf.train.num_samples / get_world_size())
        z = torch.randn(num_samples, *input_shape, device=device)
        samples = ema.ema_model(z)
        samples = torch.cat(gather_tensor(samples), dim=0)[:conf.train.num_samples]
        samples = (samples.clamp(-1, 1).cpu() + 1) / 2
        if is_main_process():
            save_image(samples, savepath, nrow=int(conf.train.num_samples ** 0.5))

    # START TRAINING
    logger.info("Start training...")
    while step < conf.train.num_steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for _batch in tqdm.tqdm(train_loader, desc="Epoch", leave=False, disable=not is_main_process()):
            # train a step
            model.train()
            train_status = train_step(_batch)
            if train_status is None:
                continue
            status_tracker.track_status("Train", train_status, step)
            wait_for_everyone()
            # validate
            model.eval()
            # save checkpoint
            if check_freq(conf.train.save_freq, step):
                save_ckpt(os.path.join(exp_dir, "ckpt", f"step{step:0>7d}"))
                wait_for_everyone()
            # sample from current model
            if check_freq(conf.train.sample_freq, step):
                sample(os.path.join(exp_dir, "samples", f"step{step:0>7d}.jpg"))
                wait_for_everyone()
            step += 1
            if step >= conf.train.num_steps:
                break
        epoch += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, "ckpt", f"step{step-1:0>7d}"))
    wait_for_everyone()

    # END OF TRAINING
    logger.info("End of training")
    status_tracker.close()
    cleanup()


if __name__ == "__main__":
    main()
