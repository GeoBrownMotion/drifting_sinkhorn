import os
import math
import json
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from einops import rearrange

from models.ema import EMA
from drifting import compute_drift_c2i
from utils.data import C2IDataset
from utils.optimizer import get_param_groups
from utils.logger import get_logger, StatusTracker
from utils.sampler import C2IDistributedSampler, C2IBatchSampler
from utils.misc import check_freq, instantiate_from_config, set_seed, get_time_str
from utils.distributed import (
    init_distributed_mode, get_world_size, get_rank, get_local_rank, cleanup,
    broadcast_tensor, gather_tensor, reduce_tensor, is_dist_avail_and_initialized,
    is_main_process, on_main_process, main_process_first, wait_for_everyone,
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

    # BUILD DATASET
    dataset = instantiate_from_config(conf.data)
    dataset = C2IDataset(dataset)

    # BUILD DATALOADER
    Ng = conf.train.num_groups
    Nr = conf.train.num_real_samples
    Nf = conf.train.num_fake_samples
    Nu = conf.train.num_unc_samples
    assert Nr % get_world_size() == 0
    Nrpp = Nr // get_world_size()
    assert Nf % get_world_size() == 0
    Nfpp = Nf // get_world_size()
    assert Nu % get_world_size() == 0
    Nupp = Nu // get_world_size()
    assert Nu <= Nr

    datasampler = C2IDistributedSampler(dataset.labels, num_replicas=get_world_size(), rank=get_rank())
    batchsampler = C2IBatchSampler(datasampler, num_classes_per_batch=Ng, num_samples_per_class=Nrpp)
    dataloader = DataLoader(dataset, batch_sampler=batchsampler, **conf.dataloader)
    logger.info("=" * 19 + " Data Info " + "=" * 20)
    logger.info(f"Size of training set: {len(dataset)}")
    logger.info(f"Groups per batch: {Ng}")
    logger.info(f"Real samples per group (per process): {Nrpp}")
    logger.info(f"Real samples per group (total): {Nr}")
    logger.info(f"Fake samples per group (per process): {Nfpp}")
    logger.info(f"Fake samples per group (total): {Nf}")
    logger.info(f"Unconditional samples per group (per process): {Nupp}")
    logger.info(f"Unconditional samples per group (total): {Nu}")

    # BUILD MODEL
    model = instantiate_from_config(conf.model).to(device)
    ema = EMA(model, decay=conf.train.ema_decay)
    logger.info("=" * 19 + " Model Info " + "=" * 19)
    logger.info(f"Built model: {model.__class__.__name__}")
    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # BUILD AUTOENCODER
    if hasattr(conf, "autoencoder"):
        with main_process_first():
            autoencoder = instantiate_from_config(conf.autoencoder).to(device).eval()
            for p in autoencoder.parameters():
                p.requires_grad = False
    else:
        from models.autoencoders.identity import IdentityAutoencoder
        autoencoder = IdentityAutoencoder().to(device).eval()
    logger.info(f"Loaded frozen autoencoder: {autoencoder.__class__.__name__}")
    logger.info(f"Number of autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")

    # LOAD FEATURE ENCODER
    if hasattr(conf, "encoder"):
        with main_process_first():
            encoder = instantiate_from_config(conf.encoder).to(device).eval()
            for p in encoder.parameters():
                p.requires_grad = False
    else:
        from models.encoders.identity import IdentityEncoder
        encoder = IdentityEncoder().to(device).eval()
    logger.info(f"Loaded frozen feature encoder: {encoder.__class__.__name__}")
    logger.info(f"Number of encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # BUILD OPTIMIZER AND SCHEDULER
    param_groups = get_param_groups(model, weight_decay=conf.train.optim.params.weight_decay)
    optimizer = instantiate_from_config(conf.train.optim, params=param_groups)
    scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)
    logger.info("=" * 15 + " Optimization Info " + "=" * 16)
    logger.info(f"Learning rate: {conf.train.optim.params.lr}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {scheduler.__class__.__name__}")
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
    num_classes = conf.model.params.num_classes
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

    def sample_alpha(n):
        alpha_min = conf.train.alpha_min
        alpha_max = conf.train.alpha_max
        alpha_power = conf.train.alpha_power
        u = torch.rand((n, ), device=device)
        if alpha_power == 1:
            alpha = alpha_min * (alpha_max / alpha_min) ** u
        else:
            pw = 1 - alpha_power
            alpha = (u * (alpha_max ** pw - alpha_min ** pw) + alpha_min ** pw) ** (1.0 / pw)
        return alpha

    def train_step(batch):
        # get data
        x_real = batch["image"].float().to(device)                                      # (Ng * Nrpp, C, H, W)
        y = batch["label"].long().to(device)                                            # (Ng * Nrpp, )
        x_unc = batch["image_unc"][:Ng * Nupp].float().to(device)                       # (Ng * Nupp, C, H, W)
        # encode to latent
        with torch.no_grad():
            x_real = autoencoder.encode(x_real)                                         # (Ng * Nrpp, C, H, W)
            x_unc = autoencoder.encode(x_unc)                                           # (Ng * Nupp, C, H, W)
        # zero gradients
        optimizer.zero_grad()
        # forward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            # generate fake samples
            z = torch.randn(Ng * Nfpp, *input_shape, device=device)
            yc = y.reshape(Ng, Nrpp)[:, 0]                                              # (Ng, )
            yc = yc.repeat_interleave(Nfpp)                                             # (Ng * Nfpp, )
            alpha = sample_alpha(Ng)                                                    # (Ng, )
            alpha = alpha.repeat_interleave(Nfpp)                                       # (Ng * Nfpp, )
            x_fake = model(z, y=yc, alpha=alpha)                                        # (Ng * Nfpp, C, H, W)
            # set seed for encoders with randomness
            seed = torch.randint(0, 1 << 31, (1,), device=device)
            seed = broadcast_tensor(seed).item()
            # extract features
            with torch.no_grad():
                feat_real = encoder(x_real, autoencoder=autoencoder, seed=seed)         # dict of (B, Ng * Nrpp, D)
                feat_unc = encoder(x_unc, autoencoder=autoencoder, seed=seed)           # dict of (B, Ng * Nupp, D)
            feat_fake = encoder(x_fake, autoencoder=autoencoder, seed=seed)             # dict of (B, Ng * Nfpp, D)
        # compute drifting field for each feature
        loss = torch.tensor(0.0, device=device)
        info = {}
        for name in feat_real.keys():
            f_real = feat_real[name].float()                                            # (B, Ng * Nrpp, D)
            f_unc = feat_unc[name].float()                                              # (B, Ng * Nupp, D)
            f_fake = feat_fake[name].float()                                            # (B, Ng * Nfpp, D)
            # move the group dimension to the feature dimension
            # drifting field is computed independently for each group
            B = f_real.shape[0]
            f_real = rearrange(f_real, "b (ng nr) d -> (b ng) nr d", ng=Ng, nr=Nrpp)    # (B * Ng, Nrpp, D)
            f_unc = rearrange(f_unc, "b (ng nr) d -> (b ng) nr d", ng=Ng, nr=Nupp)      # (B * Ng, Nupp, D)
            f_fake = rearrange(f_fake, "b (ng nf) d -> (b ng) nf d", ng=Ng, nf=Nfpp)    # (B * Ng, Nfpp, D)
            alpha_reshape = alpha.unsqueeze(0).repeat(B, 1).reshape(B * Ng, Nfpp)       # (B * Ng, Nfpp)
            with torch.no_grad():
                V, _info = compute_drift_c2i(
                    x_real=f_real.detach(),
                    x_fake=f_fake.detach(),
                    x_unc=f_unc.detach(),
                    alpha=alpha_reshape,
                    kernel_temp=conf.drifting.kernel_temp,
                    implementation=conf.drifting.implementation,
                    normalize_feature=conf.drifting.normalize_feature,
                    normalize_drift=conf.drifting.normalize_drift,
                )
            # regression loss
            f_fake = f_fake / max(_info["data-scale"], 1e-3)
            loss = loss + F.mse_loss(f_fake, (f_fake + V).detach())
            info = {**info, **{f"{name}-{k}": v for k, v in _info.items()}}
        loss = loss / len(feat_real)
        # backward
        loss.backward()
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
        loss = reduce_tensor(loss.detach())
        grad_norm = reduce_tensor(grad_norm)
        return dict(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            lr=optimizer.param_groups[0]["lr"],
            **info,
        )

    @torch.no_grad()
    def sample(savepath: str):
        generator = torch.Generator(device).manual_seed(get_rank())
        if num_classes <= 10:
            labels = torch.arange(num_classes, device=device)
        else:
            labels = torch.randperm(num_classes, generator=generator, device=device)[:10]
            labels = broadcast_tensor(labels)
        samples_cat = []
        for label in labels:
            num_samples = math.ceil(10 / get_world_size())
            z = torch.randn(num_samples, *input_shape, generator=generator, device=device)
            y = torch.full((num_samples, ), label, dtype=torch.long, device=device)
            alpha = torch.ones_like(y, dtype=torch.float32)
            samples = ema.ema_model(z, y=y, alpha=alpha)
            samples = autoencoder.decode(samples)
            samples = torch.cat(gather_tensor(samples), dim=0)[:10]
            samples = (samples.clamp(-1, 1).cpu() + 1) / 2
            samples_cat.append(samples)
        samples_cat = torch.cat(samples_cat, dim=0)
        if is_main_process():
            save_image(samples_cat, savepath, nrow=10)

    # START TRAINING
    logger.info("Start training...")
    while step < conf.train.num_steps:
        datasampler.set_epoch(epoch)
        for _batch in dataloader:
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
