import os
import math
import json
import argparse
import time
from omegaconf import OmegaConf
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from einops import rearrange

from models.ema import EMA
from drifting_debug import compute_drift   # switched to debug module for mutual-softmax2x / softmax-extra ablations
from drifting_split import compute_drift_split
from utils.optimizer import get_param_groups
from utils.logger import get_logger, StatusTracker
from utils.misc import check_freq, instantiate_from_config, set_seed, get_time_str
from utils.distributed import (
    init_distributed_mode, get_world_size, get_rank, get_local_rank, cleanup,
    gather_tensor, reduce_tensor, is_dist_avail_and_initialized,
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
    parser.add_argument("--use-latent-dataset", action="store_true", default=False, help="Use latent dataset")
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
    if not args.use_latent_dataset:
        dataset = instantiate_from_config(conf.data)
    else:
        dataset = instantiate_from_config(conf.latent)

    # BUILD DATALOADER
    grad_acc_steps = conf.train.get("gradient_accumulation", 1)
    assert conf.train.num_groups % grad_acc_steps == 0
    assert conf.train.num_real_samples % get_world_size() == 0
    assert conf.train.num_fake_samples % get_world_size() == 0
    Ng = conf.train.num_groups // grad_acc_steps
    Nr = conf.train.num_real_samples // get_world_size()
    Nf = conf.train.num_fake_samples // get_world_size()

    datasampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, batch_size=Ng * Nr, sampler=datasampler, drop_last=True, **conf.dataloader)
    logger.info("=" * 19 + " Data Info " + "=" * 20)
    logger.info(f"Size of dataset: {len(dataset)}")
    logger.info(f"Gradient accumulation: {grad_acc_steps}")
    logger.info(f"Micro batch:")
    logger.info(f"    Number of groups: {Ng}")
    logger.info(f"    Real samples per group: {Nr}")
    logger.info(f"    Fake samples per group: {Nf}")
    logger.info(f"Effective batch:")
    logger.info(f"    Number of groups: {Ng}x{grad_acc_steps}={Ng * grad_acc_steps}")
    logger.info(f"    Real samples per group: {Nr}x{get_world_size()}={Nr * get_world_size()}")
    logger.info(f"    Fake samples per group: {Nf}x{get_world_size()}={Nf * get_world_size()}")

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
    if os.environ.get("USE_TORCH_COMPILE", "0") == "1":
        autoencoder.encode = torch.compile(autoencoder.encode)
        autoencoder.decode = torch.compile(autoencoder.decode)
        logger.info("Compiled autoencoder with torch.compile")

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
    if os.environ.get("USE_TORCH_COMPILE", "0") == "1":
        encoder.forward = torch.compile(encoder.forward)
        logger.info("Compiled encoder with torch.compile")

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
        # load model
        ckpt = torch.load(os.path.join(args.resume, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        # load ema model
        ckpt = torch.load(os.path.join(args.resume, "model_ema.pt"), map_location="cpu", weights_only=True)
        ema.ema_model.load_state_dict(ckpt["model"])
        # load training states (optimizer, scheduler, step, epoch)
        ckpt = torch.load(os.path.join(args.resume, "training_states.pt"), map_location="cpu", weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"] + 1
        epoch = ckpt["epoch"]
        logger.info(f"Successfully resume from {args.resume}")
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
    grad_acc_counter = 0

    def infinite_iterator():
        nonlocal epoch
        while True:
            datasampler.set_epoch(epoch)
            for batch in dataloader:
                yield batch
            epoch += 1

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

    # === negative-feature memory bank (only used when drifting.neg_source == 'feature_bank') ===
    # Per-rank storage of past detached fake features (one list of CPU tensors per
    # stream name). At each step we push the current fake-feature chunk and, after
    # `bank_warmup_steps` chunks have accumulated, sample a fresh negative batch
    # from this bank instead of paying for an extra generator+encoder forward.
    # Bank is on CPU to avoid GPU-memory pressure; only the sampled batch moves
    # back to GPU each step (~50 ms at B=2048 with DINOv2 features).
    neg_source = conf.drifting.get("neg_source", "shared")
    valid_neg_sources = {"shared", "independent_fake", "feature_bank"}
    if neg_source not in valid_neg_sources:
        raise ValueError(f"Unknown drifting.neg_source={neg_source!r}; expected one of {sorted(valid_neg_sources)}")
    bank_size = int(conf.drifting.get("bank_size", 8192))
    bank_warmup_steps = int(conf.drifting.get("bank_warmup_steps", 5))
    if bank_size <= 0:
        raise ValueError(f"drifting.bank_size must be positive, got {bank_size}")
    if bank_warmup_steps < 0:
        raise ValueError(f"drifting.bank_warmup_steps must be >= 0, got {bank_warmup_steps}")
    if neg_source == "feature_bank" and bank_size < Ng * Nf:
        raise ValueError(
            f"drifting.bank_size={bank_size} is smaller than the local negative batch size {Ng * Nf}; "
            "increase bank_size or reduce train.num_fake_samples"
        )

    neg_bank: dict = {}    # name -> list of (F, n_chunk, D) CPU tensors

    def bank_num_samples(name: str) -> int:
        return sum(t.shape[1] for t in neg_bank.get(name, []))

    def sample_feature_bank(name: str, n_target: int, dtype: torch.dtype) -> torch.Tensor | None:
        """Sample CPU bank entries and move only the selected features to GPU."""
        chunks = neg_bank.get(name, [])
        total = sum(t.shape[1] for t in chunks)
        if total < n_target:
            return None
        idx = torch.randperm(total)[:n_target].sort().values
        pieces = []
        offset = 0
        for chunk in chunks:
            next_offset = offset + chunk.shape[1]
            take = (idx >= offset) & (idx < next_offset)
            if take.any():
                pieces.append(chunk.index_select(1, idx[take] - offset))
            offset = next_offset
        sampled = torch.cat(pieces, dim=1)
        return sampled.to(device=device, dtype=dtype, non_blocking=True)

    def train_step(batch):
        nonlocal grad_acc_counter
        # get data
        x_real = batch["image"].float().to(device)
        # encode to latent
        if not args.use_latent_dataset:
            with torch.no_grad():
                x_real = autoencoder.encode(x_real)  # (Ng * Nr, C, H, W)
        # zero gradients
        if grad_acc_counter == 0:
            optimizer.zero_grad()
        # nosync context
        maybe_nosync = nullcontext()
        if is_dist_avail_and_initialized():
            if grad_acc_counter < grad_acc_steps - 1:
                maybe_nosync = model.no_sync()
        with maybe_nosync:
            # forward
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                # generate fake samples (query batch)
                z = torch.randn(Ng * Nf, *input_shape, device=device)     # (Ng * Nf, C, H, W)
                x_fake = model(z)                                         # (Ng * Nf, C, H, W)
                # extract features
                with torch.no_grad():
                    feat_real = encoder(x_real, autoencoder=autoencoder)  # dict of (F, Ng * Nr, D)
                feat_fake = encoder(x_fake, autoencoder=autoencoder)      # dict of (F, Ng * Nf, D)

                # Optional: independent negative batch. Choices:
                #   'shared' (default)         : Y_neg = X_query (paper's protocol;
                #                                creates structural self-pair → Sinkhorn
                #                                collapse at sharp τ + high-D).
                #   'independent_fake'         : sample second noise + extra generator +
                #                                encoder forward (no_grad). +25-35% wallclock.
                #   'feature_bank'             : sample past fake-features from rank-local
                #                                bank. ~zero extra forward; slight CPU↔GPU
                #                                transfer cost. Stale negatives but cheap.
                feat_neg = None
                if neg_source == "independent_fake":
                    with torch.no_grad():
                        z_neg = torch.randn(Ng * Nf, *input_shape, device=device)
                        x_neg = model(z_neg)
                        feat_neg = encoder(x_neg, autoencoder=autoencoder)
                elif neg_source == "feature_bank":
                    # Bank is ready iff every stream has enough stale samples.
                    bank_ready = all(
                        len(neg_bank.get(name, [])) >= bank_warmup_steps
                        and bank_num_samples(name) >= Ng * Nf
                        for name in feat_fake.keys()
                    )
                    if bank_ready:
                        feat_neg = {}
                        n_target = Ng * Nf
                        for name in feat_fake.keys():
                            sampled = sample_feature_bank(name, n_target, feat_fake[name].dtype)
                            if sampled is None:
                                bank_ready = False
                                feat_neg = None
                                break
                            feat_neg[name] = sampled
                    if not bank_ready:
                        # Warmup fallback: avoid shared Y_neg=X even before the bank is ready.
                        with torch.no_grad():
                            z_neg = torch.randn(Ng * Nf, *input_shape, device=device)
                            x_neg = model(z_neg)
                            feat_neg = encoder(x_neg, autoencoder=autoencoder)
            # compute drifting field for each feature
            loss = torch.tensor(0.0, device=device)
            info = {}
            for name in feat_real.keys():
                f_real = feat_real[name].float()  # (F, Ng * Nr, D)
                f_fake = feat_fake[name].float()  # (F, Ng * Nf, D)
                # move the group dimension to the feature dimension
                # drifting field is computed independently for each group
                f_real = rearrange(f_real, "f (ng nr) d -> (f ng) nr d", ng=Ng, nr=Nr)  # (F * Ng, Nr, D)
                f_fake = rearrange(f_fake, "f (ng nf) d -> (f ng) nf d", ng=Ng, nf=Nf)  # (F * Ng, Nf, D)
                if feat_neg is not None:
                    f_neg = feat_neg[name].float()
                    f_neg = rearrange(f_neg, "f (ng nf) d -> (f ng) nf d", ng=Ng, nf=Nf)  # (F * Ng, Nf, D)
                    f_neg_arg = f_neg.detach()
                else:
                    f_neg_arg = None
                with torch.no_grad():
                    if conf.drifting.get("method", "joint") == "split":
                        V, _info = compute_drift_split(
                            x_real=f_real.detach(),
                            x_fake=f_fake.detach(),
                            x_neg=f_neg_arg,
                            eps=conf.drifting.eps,
                            plan_type=conf.drifting.plan_type,
                            sinkhorn_iters=conf.drifting.get("sinkhorn_iters", 20),
                            dist_metric=conf.drifting.get("dist_metric", "l2_sq"),
                            normalize_feature=conf.drifting.normalize_feature,
                            normalize_drift=conf.drifting.get("normalize_drift", False),
                            disable_self_mask=conf.drifting.get("disable_self_mask", False),
                            dim_temp_scale=conf.drifting.get("dim_temp_scale", False),
                        )
                    else:
                        V, _info = compute_drift(
                            x_real=f_real.detach(),
                            x_fake=f_fake.detach(),
                            x_neg=f_neg_arg,
                            kernel_temp=conf.drifting.kernel_temp,
                            kernel_norm=conf.drifting.kernel_norm,
                            normalize_feature=conf.drifting.normalize_feature,
                            normalize_drift=conf.drifting.normalize_drift,
                        )
                # regression loss
                f_fake = f_fake / max(_info["data-scale"], 1e-3)
                loss = loss + F.mse_loss(f_fake, (f_fake + V).detach())
                info = {**info, **{f"{name}-{k}": v for k, v in _info.items()}}
            loss = loss / len(feat_real)
            # Push current step's fake features to the rank-local memory bank
            # (only if neg_source == 'feature_bank'). Pushed AFTER drift compute
            # so the bank lags by one step → future steps see stale-but-fresh
            # negatives (avoids the trivial "sample from current step" identity).
            if neg_source == "feature_bank":
                with torch.no_grad():
                    for name, feat in feat_fake.items():
                        if name not in neg_bank:
                            neg_bank[name] = []
                        # Detach + bf16/half preserved + offload to CPU.
                        neg_bank[name].append(feat.detach().cpu())
                        # Trim oldest chunks to keep total ≤ bank_size samples.
                        total_samples = sum(t.shape[1] for t in neg_bank[name])
                        while total_samples > bank_size and len(neg_bank[name]) > 1:
                            removed = neg_bank[name].pop(0)
                            total_samples -= removed.shape[1]
            # backward
            loss = loss / grad_acc_steps
            loss.backward()
        # check gradient accumulation
        status = None
        grad_acc_counter += 1
        if grad_acc_counter == grad_acc_steps:
            grad_acc_counter = 0
            # clip gradients
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.train.clip_grad_norm)
            # update params
            optimizer.step()
            ema.update(model_wo_ddp)
            # update lr
            scheduler.step()
            # reduce stats for logging
            loss = reduce_tensor(loss.detach()) * grad_acc_steps
            grad_norm = reduce_tensor(grad_norm)
            status = dict(loss=loss.item(), grad_norm=grad_norm.item(), lr=optimizer.param_groups[0]["lr"], **info)
        return status

    @torch.no_grad()
    def sample(savepath: str):
        num_samples = math.ceil(64 / get_world_size())
        generator = torch.Generator(device).manual_seed(get_rank())
        z = torch.randn(num_samples, *input_shape, generator=generator, device=device)
        samples = ema.ema_model(z)
        samples = autoencoder.decode(samples)
        samples = torch.cat(gather_tensor(samples), dim=0)[:64]
        samples = (samples.clamp(-1, 1).cpu() + 1) / 2
        if is_main_process():
            save_image(samples, savepath, nrow=8)

    # START TRAINING
    logger.info("Start training...")
    dataiter = infinite_iterator()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    timing_start_step = step
    timing_start_time = time.perf_counter()

    def add_runtime_status(status: dict, current_step: int) -> dict:
        nonlocal timing_start_step, timing_start_time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        steps_elapsed = max(current_step - timing_start_step + 1, 1)
        avg_wall_sec = (now - timing_start_time) / steps_elapsed
        status["avg_wall_sec_per_step"] = avg_wall_sec
        status["avg_wall_steps_per_sec"] = 1.0 / max(avg_wall_sec, 1e-12)

        if torch.cuda.is_available():
            peak_alloc = torch.tensor(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                device=device,
            )
            peak_reserved = torch.tensor(
                torch.cuda.max_memory_reserved(device) / (1024 ** 3),
                device=device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(peak_alloc, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(peak_reserved, op=torch.distributed.ReduceOp.MAX)
            status["peak_mem_alloc_gb"] = peak_alloc.item()
            status["peak_mem_reserved_gb"] = peak_reserved.item()

        timing_start_step = current_step + 1
        timing_start_time = time.perf_counter()
        return status

    while step < conf.train.num_steps:
        batchdata = next(dataiter)
        # train a step
        model.train()
        train_status = train_step(batchdata)
        if train_status is None:
            continue
        if status_tracker.print_freq > 0 and (step + 1) % status_tracker.print_freq == 0:
            train_status = add_runtime_status(train_status, step)
        status_tracker.track_status("Train", train_status, step)
        wait_for_everyone()
        # validate
        model.eval()
        reset_timing_after_io = False
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, "ckpt", f"step{step:0>7d}"))
            wait_for_everyone()
            reset_timing_after_io = True
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, "samples", f"step{step:0>7d}.jpg"))
            wait_for_everyone()
            reset_timing_after_io = True
        if reset_timing_after_io:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            timing_start_step = step + 1
            timing_start_time = time.perf_counter()
        step += 1
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
