import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizers warning

from copy import deepcopy

import deepspeed
import torch
import torch.distributed as dist
import wandb
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_manager import get_data_parallel_group, set_parallel_manager
from opensora.datasets import DatasetFromCSV, get_transforms_image, get_transforms_video, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import update_ema


def is_master():
    return dist.get_rank() == 0


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    print(cfg)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"
    deepspeed.init_distributed()
    torch.cuda.set_device(cfg.local_rank)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # 2.1. colossalai init distributed training
    device = torch.device(get_accelerator().device_name(), cfg.local_rank)
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not is_master():
        logger = create_logger(None)
    else:
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="minisora", name=exp_name, config=cfg._cfg_dict)

    # 2.3. initialize ColossalAI booster
    set_parallel_manager(cfg.sp_size)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = DatasetFromCSV(
        cfg.data_path,
        transform=(
            get_transforms_video(cfg.image_size[0])
            if not cfg.use_image_transform
            else get_transforms_image(cfg.image_size[0])
        ),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
    )

    # a batch contains:
    # {
    #      "video": torch.Tensor,  # [B, C, T, H, W],
    #      "text": List[str],
    # }
    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")

    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    ds_config = {
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": cfg.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
                "adam_w_mode": True,
            }
        },
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 2,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }  # fmt:skip

    model, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()), config=ds_config
    )
    num_steps_per_epoch = len(dataloader)
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    running_loss = 0.0

    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(model, ema, cfg.load)
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    dataloader.sampler.set_start_index(sampler_start_idx)
    model_sharding(ema)

    # 6.2. training loop
    for epoch in range(start_epoch, cfg.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not dist.get_rank() == 0,
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                batch = next(dataloader_iter)
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
                y = batch["text"]

                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
                    model_args = text_encoder.encode(y)

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args)

                # Backward & update
                loss = loss_dict["loss"].mean()
                model.backward(loss)
                model.step()

                # Update EMA
                update_ema(ema, model.module)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1

                # Log to tensorboard
                if dist.get_rank == 0 and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)
                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "num_samples": global_step * total_batch_size,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                            },
                            step=global_step,
                        )

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save(
                        model,
                        ema,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        exp_dir,
                        ema_shape_dict,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0


if __name__ == "__main__":
    main()
