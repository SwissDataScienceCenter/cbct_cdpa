#!/usr/bin/env python3
"""Train a diffusion model for CBCT reconstruction (DDPM noise prediction).

Usage examples
--------------
# Walnut 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/walnut --image_size 256 \\
    --exp_name Diffusion_Walnut_CBCT_256_ft20 --epochs 60 --batch_size 8

# Walnut 256³ – conditional (FDK prior as 2nd channel)
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/walnut --image_size 256 --conditioning \\
    --exp_name Diffusion_Walnut_CBCT_256_ft20_cond --epochs 60 --batch_size 8

# Walnut 501³ – unconditional (full-resolution, raw data)
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/walnut --image_size 501 \\
    --exp_name Diffusion_Walnut_CBCT_501 --epochs 60 --batch_size 4

# Walnut 501³ – conditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/walnut --image_size 501 --conditioning \\
    --exp_name Diffusion_Walnut_CBCT_501_cond --epochs 60 --batch_size 4

# Dental 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/dental --image_size 256 \\
    --exp_name Diffusion_Dental_CBCT_256_ft20 --epochs 60 --batch_size 8

# Dental 256³ – conditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/dental --image_size 256 --conditioning \\
    --exp_name Diffusion_Dental_CBCT_256_ft20_cond --epochs 60 --batch_size 8

# Spine 256³ – unconditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/spine --image_size 256 \\
    --exp_name Diffusion_Spine_CBCT_256_ft20 --epochs 60 --batch_size 8

# Spine 256³ – conditional
python -m cbct_diffusion.training.train_diffusion \\
    --data_path <DATA_DIR>/spine --image_size 256 --conditioning \\
    --exp_name Diffusion_Spine_CBCT_256_ft20_cond --epochs 60 --batch_size 8
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from cbct_diffusion.data import SliceCBCTDataset, create_cbct_args, Walnut512
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.training import PyTorchExperiment, BaseLoss
from cbct_diffusion.utils.metrics import (
    get_clamp_by_name,
    get_dataset_clamp,
    set_dataset_clamp,
)


def load_model(model: nn.Module, model_path: str) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from checkpoint {model_path}")


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return (train, test) datasets; sets the global clamp from the data name."""
    data_name = Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(data_name))
    print(f"Using dataset clamp {get_dataset_clamp()}")

    if "huggingface" in args.data_path.lower():
        cbct_args = create_cbct_args(datadir=args.data_path, nviews=20, start=0, end=360)
        nf = get_dataset_clamp()["clamp_max"]
        train_ds = SliceCBCTDataset(
            args=cbct_args, stage="train", slice_axis="axial",
            preload_all=True, augment=True, normalize_factor=nf, limit=args.dataset_limit,
        )
        test_ds = SliceCBCTDataset(
            args=cbct_args, stage="val", slice_axis="axial",
            preload_all=True, augment=False, normalize_factor=nf, limit=args.dataset_limit,
        )
    else:
        nf = get_dataset_clamp()["clamp_max"]
        train_ds = Walnut512(
            data_path=args.data_path, split_subdir="Train",
            orbit_id=args.orbit_id, angular_sub_sampling=args.angular_sub_sampling,
            k=args.sparsity, voxel_per_mm=10, device=torch.device("cuda"),
            axis=0, limit=args.load_limit, normalize_factor=nf, augment=True,
        )
        test_ds = Walnut512(
            data_path=args.data_path, split_subdir="Test",
            orbit_id=args.orbit_id, angular_sub_sampling=args.angular_sub_sampling,
            k=args.sparsity, voxel_per_mm=10, device=torch.device("cuda"),
            axis=0, limit=args.load_limit, normalize_factor=nf, augment=False,
        )
    return train_ds, test_ds


def create_model(args) -> nn.Module:
    channels = (32, 32, 32, 32, 64, 64) if args.tiny else (64, 64, 128, 128, 256, 256)
    input_channels = 2 if args.conditioning else 1
    return LatentUnet2D(
        compression=args.compression,
        input_channels=input_channels,
        output_channels=1,
        sample_size=args.image_size,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
        class_embed_type="timestep",
        num_class_embeds=args.image_size,
    )


class DiffusionReconstructionLoss(BaseLoss):
    """DDPM noise-prediction loss on CBCT slices.

    Each training step:
    1. Sample random timestep *t*.
    2. Add noise to ground-truth slice at *t*.
    3. (Optionally) concatenate FDK prior as a second channel.
    4. Predict the added noise with the UNet.
    5. MSE between predicted and true noise.
    """

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device, use_conditioning: bool):
        super().__init__(["loss"])
        self.mse = nn.MSELoss()
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.use_conditioning = use_conditioning

    def compute_loss(self, instance, model: LatentUnet2D):
        if len(instance) == 4:
            fdk_prior, ground_truth, _, slice_idx = instance
        else:
            fdk_prior, ground_truth, _, _, slice_idx, _ = instance

        if ground_truth.dim() == 2:
            x_0 = ground_truth.unsqueeze(0).unsqueeze(0)
        elif ground_truth.dim() == 3:
            x_0 = ground_truth.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected ground_truth shape: {ground_truth.shape}")

        x_0 = x_0.float().to(self.device)
        noise = torch.randn_like(x_0)
        bsz = x_0.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        if self.use_conditioning:
            if fdk_prior.dim() == 2:
                fdk_cond = fdk_prior.unsqueeze(0).unsqueeze(0)
            elif fdk_prior.dim() == 3:
                fdk_cond = fdk_prior.unsqueeze(1)
            else:
                fdk_cond = fdk_prior
            fdk_cond = fdk_cond.float().to(self.device)
            x_t = torch.cat([x_t, fdk_cond], dim=1)

        slice_tensor = slice_idx.to(self.device).view(-1).int()
        noise_pred = model(x_t, timestep=timesteps, class_labels=slice_tensor, return_dict=False)[0]
        loss = self.mse(noise_pred, noise)
        return loss, {"loss": loss.item()}


def main():
    parser = ArgumentParser(description="Train diffusion model on sparse CBCT slices")
    # Dataset
    parser.add_argument("--data_path", type=str, required=True, help="Path to CBCT dataset")
    parser.add_argument("--dataset_limit", type=int, default=-1)
    parser.add_argument("--load_limit", type=int, default=-1)
    parser.add_argument("--angular_sub_sampling", type=int, default=2)
    parser.add_argument("--orbit_id", type=int, default=1)
    parser.add_argument("--sparsity", type=int, default=None)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="[500]")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="diffusion_cbct")
    parser.add_argument("--wandb", action="store_true")
    # Model
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--compression", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--conditioning", action="store_true", help="Concatenate FDK prior as 2nd input channel")

    args = parser.parse_args()

    milestones = [int(x) for x in args.scheduler.strip("[] ").split(",") if x]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds = build_datasets(args)
    model = create_model(args)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/ddpm_{args.exp_name}{'_tiny' if args.tiny else ''}.pt"
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        try:
            load_model(model, checkpoint_path)
        except Exception as e:
            print(f"Could not load {checkpoint_path}: {e}. Training from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    loss_fn = DiffusionReconstructionLoss(noise_scheduler, device, use_conditioning=args.conditioning)

    exp = PyTorchExperiment(
        args=vars(args), train_dataset=train_ds, test_dataset=test_ds,
        batch_size=args.batch_size, model=model, loss_fn=loss_fn,
        checkpoint_path=checkpoint_path, experiment_name=args.exp_name,
        with_wandb=args.wandb, seed=args.seed, save_always=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(exp.train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )
    exp.train(args.epochs, optimizer, milestones=milestones, gamma=args.lr_decay, scheduler=lr_scheduler)


if __name__ == "__main__":
    main()
