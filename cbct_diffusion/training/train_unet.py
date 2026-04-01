#!/usr/bin/env python3
"""Train a UNet for CBCT reconstruction (L2 loss on FDK → GT slices).

Usage examples
--------------
# Dental 256³ (HuggingFace dataset)
python -m cbct_diffusion.training.train_unet \\
    --data_path <DATA_DIR>/dental --image_size 256 \\
    --exp_name Unet_Dental_CBCT_256 --epochs 60 --batch_size 4

# Walnut 256³ (HuggingFace dataset)
python -m cbct_diffusion.training.train_unet \\
    --data_path <DATA_DIR>/walnut --image_size 256 \\
    --exp_name Unet_Walnut_CBCT_256 --epochs 60 --batch_size 4

# Walnut 501³ (full-resolution, raw data)
python -m cbct_diffusion.training.train_unet \\
    --data_path <DATA_DIR>/walnut --image_size 501 \\
    --exp_name Unet_Walnut_CBCT_501 --use_slice_idx --epochs 60 --batch_size 1

# Spine 256³ (HuggingFace dataset)
python -m cbct_diffusion.training.train_unet \\
    --data_path <DATA_DIR>/spine --image_size 256 \\
    --exp_name Unet_Spine_CBCT_256 --epochs 60 --batch_size 4
"""

import os
import random
from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup

from cbct_diffusion.data import SliceCBCTDataset, create_cbct_args, Walnut512
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.training import PyTorchExperiment, BaseLoss


def load_model(model: nn.Module, model_path: str) -> None:
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from checkpoint {model_path}")


class ReconstructionLoss(BaseLoss):
    """L2 loss between UNet prediction and ground-truth slice.

    Dataset formats supported:
    - SliceCBCTDataset: ``(slice_fdk, slice_gt, k, slice_idx)``
    - Walnut512:        ``(slice_fdk, slice_gt, k, slice_idx)``
    """

    def __init__(self, use_slice_idx: bool = False):
        super().__init__(["loss"])
        self.mse = nn.MSELoss()
        self.use_slice_idx = use_slice_idx

    def compute_loss(self, instance, model: LatentUnet2D):
        if len(instance) == 4:
            conditioning, target, k, slice_idx = instance
        else:
            conditioning, target, _, k, slice_idx, _ = instance

        device = next(model.parameters()).device
        conditioning = conditioning.unsqueeze(1).float().to(device)
        target = target.unsqueeze(1).float().to(device)
        k_tensor = k.to(device).view(-1).int()
        slice_tensor = slice_idx.to(device).view(-1).int()
        if not self.use_slice_idx:
            slice_tensor = torch.zeros_like(slice_tensor)

        pred = model(conditioning, timestep=k_tensor, class_labels=slice_tensor, return_dict=False)[0]
        loss = self.mse(pred, target)
        return loss, {"loss": loss.item()}


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return (train, test) datasets based on the data path layout."""
    if "huggingface" in args.data_path.lower():
        cbct_args = create_cbct_args(datadir=args.data_path, nviews=20, start=0, end=360)
        train_ds = SliceCBCTDataset(
            args=cbct_args, stage="train", slice_axis="axial",
            preload_all=True, augment=True, limit=args.load_limit,
        )
        test_ds = SliceCBCTDataset(
            args=cbct_args, stage="val", slice_axis="axial",
            preload_all=True, augment=False, limit=args.load_limit,
        )
    else:
        train_ds = Walnut512(
            data_path=args.data_path, split_subdir="Train",
            orbit_id=args.orbit_id, angular_sub_sampling=args.angular_sub_sampling,
            k=args.sparsity, voxel_per_mm=10, device=torch.device("cuda"),
            axis=0, limit=args.load_limit, augment=True,
        )
        test_ds = Walnut512(
            data_path=args.data_path, split_subdir="Test",
            orbit_id=args.orbit_id, angular_sub_sampling=args.angular_sub_sampling,
            k=args.sparsity, voxel_per_mm=10, device=torch.device("cuda"),
            axis=0, limit=args.load_limit, augment=False,
        )
    return train_ds, test_ds


def create_model(args) -> nn.Module:
    """Instantiate the LatentUnet2D architecture."""
    channels = (64, 64, 128, 128, 256, 256)
    return LatentUnet2D(
        compression=args.compression,
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


def main():
    parser = ArgumentParser(description="Train UNet on sparse CBCT slices")
    # Dataset / paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to CBCT dataset")
    parser.add_argument("--load_limit", type=int, default=-1)
    parser.add_argument("--angular_sub_sampling", type=int, default=2)
    parser.add_argument("--use_slice_idx", action="store_true")
    parser.add_argument("--orbit_id", type=int, default=1)
    parser.add_argument("--sparsity", type=int, default=None)
    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_milestones", type=str, default="[10,15]")
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="unet_cbct")
    parser.add_argument("--wandb", action="store_true")
    # Model
    parser.add_argument("--checkpoint_path", default="", type=str, help="Checkpoint to resume from")
    parser.add_argument("--compression", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)

    args = parser.parse_args()

    milestones = [int(x) for x in args.scheduler_milestones.strip("[] ").split(",") if x]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds = build_datasets(args)
    model = create_model(args)

    checkpoint_path = f"checkpoints/{args.exp_name}.pt"
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        try:
            load_model(model, checkpoint_path)
        except Exception:
            print(f"Could not load {checkpoint_path}, training from scratch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = ReconstructionLoss(use_slice_idx=args.use_slice_idx)

    os.makedirs("checkpoints", exist_ok=True)

    exp = PyTorchExperiment(
        args=vars(args), train_dataset=train_ds, test_dataset=test_ds,
        batch_size=args.batch_size, model=model, loss_fn=loss_fn,
        checkpoint_path=checkpoint_path, experiment_name=args.exp_name,
        with_wandb=args.wandb, seed=args.seed, save_always=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(exp.train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )
    exp.train(args.epochs, optimizer, milestones=milestones, gamma=args.lr_decay, scheduler=lr_scheduler)


if __name__ == "__main__":
    main()
