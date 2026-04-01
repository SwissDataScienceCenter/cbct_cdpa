#!/usr/bin/env python3
"""Evaluate UNet-based CBCT reconstruction.

Pipeline
--------
1. Load CBCT data (low-res 256³ or high-res 501³).
2. Compute FDK reconstruction.
3. (Optional) Gradient-descent baselines from zero / FDK initialisation.
4. Run slice-wise UNet inference on the FDK volume.
5. Gradient-descent fine-tuning from the UNet prediction.
6. Report PSNR / SSIM metrics and log to W&B.

Usage
-----
# Low resolution (256³)
python -m cbct_diffusion.inference.reconstruct_unet \\
    --data_path <DATA_DIR>/walnut --cbct_id 1 --nviews 20 \\
    --unet_checkpoint checkpoints/Unet_Walnut_CBCT_256.pt

# High resolution (501³)
python -m cbct_diffusion.inference.reconstruct_unet \\
    --data_path <DATA_DIR>/walnut --high_resolution --cbct_id 1 --nviews 20 \\
    --unet_checkpoint checkpoints/Unet_Walnut_CBCT_501.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from astra_torch import fdk_reconstruction_masked, gd_reconstruction_masked

from cbct_diffusion.data import SliceCBCTDataset, create_cbct_args, Walnut512
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.utils.metrics import (
    cbct_psnr as psnr,
    cbct_ssim_3d_gaal as ssim,
    set_dataset_clamp,
    get_dataset_clamp,
    get_clamp_by_name,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
class _SliceDataset(Dataset):
    """Wraps a 3D volume so that each item is a single axial slice."""

    def __init__(self, volume: torch.Tensor):
        self.volume = volume

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, idx):
        return self.volume[idx].unsqueeze(0), idx


def _save_slice_images(volume, prefix, cbct_id):
    """Save central axial/coronal/sagittal slices as W&B images."""
    v = volume.cpu().numpy()
    nx, ny, nz = v.shape
    clamp = get_dataset_clamp()
    images = {}
    for data, name in [
        (v[:, :, nz // 2], "axial"),
        (v[:, ny // 2, :], "coronal"),
        (v[nx // 2, :, :], "sagittal"),
    ]:
        data = np.clip(data, clamp["clamp_min"], clamp["clamp_max"])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(data, cmap="gray", vmin=clamp["clamp_min"], vmax=clamp["clamp_max"])
        ax.set_title(f"{prefix} - {name}")
        ax.axis("off")
        images[f"{prefix}_{name}"] = wandb.Image(fig)
        plt.close(fig)
    return images


def _save_slice_images_to_disk(volume, prefix, output_dir, cbct_id, orbit_id):
    v = volume.cpu().numpy()
    nx, ny, nz = v.shape
    clamp = get_dataset_clamp()
    os.makedirs(output_dir, exist_ok=True)
    for data, name in [
        (v[:, :, nz // 2], "axial"),
        (v[:, ny // 2, :], "coronal"),
        (v[nx // 2, :, :], "sagittal"),
    ]:
        data = np.clip(data, clamp["clamp_min"], clamp["clamp_max"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
        ax.imshow(data, cmap="gray", vmin=clamp["clamp_min"], vmax=clamp["clamp_max"])
        ax.set_title(f"{prefix} - {name}\nCBCT {cbct_id}, Orbit {orbit_id}")
        ax.axis("off")
        fname = f"cbct{cbct_id}_orbit{orbit_id}_{prefix}_{name}.png"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ------------------------------------------------------------------
# Model / data loading
# ------------------------------------------------------------------
def _load_unet(checkpoint_path: str, image_size: int, device: torch.device) -> LatentUnet2D:
    model = LatentUnet2D(
        compression=1, sample_size=image_size, layers_per_block=2,
        block_out_channels=(64, 64, 128, 128, 256, 256),
        down_block_types=("DownBlock2D",) * 4 + ("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D") + ("UpBlock2D",) * 4,
        class_embed_type="timestep", num_class_embeds=image_size,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def _load_data(args, device):
    if args.high_resolution:
        ds = Walnut512(
            data_path=args.data_path, split_subdir="Test", orbit_id=1,
            angular_sub_sampling=1, voxel_per_mm=10, device=torch.device("cuda"),
            axis=0, normalize_factor=get_dataset_clamp()["clamp_max"],
            walnut_range=(args.cbct_id, args.cbct_id + 1),
        )
        ds.rebuild_dataset(k=args.nviews)
        gt = ds.external_volumes[0].clone().cpu()
        projs = ds.projections[0]
        vecs = torch.from_numpy(ds.vecs[0])
        mask = torch.from_numpy(ds.masks[0])
        fdk_vol = ds.reconstructions[0].clone()
        vpm, vsm = ds.voxel_per_mm, ds.voxel_size_mm
    else:
        cbct_args = create_cbct_args(datadir=args.data_path, nviews=args.nviews, start=0, end=360)
        ds = SliceCBCTDataset(
            args=cbct_args, stage="test", slice_axis="axial", preload_all=True,
            normalize_factor=get_dataset_clamp()["clamp_max"], augment=False,
        )
        idx = min(args.cbct_id, len(ds.gt_volumes) - 1)
        gt = ds.gt_volumes[idx].clone().cpu()
        projs = ds.projections[idx]
        vecs = torch.from_numpy(ds.vecs[idx])
        mask = torch.ones(len(vecs)).bool()
        fdk_vol = ds.fdk_volumes[idx].clone()
        vpm, vsm = ds.voxel_per_mm, ds.voxel_size_mm
    return projs, vecs, gt, fdk_vol, vpm, vsm, mask


# ------------------------------------------------------------------
# UNet inference
# ------------------------------------------------------------------
def _run_unet(model, fdk, mask, device, batch_size, no_slice_label):
    nz = fdk.shape[0]
    loader = DataLoader(_SliceDataset(fdk), batch_size=batch_size, shuffle=False)
    out = [None] * nz
    with torch.no_grad():
        t_val = torch.sum(mask).int()
        for slices, idxs in tqdm(loader, desc="UNet inference"):
            slices = slices.to(device)
            labels = idxs.to(device).int()
            if no_slice_label:
                labels = labels * 0
            pred = model(slices, timestep=t_val, class_labels=labels, return_dict=False)[0]
            for i, idx in enumerate(idxs):
                out[idx] = pred[i, 0].cpu()
    return torch.stack(out, dim=0)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UNet CBCT reconstruction evaluation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cbct_id", type=int, default=1)
    parser.add_argument("--orbit_id", type=int, default=2)
    parser.add_argument("--high_resolution", action="store_true")
    parser.add_argument("--nviews", type=int, default=20)
    # GD control
    parser.add_argument("--enable_gd_zero", action="store_true")
    parser.add_argument("--enable_gd_fdk", action="store_true")
    # GD zero-init
    parser.add_argument("--gd_zero_epochs", type=int, nargs="+", default=[30, 30])
    parser.add_argument("--gd_zero_lr", type=float, nargs="+", default=[2e-3, 2e-4])
    parser.add_argument("--gd_zero_batch_size", type=int, default=30)
    # GD FDK-init
    parser.add_argument("--gd_fdk_epochs", type=int, nargs="+", default=[10])
    parser.add_argument("--gd_fdk_lr", type=float, nargs="+", default=[2e-3])
    parser.add_argument("--gd_fdk_batch_size", type=int, default=30)
    # UNet
    parser.add_argument("--unet_checkpoint", type=str, required=True)
    parser.add_argument("--unet_batch_size", type=int, default=10)
    # GD finetune
    parser.add_argument("--gd_finetune_epochs", type=int, nargs="+", default=[10])
    parser.add_argument("--gd_finetune_lr", type=float, nargs="+", default=[2e-4])
    parser.add_argument("--gd_finetune_batch_size", type=int, default=30)
    # W&B / IO
    parser.add_argument("--wandb_project", type=str, default="cbct-reconstructions")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./cbct_results")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    data_name = Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(data_name))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=f"{data_name}{args.cbct_id}_unet", config=vars(args))

    try:
        projs, vecs, gt, fdk_vol, vpm, vsm, mask = _load_data(args, device)

        vol_fdk = fdk_reconstruction_masked(
            projs_vrc=projs.to(device), vecs=vecs, mask=mask,
            voxel_per_mm=vpm, voxel_size_mm=vsm, device=device,
        ).unsqueeze(0).unsqueeze(0)

        fdk = vol_fdk[0, 0].cpu()
        vol_shape = tuple(gt.shape)
        print(f"FDK PSNR: {psnr(gt, fdk):.3f}  SSIM: {ssim(gt, fdk):.4f}")

        # Optional GD baselines
        results = {"cbct_id": args.cbct_id, "fdk_psnr": float(psnr(gt, fdk)), "fdk_ssim": float(ssim(gt, fdk))}

        if args.enable_gd_zero:
            t0 = time.time()
            r = gd_reconstruction_masked(
                projs_vrc=projs, vecs=vecs, mask=mask, voxel_per_mm=vpm, voxel_size_mm=vsm,
                vol_shape=vol_shape, max_epochs=args.gd_zero_epochs, batch_size=args.gd_zero_batch_size,
                lr=args.gd_zero_lr, verbose=True,
            ).detach().cpu()
            results["gd_zero_psnr"] = float(psnr(gt, r))
            results["gd_zero_ssim"] = float(ssim(gt, r))
            print(f"GD(zero) PSNR: {results['gd_zero_psnr']:.3f}  SSIM: {results['gd_zero_ssim']:.4f}  ({time.time()-t0:.1f}s)")

        if args.enable_gd_fdk:
            t0 = time.time()
            r = gd_reconstruction_masked(
                projs_vrc=projs, vecs=vecs, mask=mask, voxel_per_mm=vpm, voxel_size_mm=vsm,
                vol_shape=vol_shape, max_epochs=args.gd_fdk_epochs, batch_size=args.gd_fdk_batch_size,
                lr=args.gd_fdk_lr, vol_init=vol_fdk, verbose=True,
            ).detach().cpu()
            results["gd_fdk_psnr"] = float(psnr(gt, r))
            results["gd_fdk_ssim"] = float(ssim(gt, r))
            print(f"GD(FDK) PSNR: {results['gd_fdk_psnr']:.3f}  SSIM: {results['gd_fdk_ssim']:.4f}  ({time.time()-t0:.1f}s)")

        # UNet
        model = _load_unet(args.unet_checkpoint, fdk.shape[0], device)
        t0 = time.time()
        no_slice = "no_slice" in args.unet_checkpoint
        unet_vol = _run_unet(model, fdk, mask, device, args.unet_batch_size, no_slice)
        results["unet_psnr"] = float(psnr(gt, unet_vol))
        results["unet_ssim"] = float(ssim(gt, unet_vol))
        print(f"UNet PSNR: {results['unet_psnr']:.3f}  SSIM: {results['unet_ssim']:.4f}  ({time.time()-t0:.1f}s)")

        # GD finetune from UNet
        t0 = time.time()
        recon_ft = gd_reconstruction_masked(
            projs_vrc=projs, vecs=vecs, mask=mask, voxel_per_mm=vpm, voxel_size_mm=vsm,
            vol_shape=vol_shape, max_epochs=args.gd_finetune_epochs,
            batch_size=args.gd_finetune_batch_size, lr=args.gd_finetune_lr,
            vol_init=unet_vol.unsqueeze(0).unsqueeze(0), verbose=True,
        ).detach().cpu()
        results["gd_finetune_psnr"] = float(psnr(gt, recon_ft))
        results["gd_finetune_ssim"] = float(ssim(gt, recon_ft))
        print(f"GD(UNet-ft) PSNR: {results['gd_finetune_psnr']:.3f}  SSIM: {results['gd_finetune_ssim']:.4f}  ({time.time()-t0:.1f}s)")

        # Log & save
        if not args.no_wandb:
            wandb.log(results)
        _save_slice_images_to_disk(gt, "GT", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(fdk, "FDK", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(unet_vol, "UNet", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(recon_ft, "GD_Finetune", args.output_dir, args.cbct_id, args.orbit_id)

        if not args.no_wandb:
            wandb.finish()
        print("Done!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()
