#!/usr/bin/env python3
"""Evaluate diffusion-based CBCT reconstruction.

Pipeline
--------
1. Load CBCT data (low-res 256³ or high-res 501³).
2. Compute FDK reconstruction.
3. (Optional) Gradient-descent baselines from zero / FDK initialisation.
4. Run M diffusion reconstructions (slice-wise DDIM + sinogram guidance).
5. Fine-tune each sample with GD; compute running averages.
6. Report PSNR / SSIM for first sample, per-run mean, and averaged volume.

Usage
-----
# Walnut 256³ – conditional
python -m cbct_diffusion.inference.reconstruct_diffusion \\
    --data_path <DATA_DIR>/walnut --cbct_id 1 --nviews 20 \\
    --diffusion_checkpoint checkpoints/Diffusion_Walnut_CBCT_256_ft20_cond.pt \\
    --conditioning

# Walnut 501³ – unconditional
python -m cbct_diffusion.inference.reconstruct_diffusion \\
    --data_path <DATA_DIR>/walnut --high_resolution --cbct_id 1 --nviews 20 \\
    --diffusion_checkpoint checkpoints/Diffusion_Walnut_CBCT_501.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from astra_torch import fdk_reconstruction_masked, gd_reconstruction_masked

from cbct_diffusion.data import SliceCBCTDataset, create_cbct_args, Walnut512
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.models.tomographic_cbct_diffusion import GuidanceConfig
from cbct_diffusion.schedulers import GuidedDDIMScheduler, DDIMPipeline
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
def _sinogram_guidance(gcfg, projs_vrc, vecs, vol_init, mask, device):
    return gd_reconstruction_masked(
        projs_vrc=projs_vrc.to(device),
        vecs=vecs,
        mask=mask,
        vol_init=vol_init.to(device),
        **gcfg.to_kwargs(),
    )


def _save_slice_images(volume, prefix, cbct_id):
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
# Data loading
# ------------------------------------------------------------------
def _load_data(args, device):
    if args.high_resolution:
        ds = Walnut512(
            data_path=args.data_path, split_subdir="Test", orbit_id=1,
            k=args.nviews, angular_sub_sampling=1, voxel_per_mm=10,
            device=torch.device("cuda"), axis=0,
            normalize_factor=get_dataset_clamp()["clamp_max"],
            walnut_range=(args.cbct_id, args.cbct_id + 1),
        )
        ds.rebuild_dataset(k=args.nviews)
        gt = ds.external_volumes[0].clone().cpu()
        projs = ds.projections[0]
        vecs = torch.from_numpy(ds.vecs[0])
        mask = ds.masks[0]
        fdk_vol = ds.reconstructions[0].clone()
        return projs, vecs, gt, fdk_vol.unsqueeze(0).unsqueeze(0), tuple(gt.shape), ds.voxel_per_mm, ds.voxel_size_mm, mask, ds
    else:
        cbct_args = create_cbct_args(datadir=args.data_path, nviews=args.nviews, start=0, end=360)
        ds = SliceCBCTDataset(
            args=cbct_args, stage="test", slice_axis="axial", preload_all=True,
            normalize_factor=get_dataset_clamp()["clamp_max"], augment=False,
        )
        idx = args.cbct_id
        gt = ds.gt_volumes[idx].clone().cpu()
        projs = ds.projections[idx]
        vecs = torch.from_numpy(ds.vecs[idx])
        mask = torch.ones(len(vecs)).bool()
        fdk_vol = ds.fdk_volumes[idx].clone()
        return projs, vecs, gt, fdk_vol.unsqueeze(0).unsqueeze(0), tuple(gt.shape), ds.voxel_per_mm, ds.voxel_size_mm, mask, ds


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def _load_diffusion_unet(checkpoint_path, image_size, device, input_channels):
    model = LatentUnet2D(
        compression=1, input_channels=input_channels, output_channels=1,
        sample_size=image_size, layers_per_block=2,
        block_out_channels=(64, 64, 128, 128, 256, 256),
        down_block_types=("DownBlock2D",) * 4 + ("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D") + ("UpBlock2D",) * 4,
        class_embed_type="timestep", num_class_embeds=image_size,
    )
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(f"WARNING: checkpoint not found at {checkpoint_path}")
    model.to(device).eval()
    return model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diffusion CBCT reconstruction evaluation")
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cbct_id", type=int, default=1)
    parser.add_argument("--orbit_id", type=int, default=2)
    parser.add_argument("--high_resolution", action="store_true")
    parser.add_argument("--nviews", type=int, default=20)
    # GD control
    parser.add_argument("--enable_gd_zero", action="store_true")
    parser.add_argument("--enable_gd_fdk", action="store_true")
    # GD params
    parser.add_argument("--gd_zero_epochs", type=int, nargs="+", default=[100, 50, 10])
    parser.add_argument("--gd_zero_lr", type=float, nargs="+", default=[5e-3, 5e-4, 1e-4])
    parser.add_argument("--gd_zero_batch_size", type=int, default=30)
    parser.add_argument("--gd_fdk_epochs", type=int, nargs="+", default=[100, 50, 10])
    parser.add_argument("--gd_fdk_lr", type=float, nargs="+", default=[5e-3, 5e-4, 1e-4])
    parser.add_argument("--gd_fdk_batch_size", type=int, default=30)
    # Diffusion
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--slice_batch_size", type=int, default=40)
    parser.add_argument("--diffusion_num_steps", type=int, default=50)
    parser.add_argument("--diffusion_runs", type=int, default=10)
    parser.add_argument("--scheduler_train_timesteps", type=int, default=1000)
    # Guidance
    parser.add_argument("--guidance_max_epochs", type=int, nargs="+", default=[5])
    parser.add_argument("--guidance_batch_size", type=int, default=30)
    parser.add_argument("--guidance_lr", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--guidance_clamp_min", type=float, default=0.0)
    parser.add_argument("--guidance_verbose", action="store_true")
    # W&B / IO
    parser.add_argument("--wandb_project", type=str, default="cbct-reconstructions-ddim")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./cbct_results")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    data_name = Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(data_name))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_wandb:
        run_name = f"cbct{args.cbct_id}_diff_{'cond_' if args.conditioning else ''}{'high' if args.high_resolution else 'low'}res"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    try:
        projs, vecs, gt, vol_fdk, vol_shape, vpm, vsm, mask, normalizer = _load_data(args, device)
        fdk = vol_fdk[0, 0]
        gt = gt.to(device)

        print(f"FDK PSNR: {psnr(gt, fdk):.3f}  SSIM: {ssim(gt, fdk):.4f}")
        results: Dict = {"cbct_id": args.cbct_id, "fdk_psnr": float(psnr(gt, fdk)), "fdk_ssim": float(ssim(gt, fdk))}

        # GD baselines
        if args.enable_gd_zero:
            t0 = time.time()
            r = gd_reconstruction_masked(
                projs_vrc=projs, vecs=vecs, mask=mask, voxel_per_mm=vpm, voxel_size_mm=vsm,
                vol_shape=vol_shape, max_epochs=args.gd_zero_epochs,
                batch_size=args.gd_zero_batch_size, lr=args.gd_zero_lr, verbose=True,
            ).detach().cpu()
            results["gd_zero_psnr"] = float(psnr(gt, r))
            results["gd_zero_ssim"] = float(ssim(gt, r))
            print(f"GD(zero) PSNR: {results['gd_zero_psnr']:.3f}  ({time.time()-t0:.1f}s)")

        if args.enable_gd_fdk:
            t0 = time.time()
            r = gd_reconstruction_masked(
                projs_vrc=projs, vecs=vecs, mask=mask, voxel_per_mm=vpm, voxel_size_mm=vsm,
                vol_shape=vol_shape, max_epochs=args.gd_fdk_epochs,
                batch_size=args.gd_fdk_batch_size, lr=args.gd_fdk_lr, vol_init=vol_fdk, verbose=True,
            ).detach().cpu()
            results["gd_fdk_psnr"] = float(psnr(gt, r))
            results["gd_fdk_ssim"] = float(ssim(gt, r))
            print(f"GD(FDK) PSNR: {results['gd_fdk_psnr']:.3f}  ({time.time()-t0:.1f}s)")

        # Load diffusion model
        image_size = fdk.shape[0]
        input_channels = 2 if args.conditioning else 1
        model = _load_diffusion_unet(args.diffusion_checkpoint, image_size, device, input_channels)

        # Build guidance + pipeline
        gcfg = GuidanceConfig(
            voxel_per_mm=vpm, voxel_size_mm=vsm,
            max_epochs=args.guidance_max_epochs, batch_size=args.guidance_batch_size,
            lr=args.guidance_lr, clamp_min=args.guidance_clamp_min, verbose=args.guidance_verbose,
        )

        def guidance_fn(x, t):
            return _sinogram_guidance(gcfg, projs, vecs, x, mask, device)

        scheduler = GuidedDDIMScheduler(
            num_train_timesteps=args.scheduler_train_timesteps, guidance_function=guidance_fn,
        )

        fdk_prior = fdk.to(device) if args.conditioning else None
        pipeline = DDIMPipeline(
            unet=model, scheduler=scheduler, fdk_prior=fdk_prior,
            normalize_fn=normalizer.normalize, denormalize_fn=normalizer.denormalize,
            slice_batch_size=args.slice_batch_size,
        )

        # Multi-run diffusion
        n_runs = args.diffusion_runs
        psnr_runs: List[float] = []
        ssim_runs: List[float] = []
        psnr_ft_runs: List[float] = []
        ssim_ft_runs: List[float] = []
        avg_raw: Optional[torch.Tensor] = None
        avg_ft: Optional[torch.Tensor] = None
        first_raw: Optional[torch.Tensor] = None
        first_ft: Optional[torch.Tensor] = None

        for i in range(n_runs):
            t0 = time.time()
            recon = pipeline(batch_size=gt.shape[0], num_inference_steps=args.diffusion_num_steps).images.detach()
            dt = time.time() - t0

            if i == 0:
                first_raw = recon.clone()
            psnr_runs.append(float(psnr(gt, recon)))
            ssim_runs.append(float(ssim(gt, recon)))

            # Finetune
            ft_cfg = GuidanceConfig(max_epochs=[10, 10], lr=[5e-4, 2e-4], voxel_per_mm=vpm, voxel_size_mm=vsm, verbose=False)
            recon_ft = gd_reconstruction_masked(
                projs_vrc=projs.to(device), vecs=vecs, mask=mask,
                vol_init=recon.unsqueeze(0).unsqueeze(0).to(device), **ft_cfg.to_kwargs(),
            ).detach()
            if i == 0:
                first_ft = recon_ft.clone()
            psnr_ft_runs.append(float(psnr(gt, recon_ft)))
            ssim_ft_runs.append(float(ssim(gt, recon_ft)))

            avg_raw = recon.clone() if avg_raw is None else avg_raw + (recon - avg_raw) / (i + 1)
            avg_ft = recon_ft.clone() if avg_ft is None else avg_ft + (recon_ft - avg_ft) / (i + 1)

            print(f"Run {i+1}/{n_runs}: raw PSNR {psnr_runs[-1]:.3f} | ft PSNR {psnr_ft_runs[-1]:.3f} ({dt:.1f}s)")
            torch.cuda.empty_cache()

        results.update({
            "diff_avg_psnr": float(psnr(gt, avg_raw)),
            "diff_avg_ssim": float(ssim(gt, avg_raw)),
            "diff_ft_avg_psnr": float(psnr(gt, avg_ft)),
            "diff_ft_avg_ssim": float(ssim(gt, avg_ft)),
            "diff_runs_psnr_mean": float(np.mean(psnr_runs)),
            "diff_runs_ssim_mean": float(np.mean(ssim_runs)),
            "diff_ft_runs_psnr_mean": float(np.mean(psnr_ft_runs)),
            "diff_ft_runs_ssim_mean": float(np.mean(ssim_ft_runs)),
        })

        # Summary
        print("\n" + "=" * 60)
        print(f"FDK:           PSNR {results['fdk_psnr']:.3f} | SSIM {results['fdk_ssim']:.4f}")
        print(f"Diff avg:      PSNR {results['diff_avg_psnr']:.3f} | SSIM {results['diff_avg_ssim']:.4f}")
        print(f"Diff+ft avg:   PSNR {results['diff_ft_avg_psnr']:.3f} | SSIM {results['diff_ft_avg_ssim']:.4f}")
        print("=" * 60)

        # Save
        if not args.no_wandb:
            wandb.log(results)
        _save_slice_images_to_disk(gt, "GT", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(fdk, "FDK", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(first_raw, "Diffusion_First", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(avg_raw, "Diffusion_Average", args.output_dir, args.cbct_id, args.orbit_id)
        _save_slice_images_to_disk(avg_ft, "Diffusion_FT_Average", args.output_dir, args.cbct_id, args.orbit_id)

        if not args.no_wandb:
            wandb.finish()
        print("Done!")

    except Exception:
        import traceback
        traceback.print_exc()
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()
