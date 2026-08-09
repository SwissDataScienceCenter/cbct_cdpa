#!/usr/bin/env python3
"""Probe: DiffusionMBIR (ADMM + 1D-TV-along-z guidance) on our existing
unconditional DPA checkpoint, no retraining. First-pass hyperparameter probe,
NOT the resumable sweep infrastructure -- reuses evaluate_volume.py's data
loading and scoring, but runs standalone so lambda/rho/iteration counts can be
swept quickly without touching the resumable JSON bookkeeping.

With --n_samples > 1, draws N independent posterior samples (different seeds)
and reports both the single-sample metrics and the mu(.) (running-mean)
metrics via Welford's algorithm, plus the mean pixel-wise posterior STD --
this is the direct test of whether the ADMM-TV constraint collapses sample
diversity relative to DPA/CDPA's own posterior spread (see
cbct_diffusion.inference.evaluate_volume.run_diffusion for the equivalent
DPA/CDPA computation this is meant to be compared against).

Usage
-----
python scripts/probe_diffusionmbir.py \\
    --data_path /mydata/sdate/shared/data/huggingface/walnut --cbct_id 0 --nviews 20 \\
    --diffusion_checkpoint /mydata/sdate/shared/checkpoints/cbct256/Diffusion_Walnut_CBCT_256_ft20.pt \\
    --lam 0.01 --rho 1.0 --n_admm_iters 3 --n_cg_iters 5 --n_samples 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import cbct_diffusion  # noqa: F401  (import-order guard)

from cbct_diffusion.inference.evaluate_volume import load_volume, load_unet, score
from cbct_diffusion.inference.admm_tv_guidance import ADMMTVGuidance
from cbct_diffusion.schedulers import GuidedDDIMScheduler, DDIMPipeline
from cbct_diffusion.utils.metrics import set_dataset_clamp, get_clamp_by_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--cbct_id", type=int, default=0)
    p.add_argument("--nviews", type=int, default=20)
    p.add_argument("--high_resolution", action="store_true")
    p.add_argument("--dataset_label", type=str, default=None)
    p.add_argument("--diffusion_checkpoint", required=True)
    p.add_argument("--conditional", action="store_true",
                   help="use the FDK-conditioned (CDPA) checkpoint + fdk_prior "
                        "instead of the unconditional (DPA) one -- tests whether "
                        "ADMM-TV guidance adds anything on top of conditioning")
    p.add_argument("--slice_batch_size", type=int, default=40)
    p.add_argument("--diffusion_num_steps", type=int, default=50)
    p.add_argument("--scheduler_train_timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1,
                   help="posterior samples to draw; >1 also reports mu(.) and "
                        "the mean pixel-wise posterior STD")
    # ADMM-TV hyperparameters -- the ones we're here to tune.
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--n_admm_iters", type=int, default=3)
    p.add_argument("--n_cg_iters", type=int, default=5)
    p.add_argument("--verbose_admm", action="store_true")
    args = p.parse_args()

    dataset = args.dataset_label or Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = load_volume(args, device)
    gt = d["gt"]
    print(f"loaded gt{tuple(gt.shape)}, vol_shape={d['vol_shape']}")

    model = load_unet(
        args.diffusion_checkpoint, d["fdk"].shape[0], device,
        2 if args.conditional else 1, with_time=True,
    )

    # Rebuilding the guidance object per sample would rebuild the ASTRA
    # geometry N times for no reason -- the projector/A^T y only depend on the
    # fixed (vecs, mask, y), not on the sample index, so build it once.
    guidance = ADMMTVGuidance(
        vecs=d["vecs"], mask=d["mask"], vol_shape=d["vol_shape"],
        det_rows=d["projs"].shape[1], det_cols=d["projs"].shape[2],
        voxel_size_mm=d["vsm"], y=d["projs"][d["mask"]].to(device),
        device=device, rho=args.rho, lam=args.lam,
        n_admm_iters=args.n_admm_iters, n_cg_iters=args.n_cg_iters,
        verbose=args.verbose_admm,
    )
    scheduler = GuidedDDIMScheduler(
        num_train_timesteps=args.scheduler_train_timesteps, guidance_function=guidance,
    )
    pipeline = DDIMPipeline(
        unet=model, scheduler=scheduler,
        fdk_prior=d["fdk"].to(device) if args.conditional else None,
        normalize_fn=d["normalizer"].normalize, denormalize_fn=d["normalizer"].denormalize,
        slice_batch_size=args.slice_batch_size,
    )

    tag = "CDPA+ADMM-TV" if args.conditional else "DPA+ADMM-TV (DiffusionMBIR)"
    torch.cuda.reset_peak_memory_stats()

    # Welford accumulators, matching evaluate_volume.run_diffusion's pattern --
    # mean and voxel-wise STD from a single pass, without holding all N
    # volumes in memory at once.
    mean = None
    m2 = None
    per_sample = []
    t_all = time.time()
    for i in range(args.n_samples):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        t0 = time.time()
        rec = pipeline(
            batch_size=gt.shape[0], num_inference_steps=args.diffusion_num_steps, generator=gen,
        ).images.detach()
        dt = time.time() - t0
        m = score(gt, rec)
        per_sample.append({"sample_index": i, "recon_time_s": dt, "psnr": m["psnr"], "ssim": m["ssim"]})
        print(f"  sample {i+1}/{args.n_samples}: PSNR {m['psnr']:.3f} SSIM {m['ssim']:.4f} ({dt:.1f}s)",
              flush=True)

        rec_c = rec.detach().cpu()
        if mean is None:
            mean = rec_c.clone()
            m2 = torch.zeros_like(rec_c)
        else:
            delta = rec_c - mean
            mean += delta / (i + 1)
            m2 += delta * (rec_c - mean)
        del rec, rec_c

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== {tag} probe: rho={args.rho} lam={args.lam} "
          f"admm_iters={args.n_admm_iters} cg_iters={args.n_cg_iters} n_samples={args.n_samples} ===")
    m0 = per_sample[0]
    print(f"sample 1  : PSNR {m0['psnr']:.3f}  SSIM {m0['ssim']:.4f}  ({m0['recon_time_s']:.1f}s)")

    if args.n_samples > 1:
        mu_metrics = score(gt, mean)
        std = (m2 / (args.n_samples - 1)).sqrt()
        mean_std = float(std.mean())
        print(f"mu(n={args.n_samples}): PSNR {mu_metrics['psnr']:.3f}  SSIM {mu_metrics['ssim']:.4f}  "
              f"tv_ratio ax {mu_metrics['tv_ratio_axial']:.3f} in-plane {mu_metrics['tv_ratio_inplane']:.3f} "
              f"excess {mu_metrics['tv_ratio_excess']:+.3f}")
        print(f"mean pixel-wise posterior STD: {mean_std:.6f}  "
              f"(mean sample PSNR: {sum(p['psnr'] for p in per_sample)/len(per_sample):.3f}, "
              f"PSNR gain from averaging: {mu_metrics['psnr'] - sum(p['psnr'] for p in per_sample)/len(per_sample):+.3f} dB)")

    print(f"total_time_s={time.time()-t_all:.1f}  peak_gpu_gb={peak_gb:.2f}")


if __name__ == "__main__":
    main()
