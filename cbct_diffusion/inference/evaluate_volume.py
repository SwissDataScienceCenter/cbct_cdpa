#!/usr/bin/env python3
"""Reconstruct and score ONE test volume with ONE method group.

This is the RunAI worker for the 256^3 evaluation suite. It exists alongside
``reconstruct_diffusion.py`` / ``reconstruct_unet.py`` rather than replacing
them, because a table-generating sweep needs three things those scripts do not
provide:

1. **Persisted volumes.** The originals logged only mid-slice PNGs, so any new
   metric (per-axis SSIM, LPIPS, segmentation Dice) required re-running
   inference. Here every reconstruction is written to ``--output_dir`` as a
   compressed ``.npz``, so downstream metrics become offline post-processing.
2. **Resumability.** RunAI jobs are preemptible. Each (volume, method) pair
   writes its own result JSON and is skipped if that JSON already exists, so a
   relaunched job resumes instead of redoing finished work.
3. **Per-axis metrics.** PSNR/SSIM are recorded per orthogonal plane, not only
   as the 3-axis average, plus the adjacent-slice TV ratio. See
   ``cbct_diffusion.utils.metrics`` for the axis convention (axis 0 = axial =
   the plane the 2D model operates in).

One process handles one ``--method_group`` for one ``--cbct_id``:

``classical``  FDK, GD from zero, GD from FDK, U-Net denoiser, U-Net + FT
``dpa``        unconditional diffusion: single sample and the mean of N
``cdpa``       conditional diffusion: single sample and the mean of N

Splitting this way keeps the cheap deterministic work (~90 s) out of the same
job as the expensive sampling (~30-40 min), so a preemption costs less.

Usage
-----
python -m cbct_diffusion.inference.evaluate_volume \\
    --data_path /mydata/sdate/shared/data/huggingface/walnut \\
    --cbct_id 0 --nviews 20 --method_group cdpa --n_samples 20 \\
    --diffusion_checkpoint $CKPT_DIR/Diffusion_Walnut_CBCT_256_ft20_cond.pt \\
    --output_dir /mydata/sdate/shared/results/cbct256
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# Import cbct_diffusion (and therefore diffusers) BEFORE astra_torch. Reversing
# these two lines makes every run of this script die with a torch
# ``_c10d_functional``/``wait_tensor`` double-registration RuntimeError -- see
# the import-order guard in cbct_diffusion/__init__.py for the details.
import cbct_diffusion  # noqa: F401  (import for side effect: ordering guard)

from astra_torch import gd_reconstruction_masked

from cbct_diffusion.data import SliceCBCTDataset, create_cbct_args
from cbct_diffusion.inference.admm_tv_guidance import ADMMTVGuidance
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.models.tomographic_cbct_diffusion import GuidanceConfig
from cbct_diffusion.schedulers import GuidedDDIMScheduler, DDIMPipeline
from cbct_diffusion.utils.metrics import (
    cbct_psnr,
    cbct_psnr_per_axis,
    cbct_ssim_3d_gaal,
    inter_slice_consistency,
    slice_bias_jitter,
    set_dataset_clamp,
    get_dataset_clamp,
    get_clamp_by_name,
)


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
def score(gt: torch.Tensor, pred: torch.Tensor, n_jobs: int = -1) -> Dict[str, float]:
    """Full metric bundle for one reconstruction against the reference."""
    gt_c = gt.detach().cpu()
    pred_c = pred.detach().cpu()
    out: Dict[str, float] = {"psnr": float(cbct_psnr(gt_c, pred_c))}
    out.update(cbct_ssim_3d_gaal(gt_c, pred_c, return_per_axis=True, n_jobs=n_jobs))
    out.update(cbct_psnr_per_axis(gt_c, pred_c))
    out.update(inter_slice_consistency(gt_c, pred_c))
    out.update(slice_bias_jitter(gt_c, pred_c))
    return out


def _result_path(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str) -> Path:
    return output_dir / "metrics" / f"{dataset}_n{nviews}_id{cbct_id}_{method}.json"


def _volume_path(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str) -> Path:
    return output_dir / "volumes" / f"{dataset}_n{nviews}_id{cbct_id}_{method}.npz"


def _save_npz(path: Path, array: np.ndarray) -> None:
    """Atomically write a compressed .npz.

    Writes through a temp file so a preemption mid-write cannot leave a
    half-written volume behind. The temp file is opened as a handle rather than
    passed as a name because ``np.savez_compressed`` appends ``.npz`` to any
    filename that does not already end in it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, volume=array)
    tmp.replace(path)


def emit(
    output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str,
    gt: torch.Tensor, pred: torch.Tensor, extra: Optional[Dict] = None,
    save_volume: bool = True, n_jobs: int = -1,
) -> Dict[str, float]:
    """Score *pred*, write its JSON and (optionally) the volume itself."""
    t0 = time.time()
    metrics = score(gt, pred, n_jobs=n_jobs)
    metrics.update({
        "dataset": dataset, "cbct_id": cbct_id, "nviews": nviews, "method": method,
        "scoring_time_s": time.time() - t0,
        **(extra or {}),
    })

    rp = _result_path(output_dir, dataset, cbct_id, nviews, method)
    rp.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file so a preemption mid-write cannot leave a truncated
    # JSON that a later run would mistake for a completed result.
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.replace(rp)

    if save_volume:
        vp = _volume_path(output_dir, dataset, cbct_id, nviews, method)
        _save_npz(vp, pred.detach().cpu().numpy().astype(np.float32))

    print(
        f"  [{method:22s}] PSNR {metrics['psnr']:6.3f} | SSIM {metrics['ssim']:.4f} "
        f"(ax {metrics['ssim_axial']:.4f} / off {metrics['ssim_offaxis']:.4f}, "
        f"gap {metrics['ssim_axial_gap']:+.4f}) | tv_ratio ax {metrics['tv_ratio_axial']:.3f} "
        f"in-plane {metrics['tv_ratio_inplane']:.3f}",
        flush=True,
    )
    return metrics


def done(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str) -> bool:
    p = _result_path(output_dir, dataset, cbct_id, nviews, method)
    if not p.is_file():
        return False
    try:
        json.loads(p.read_text())
        return True
    except json.JSONDecodeError:
        print(f"  (corrupt result {p.name} — recomputing)")
        return False


# ------------------------------------------------------------------
# Data / model
# ------------------------------------------------------------------
def load_volume_high_res(args, device):
    """Load one 501^3 raw-TIFF walnut test volume (the ``--high_resolution`` path).

    Mirrors ``reconstruct_diffusion.py``'s high-resolution branch: ``orbit_id=1``
    is hardcoded (not read from ``args.orbit_id``, which only ever affected
    output filenames there) because that is what the archived W&B runs for the
    published N=30 walnut_metrics table actually used -- confirmed from their
    logged config, cross-checked against the two release scripts that both
    hardcode the same literal. See ``Walnut512.normalize/denormalize`` for a
    related regression this depends on (those methods were missing from the
    released class; restored in cbct_diffusion/data/walnut512.py).
    """
    from cbct_diffusion.data import Walnut512

    ds = Walnut512(
        data_path=args.data_path, split_subdir="Test", orbit_id=1,
        k=args.nviews, angular_sub_sampling=1, voxel_per_mm=10,
        device=device, axis=0,
        normalize_factor=get_dataset_clamp()["clamp_max"],
        walnut_range=(args.cbct_id, args.cbct_id + 1),
    )
    # rebuild_dataset reseeds its own RNG from self.seed on every call, so the
    # constructor's internal rebuild_dataset(k=self.k) above already produced
    # the final, deterministic mask -- a second call would just repeat it.
    gt = ds.external_volumes[0].clone()
    fdk = ds.reconstructions[0].clone()
    projs = ds.projections[0]
    vecs_np = ds.vecs[0].astype(np.float32)
    mask = torch.from_numpy(ds.masks[0]).bool()

    return dict(
        gt=gt.to(device), fdk=fdk, projs=projs, vecs=torch.from_numpy(vecs_np),
        mask=mask, vol_shape=tuple(gt.shape),
        vpm=ds.voxel_per_mm, vsm=ds.voxel_size_mm, normalizer=ds,
    )


def load_volume(args, device):
    """Load one test volume, its projections, geometry and FDK reconstruction."""
    if getattr(args, "high_resolution", False):
        return load_volume_high_res(args, device)

    cbct_args = create_cbct_args(datadir=args.data_path, nviews=args.nviews, start=0, end=360)
    ds = SliceCBCTDataset(
        args=cbct_args, stage="test", slice_axis="axial", preload_all=False,
        normalize_factor=get_dataset_clamp()["clamp_max"], augment=False,
    )
    # preload_all=False avoids loading and FDK-reconstructing all 20 test
    # volumes just to reach one of them (the originals paid that cost on every
    # run). Index the underlying dataset directly instead; ds.cbct_dataset[i]
    # resolves to the same test-split entry that ds.gt_volumes[i] would have.
    sample = ds.cbct_dataset[args.cbct_id]
    gt = sample["3Dvolume"]
    projs = sample["images"].squeeze(1)
    vecs_np = sample["poses"].detach().cpu().numpy().astype(np.float32)

    from astra_torch import fdk_reconstruction_masked
    import torch.nn.functional as F

    fdk = fdk_reconstruction_masked(
        projs_vrc=projs, vecs=vecs_np, mask=None,
        voxel_per_mm=ds.voxel_per_mm, voxel_size_mm=ds.voxel_size_mm, device=device,
    )
    if fdk.shape != gt.shape:
        fdk = F.interpolate(
            fdk.unsqueeze(0).unsqueeze(0), size=gt.shape,
            mode="trilinear", align_corners=False,
        ).squeeze(0).squeeze(0)

    return dict(
        gt=gt.to(device), fdk=fdk, projs=projs, vecs=torch.from_numpy(vecs_np),
        mask=torch.ones(len(vecs_np)).bool(), vol_shape=tuple(gt.shape),
        vpm=ds.voxel_per_mm, vsm=ds.voxel_size_mm, normalizer=ds,
    )


def load_unet(checkpoint_path, image_size, device, input_channels, with_time: bool):
    """Instantiate LatentUnet2D and load a checkpoint, matching the paper's config."""
    model = LatentUnet2D(
        compression=1, input_channels=input_channels, output_channels=1,
        sample_size=image_size, layers_per_block=2,
        block_out_channels=(64, 64, 128, 128, 256, 256),
        down_block_types=("DownBlock2D",) * 4 + ("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D") + ("UpBlock2D",) * 4,
        class_embed_type="timestep", num_class_embeds=image_size,
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  note: {len(missing)} missing keys in checkpoint (e.g. {missing[:3]})")
    model.to(device).eval()
    return model


# ------------------------------------------------------------------
# Method groups
# ------------------------------------------------------------------
def run_classical(args, d, device, output_dir, dataset):
    """FDK, the two GD baselines, and the FDK-denoiser with and without FT."""
    gt = d["gt"]
    common = dict(output_dir=output_dir, dataset=dataset, cbct_id=args.cbct_id,
                  nviews=args.nviews, gt=gt, n_jobs=args.ssim_jobs)

    if not done(output_dir, dataset, args.cbct_id, args.nviews, "fdk"):
        emit(method="fdk", pred=d["fdk"], **common)

    gd_kw = dict(projs_vrc=d["projs"], vecs=d["vecs"], mask=d["mask"],
                 voxel_per_mm=d["vpm"], voxel_size_mm=d["vsm"], vol_shape=d["vol_shape"],
                 max_epochs=args.gd_epochs, batch_size=args.gd_batch_size,
                 lr=args.gd_lr, verbose=False)

    # The GD baselines are deterministic given the schedule and take ~30 s to
    # recompute, so their volumes are not persisted by default -- they would add
    # ~40 MB x 2 x 45 volumes for little benefit. Pass --save_gd_volumes to keep
    # them (e.g. if they are wanted for a figure).
    if not done(output_dir, dataset, args.cbct_id, args.nviews, "gd_zero"):
        t0 = time.time()
        r = gd_reconstruction_masked(**gd_kw).detach()
        emit(method="gd_zero", pred=r, extra={"recon_time_s": time.time() - t0},
             save_volume=args.save_gd_volumes, **common)
        del r; torch.cuda.empty_cache()

    if not done(output_dir, dataset, args.cbct_id, args.nviews, "gd_fdk"):
        t0 = time.time()
        r = gd_reconstruction_masked(
            vol_init=d["fdk"].unsqueeze(0).unsqueeze(0), **gd_kw
        ).detach()
        emit(method="gd_fdk", pred=r, extra={"recon_time_s": time.time() - t0},
             save_volume=args.save_gd_volumes, **common)
        del r; torch.cuda.empty_cache()

    need_unet = not done(output_dir, dataset, args.cbct_id, args.nviews, "unet")
    need_ft = not done(output_dir, dataset, args.cbct_id, args.nviews, "unet_ft")
    if not (need_unet or need_ft):
        return

    model = load_unet(args.unet_checkpoint, d["fdk"].shape[0], device, 1, with_time=False)
    D = d["fdk"].shape[0]

    # NOTE: the FDK-denoiser operates on *raw* attenuation values, unlike the
    # diffusion models. train_unet.py builds its SliceCBCTDataset without a
    # normalize_factor (so it defaults to 1.0), and both reconstruct_unet.py
    # and the original chip script feed the unnormalised FDK volume straight in.
    # Dividing by clamp_max here costs ~8 dB.
    #
    # The denoiser is conditioned on the *number of views* through the timestep
    # embedding (t = sum(mask)) and on the slice index through class_labels --
    # see _run_unet in reconstruct_unet.py. Passing t=0 changes the
    # conditioning and the reported numbers.
    t_views = torch.sum(d["mask"]).int()
    # Checkpoints trained without slice conditioning expect a zeroed label;
    # the ablation checkpoints are the ones with "no_slice" in the filename.
    zero_labels = "no_slice" in os.path.basename(args.unet_checkpoint)

    t0 = time.time()
    with torch.no_grad():
        chunks = []
        for s in range(0, D, args.unet_batch_size):
            e = min(s + args.unet_batch_size, D)
            x = d["fdk"][s:e].unsqueeze(1).to(device)
            idx = torch.arange(s, e, device=device).int()
            if zero_labels:
                idx = idx * 0
            chunks.append(
                model(x, timestep=t_views, class_labels=idx, return_dict=False)[0][:, 0]
            )
        unet_vol = torch.cat(chunks, dim=0)
    unet_time = time.time() - t0
    del model, chunks; torch.cuda.empty_cache()

    if need_unet:
        emit(method="unet", pred=unet_vol, extra={"recon_time_s": unet_time}, **common)

    if need_ft:
        t0 = time.time()
        ft = gd_reconstruction_masked(
            projs_vrc=d["projs"].to(device), vecs=d["vecs"], mask=d["mask"],
            vol_shape=d["vol_shape"],
            vol_init=unet_vol.unsqueeze(0).unsqueeze(0).to(device),
            voxel_per_mm=d["vpm"], voxel_size_mm=d["vsm"],
            max_epochs=args.ft_epochs, batch_size=args.ft_batch_size,
            lr=args.ft_lr, verbose=False,
        ).detach()
        emit(method="unet_ft", pred=ft, extra={"recon_time_s": time.time() - t0}, **common)


def run_diffusion(args, d, device, output_dir, dataset, conditional: bool):
    """N posterior samples; score sample #1, the running mean, and the voxel STD."""
    tag = "cdpa" if conditional else "dpa"
    gt = d["gt"]
    common = dict(output_dir=output_dir, dataset=dataset, cbct_id=args.cbct_id,
                  nviews=args.nviews, gt=gt, n_jobs=args.ssim_jobs)

    mean_method = f"mu_{tag}_n{args.n_samples}"
    if done(output_dir, dataset, args.cbct_id, args.nviews, tag) and \
       done(output_dir, dataset, args.cbct_id, args.nviews, mean_method):
        print(f"  {tag}: already complete, nothing to do")
        return

    model = load_unet(
        args.diffusion_checkpoint, d["fdk"].shape[0], device,
        2 if conditional else 1, with_time=True,
    )

    gcfg = GuidanceConfig(
        voxel_per_mm=d["vpm"], voxel_size_mm=d["vsm"],
        max_epochs=args.guidance_max_epochs, batch_size=args.guidance_batch_size,
        lr=args.guidance_lr, clamp_min=args.guidance_clamp_min, verbose=False,
    )

    def guidance_fn(x, t):
        return gd_reconstruction_masked(
            projs_vrc=d["projs"].to(device), vecs=d["vecs"], mask=d["mask"],
            vol_init=x.to(device), **gcfg.to_kwargs(),
        )

    scheduler = GuidedDDIMScheduler(
        num_train_timesteps=args.scheduler_train_timesteps, guidance_function=guidance_fn,
    )
    pipeline = DDIMPipeline(
        unet=model, scheduler=scheduler,
        fdk_prior=d["fdk"].to(device) if conditional else None,
        normalize_fn=d["normalizer"].normalize, denormalize_fn=d["normalizer"].denormalize,
        slice_batch_size=args.slice_batch_size,
    )

    # Welford accumulators so the mean and the voxel-wise STD (used for the
    # uncertainty analysis) come out of the same pass without holding all N
    # volumes in memory.
    mean = None
    m2 = None
    per_sample = []
    torch.cuda.reset_peak_memory_stats()

    for i in range(args.n_samples):
        t0 = time.time()
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * args.cbct_id + i)
        rec = pipeline(
            batch_size=gt.shape[0], num_inference_steps=args.diffusion_num_steps,
            generator=gen,
        ).images.detach()
        dt = time.time() - t0

        if i == 0 and not done(output_dir, dataset, args.cbct_id, args.nviews, tag):
            emit(method=tag, pred=rec, extra={"recon_time_s": dt, "sample_index": 0}, **common)

        rec_c = rec.detach().cpu()
        if mean is None:
            mean = rec_c.clone()
            m2 = torch.zeros_like(rec_c)
        else:
            delta = rec_c - mean
            mean += delta / (i + 1)
            m2 += delta * (rec_c - mean)

        per_sample.append({"sample_index": i, "recon_time_s": dt,
                           "psnr": float(cbct_psnr(gt.cpu(), rec_c))})
        print(f"  sample {i+1}/{args.n_samples}: PSNR {per_sample[-1]['psnr']:.3f} ({dt:.1f}s)",
              flush=True)
        del rec, rec_c; torch.cuda.empty_cache()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    emit(method=mean_method, pred=mean,
         extra={"n_samples": args.n_samples, "per_sample": per_sample,
                "total_recon_time_s": sum(p["recon_time_s"] for p in per_sample),
                "peak_gpu_gb": peak_gb},
         **common)

    if args.n_samples > 1:
        std = (m2 / (args.n_samples - 1)).sqrt()
        _save_npz(
            _volume_path(output_dir, dataset, args.cbct_id, args.nviews, f"std_{tag}"),
            std.numpy().astype(np.float32),
        )
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")


def run_admmtv(args, d, device, output_dir, dataset, conditional: bool):
    """DiffusionMBIR (ADMM + 1D-TV-along-z guidance), optionally FDK-conditioned.
    See ``cbct_diffusion.inference.admm_tv_guidance``. Mirrors ``run_diffusion``'s
    Welford mean/STD pattern when ``args.n_samples > 1``, so mu(DPA+ADMM-TV) and
    mu(CDPA+ADMM-TV) get the same n=5-volume-averaged, LPIPS-scored treatment as
    every other row of Table~\\ref{tab:hr_full_volume} instead of being pinned to
    a single representative test volume.
    """
    tag = "cdpa_admmtv" if conditional else "dpa_admmtv"
    gt = d["gt"]
    common = dict(output_dir=output_dir, dataset=dataset, cbct_id=args.cbct_id,
                  nviews=args.nviews, gt=gt, n_jobs=args.ssim_jobs)

    mean_method = f"mu_{tag}_n{args.n_samples}"
    if done(output_dir, dataset, args.cbct_id, args.nviews, tag) and \
       (args.n_samples == 1 or done(output_dir, dataset, args.cbct_id, args.nviews, mean_method)):
        print(f"  {tag}: already complete, nothing to do")
        return

    model = load_unet(
        args.diffusion_checkpoint, d["fdk"].shape[0], device,
        2 if conditional else 1, with_time=True,
    )

    guidance = ADMMTVGuidance(
        vecs=d["vecs"], mask=d["mask"], vol_shape=d["vol_shape"],
        det_rows=d["projs"].shape[1], det_cols=d["projs"].shape[2],
        voxel_size_mm=d["vsm"], y=d["projs"][d["mask"]].to(device),
        device=device, rho=args.admmtv_rho, lam=args.admmtv_lam,
        n_admm_iters=args.admmtv_admm_iters, n_cg_iters=args.admmtv_cg_iters,
    )
    scheduler = GuidedDDIMScheduler(
        num_train_timesteps=args.scheduler_train_timesteps, guidance_function=guidance,
    )
    pipeline = DDIMPipeline(
        unet=model, scheduler=scheduler,
        fdk_prior=d["fdk"].to(device) if conditional else None,
        normalize_fn=d["normalizer"].normalize, denormalize_fn=d["normalizer"].denormalize,
        slice_batch_size=args.slice_batch_size,
    )

    mean = None
    m2 = None
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.n_samples):
        t0 = time.time()
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * args.cbct_id + i)
        rec = pipeline(
            batch_size=gt.shape[0], num_inference_steps=args.diffusion_num_steps, generator=gen,
        ).images.detach()
        dt = time.time() - t0

        if i == 0 and not done(output_dir, dataset, args.cbct_id, args.nviews, tag):
            emit(method=tag, pred=rec, extra={
                "recon_time_s": dt, "sample_index": 0,
                "admmtv_rho": args.admmtv_rho, "admmtv_lam": args.admmtv_lam,
                "admmtv_admm_iters": args.admmtv_admm_iters, "admmtv_cg_iters": args.admmtv_cg_iters,
            }, **common)

        rec_c = rec.detach().cpu()
        if mean is None:
            mean = rec_c.clone()
            m2 = torch.zeros_like(rec_c)
        else:
            delta = rec_c - mean
            mean += delta / (i + 1)
            m2 += delta * (rec_c - mean)
        print(f"  sample {i+1}/{args.n_samples}: PSNR {float(cbct_psnr(gt.cpu(), rec_c)):.3f} ({dt:.1f}s)",
              flush=True)
        del rec, rec_c; torch.cuda.empty_cache()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    if args.n_samples > 1:
        emit(method=mean_method, pred=mean, extra={
            "n_samples": args.n_samples, "peak_gpu_gb": peak_gb,
            "admmtv_rho": args.admmtv_rho, "admmtv_lam": args.admmtv_lam,
            "admmtv_admm_iters": args.admmtv_admm_iters, "admmtv_cg_iters": args.admmtv_cg_iters,
        }, **common)
        std = (m2 / (args.n_samples - 1)).sqrt()
        _save_npz(
            _volume_path(output_dir, dataset, args.cbct_id, args.nviews, f"std_{tag}"),
            std.numpy().astype(np.float32),
        )
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")


def run_save_gt(args, d, output_dir, dataset):
    """Persist the ground-truth volume alongside every method's reconstruction.

    ``load_volume``/``load_volume_high_res`` already return ``gt`` as a side
    effect of loading any method group, but nothing previously saved it to
    disk -- every downstream consumer (LPIPS scoring, this function's own
    caller) re-derives it from the dataset loader instead. This is cheap for
    the 501^3 walnut case specifically: the ground truth is a pre-computed
    ``full_AGD_50_*`` reconstruction read straight off disk (see
    ``cbct_diffusion.data.walnut512.build_external_volume``), not something
    requiring the ASTRA/GPU forward model -- only the classical/diffusion
    method groups pay that cost, as a side effect of also building the FDK
    prior.
    """
    gt_path = _volume_path(output_dir, dataset, args.cbct_id, args.nviews, "gt")
    if gt_path.is_file():
        print("  gt: already saved, nothing to do")
        return
    _save_npz(gt_path, d["gt"].detach().cpu().numpy().astype(np.float32))
    print(f"  gt: saved to {gt_path}")


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Evaluate one CBCT volume with one method group")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--cbct_id", type=int, nargs="+", required=True,
                   help="one or more test-split indices; processed sequentially in "
                        "one process so cheap method groups can share a pod")
    p.add_argument("--nviews", type=int, default=20)
    p.add_argument("--method_group",
                    choices=["classical", "dpa", "cdpa", "dpa_admmtv", "cdpa_admmtv", "gt"],
                    required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--high_resolution", action="store_true",
                   help="501^3 raw-TIFF walnut path (Walnut512) instead of the "
                        "256^3 HuggingFace-format SliceCBCTDataset path")
    p.add_argument("--dataset_label", type=str, default=None,
                   help="override the 'dataset' string used in result/volume "
                        "filenames (default: basename of --data_path). Use this "
                        "for --high_resolution so files don't collide with a "
                        "256^3 sweep over a directory of the same basename.")

    p.add_argument("--unet_checkpoint", type=str, default=None)
    p.add_argument("--diffusion_checkpoint", type=str, default=None)

    p.add_argument("--n_samples", type=int, default=20,
                   help="posterior samples averaged for mu(.) — the paper's Table I caption says 20")
    p.add_argument("--diffusion_num_steps", type=int, default=50)
    p.add_argument("--scheduler_train_timesteps", type=int, default=1000)
    p.add_argument("--slice_batch_size", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)

    # Guidance: the published settings differ between conditional and
    # unconditional (5 epochs @ 5e-4 vs 20 @ 2e-3), so they are passed in by
    # the orchestrator rather than defaulted here.
    p.add_argument("--guidance_max_epochs", type=int, nargs="+", default=[5])
    p.add_argument("--guidance_batch_size", type=int, default=30)
    p.add_argument("--guidance_lr", type=float, nargs="+", default=[5e-4])
    p.add_argument("--guidance_clamp_min", type=float, default=0.0)

    # ADMM-TV (DiffusionMBIR) hyperparameters, only used by --method_group
    # {dpa,cdpa}_admmtv. Defaults are the ones tuned for walnut (see
    # scripts/probe_diffusionmbir.py's lambda/iteration sweep); not retuned at
    # 501^3, matching the main text's stated methodology.
    p.add_argument("--admmtv_rho", type=float, default=1.0)
    p.add_argument("--admmtv_lam", type=float, default=0.005)
    p.add_argument("--admmtv_admm_iters", type=int, default=3)
    p.add_argument("--admmtv_cg_iters", type=int, default=5)

    # GD baseline schedule: the GD_zero / GD_FDK rows of Table I were produced
    # by the *diffusion* script's schedule ([100,50,10] @ [5e-3,5e-4,1e-4]),
    # not the weaker one in reconstruct_unet.py -- verified against the logged
    # walnut values (23.93 / 23.36 dB).
    p.add_argument("--gd_epochs", type=int, nargs="+", default=[100, 50, 10])
    p.add_argument("--gd_lr", type=float, nargs="+", default=[5e-3, 5e-4, 1e-4])
    p.add_argument("--gd_batch_size", type=int, default=30)
    # FT schedule for the FDK-denoiser: the published runs used [10] @ [2e-4]
    # for walnut but [5] @ [5e-4] for dental/spine (see the per-dataset launch
    # scripts in chip-project/scripts/cbct/). The orchestrator sets these per
    # dataset; the defaults below are the walnut ones.
    p.add_argument("--ft_epochs", type=int, nargs="+", default=[10])
    p.add_argument("--ft_lr", type=float, nargs="+", default=[2e-4])
    p.add_argument("--ft_batch_size", type=int, default=30)
    p.add_argument("--unet_batch_size", type=int, default=10)

    p.add_argument("--save_gd_volumes", action="store_true",
                   help="also persist the GD baseline volumes (deterministic and "
                        "cheap to recompute, so skipped by default)")
    p.add_argument("--ssim_jobs", type=int, default=-1,
                   help="joblib workers for SSIM; keep <= the job's CPU limit")
    p.add_argument("--wandb_project", type=str, default="cbct-256-suite")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()

    dataset = args.dataset_label or Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(dataset))
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    ids = list(args.cbct_id)
    print(f"=== {dataset} ids={ids} nviews={args.nviews} "
          f"group={args.method_group} device={device}")
    print(f"    clamp={get_dataset_clamp()}  out={output_dir}")

    if args.method_group == "classical" and not args.unet_checkpoint:
        raise SystemExit("--unet_checkpoint is required for --method_group classical")
    if args.method_group not in ("classical", "gt") and not args.diffusion_checkpoint:
        raise SystemExit(f"--diffusion_checkpoint is required for --method_group {args.method_group}")

    run = None
    if not args.no_wandb:
        import wandb
        span = f"id{ids[0]}" if len(ids) == 1 else f"id{ids[0]}-{ids[-1]}"
        run = wandb.init(
            project=args.wandb_project,
            name=f"{dataset}_{span}_n{args.nviews}_{args.method_group}",
            config=vars(args), reinit=True,
        )

    t_all = time.time()
    failures = []

    for cid in ids:
        # run_classical / run_diffusion read args.cbct_id, so point it at the
        # volume currently being processed.
        args.cbct_id = cid
        t0 = time.time()
        try:
            d = load_volume(args, device)
            print(f"--- id={cid}: loaded gt{tuple(d['gt'].shape)} in {time.time()-t0:.1f}s")

            if args.method_group == "classical":
                run_classical(args, d, device, output_dir, dataset)
            elif args.method_group in ("dpa", "cdpa"):
                run_diffusion(args, d, device, output_dir, dataset,
                              conditional=(args.method_group == "cdpa"))
            elif args.method_group == "gt":
                run_save_gt(args, d, output_dir, dataset)
            else:
                run_admmtv(args, d, device, output_dir, dataset,
                           conditional=(args.method_group == "cdpa_admmtv"))
            print(f"--- id={cid}: done in {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:
            # One bad volume must not discard the volumes already finished in
            # this pod -- their result JSONs are already on disk, and a relaunch
            # will skip them.
            import traceback
            traceback.print_exc()
            failures.append((cid, repr(exc)))
            print(f"--- id={cid}: FAILED ({exc!r}) — continuing", flush=True)
        finally:
            d = None
            torch.cuda.empty_cache()

    if run is not None:
        import wandb
        summary = {}
        mdir = output_dir / "metrics"
        if mdir.is_dir():
            for cid in ids:
                for f in sorted(mdir.glob(f"{dataset}_n{args.nviews}_id{cid}_*.json")):
                    m = json.loads(f.read_text())
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            summary[f"id{cid}/{m['method']}/{k}"] = v
        wandb.log(summary)
        wandb.finish(exit_code=1 if failures else 0)

    print(f"=== all {len(ids)} volume(s) in {time.time()-t_all:.1f}s")
    if failures:
        print(f"=== {len(failures)} FAILED: {failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
