"""Tomographic CBCT Diffusion model for single 3D volume reconstruction.

This module implements a diffusion-based prior for CBCT reconstruction.
The diffusion model uses a 2D UNet applied slice-wise across the 3D volume,
with optional FDK conditioning and sinogram-consistency guidance via
gradient-descent reconstruction.

Shapes
------
- Working volume: ``(D, H, W)``
- Slice stack for UNet: ``(D, C, H, W)`` where ``C=1`` (or ``C=2`` with FDK conditioning)

Conditioning
------------
An optional ``fdk_prior`` of shape ``(D, H, W)`` is concatenated as a second
channel per slice (producing ``(D, 2, H, W)`` model input). The prior is NOT noised.

Usage
-----
The main entry point is :meth:`TomographicCBCTDiffusion.guided_diffusion_pipeline`,
which alternates diffusion denoising steps with sinogram-consistency guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from cbct_diffusion.models.latent_unet2d import LatentUnet2D

try:
    from astra_torch.cbct import gd_reconstruction_masked
except Exception:
    gd_reconstruction_masked = None


@dataclass
class GuidanceConfig:
    """Parameters for sinogram-consistency gradient-descent guidance.

    Attributes
    ----------
    voxel_per_mm : int
        Voxel density (voxels per millimetre).
    voxel_size_mm : float
        Physical size of each voxel in millimetres.
    max_epochs : int or list[int]
        Number of GD epochs (or list for multi-stage schedules).
    batch_size : int
        Number of projection views per inner GD batch.
    lr : float or list[float]
        Learning rate(s) for the GD optimiser.
    clamp_min : float
        Lower bound for clamping the reconstruction.
    optimizer_type : str
        ``"adam"`` or ``"sgd"``.
    momentum : float
        Momentum for SGD.
    weight_decay : float
        Weight decay for the optimiser.
    verbose : bool
        Whether to print per-epoch progress.
    """
    voxel_per_mm: int = 10
    voxel_size_mm: float = 0.1957
    max_epochs: int | Iterable[int] = 15
    batch_size: int = 12
    lr: float | Iterable[float] = 1e-3
    clamp_min: float = 0.0
    optimizer_type: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    verbose: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for ``gd_reconstruction_masked``."""
        return {
            "voxel_per_mm": self.voxel_per_mm,
            "voxel_size_mm": self.voxel_size_mm,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "clamp_min": self.clamp_min,
            "optimizer_type": self.optimizer_type,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "verbose": self.verbose,
        }


class TomographicCBCTDiffusion(nn.Module):
    """Single-volume CBCT diffusion with slice-wise UNet prior.

    Parameters
    ----------
    volume_shape : tuple of int
        ``(D, H, W)`` shape of the reconstruction volume.
    unet : LatentUnet2D
        Pre-trained 2D diffusion UNet.
    guidance_config : GuidanceConfig or dict, optional
        Configuration for sinogram-consistency guidance steps.
    slice_batch_size : int
        Number of slices processed simultaneously through the UNet.
    dataset_normalizer : object, optional
        Object providing ``normalize`` / ``denormalize`` methods. If ``None``,
        identity normalisation is used.
    """

    def __init__(
        self,
        volume_shape,
        unet: LatentUnet2D,
        guidance_config: GuidanceConfig | Dict[str, Any] | None = None,
        slice_batch_size: int = 10,
        dataset_normalizer=None,
    ):
        super().__init__()
        assert len(volume_shape) == 3, "volume_shape must be (D, H, W)"
        self.unet = unet
        for p in self.unet.parameters():
            p.requires_grad = False
        self.volume_shape = volume_shape

        if guidance_config is None:
            self.guidance_config = GuidanceConfig()
        elif isinstance(guidance_config, GuidanceConfig):
            self.guidance_config = guidance_config
        else:
            self.guidance_config = GuidanceConfig(**guidance_config)
        self.slice_batch_size = slice_batch_size

        # Normaliser with identity fallback
        if dataset_normalizer is None:
            class _Identity:
                @staticmethod
                def normalize(x):
                    return x
                @staticmethod
                def denormalize(x):
                    return x
            self.dataset_class = _Identity()
        else:
            self.dataset_class = dataset_normalizer

    # ------------------------------------------------------------------
    # Diffusion step: noise → UNet prediction → scheduler step
    # ------------------------------------------------------------------
    def diffusion_step(
        self,
        device: torch.device,
        t: int,
        noise_scheduler,
        x_0_pred: torch.Tensor,
        fdk_prior: Optional[torch.Tensor] = None,
    ):
        """Run one reverse-diffusion step.

        Parameters
        ----------
        device : torch.device
            Computation device.
        t : int
            Current diffusion timestep.
        noise_scheduler
            Diffusion noise scheduler (e.g., ``DDPMScheduler``).
        x_0_pred : torch.Tensor
            Current clean estimate ``(D, H, W)``.
        fdk_prior : torch.Tensor, optional
            FDK volume ``(D, H, W)`` for conditioning.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(noise_pred, x_0_clean)`` both of shape ``(D, H, W)``.
        """
        timesteps = torch.LongTensor([t]).to(device)
        x_0_pred = x_0_pred.unsqueeze(0).unsqueeze(0) if x_0_pred.dim() == 3 else x_0_pred

        x_0_pred = self.dataset_class.normalize(x_0_pred)

        x_t = noise_scheduler.add_noise(
            x_0_pred, torch.randn_like(x_0_pred), torch.LongTensor([t]).to(device)
        )

        _, _, D, H, W = x_t.shape
        noisy_slices = x_t[0, 0]

        if fdk_prior is not None:
            if fdk_prior.dim() != 3:
                raise ValueError("fdk_prior must be (D, H, W)")
            fdk_prior = self.dataset_class.normalize(fdk_prior)
            model_input = torch.stack([noisy_slices, fdk_prior.to(device)], dim=1)
        else:
            model_input = noisy_slices.unsqueeze(1)

        slice_idx = torch.arange(D, device=device)
        with torch.no_grad():
            noise_pred_slices = torch.empty((D, 1, H, W), device=device, dtype=model_input.dtype)
            for start in range(0, D, self.slice_batch_size):
                end = min(start + self.slice_batch_size, D)
                pred_chunk = self.unet(
                    model_input[start:end], timesteps,
                    class_labels=slice_idx[start:end], return_dict=False,
                )[0]
                noise_pred_slices[start:end] = pred_chunk

        noise_pred = noise_pred_slices.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        step_out = noise_scheduler.step(noise_pred, timesteps.item(), x_t)
        x_0_clean = step_out.pred_original_sample

        x_0_clean = self.dataset_class.denormalize(x_0_clean)
        if fdk_prior is not None:
            fdk_prior = self.dataset_class.denormalize(fdk_prior)

        return noise_pred.squeeze(0).squeeze(0), x_0_clean.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # Sinogram guidance via GD reconstruction
    # ------------------------------------------------------------------
    def sinogram_guidance(
        self,
        projs_vrc: torch.Tensor,
        vecs,
        vol_init: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        """Apply sinogram-consistency gradient-descent guidance.

        Parameters
        ----------
        projs_vrc : torch.Tensor
            Projection data ``(V, R, C)``.
        vecs : array-like
            Geometry vectors ``(V, 12)``.
        vol_init : torch.Tensor
            Current volume estimate ``(D, H, W)``.
        mask : array-like, optional
            Boolean mask selecting active views.

        Returns
        -------
        torch.Tensor
            Guided reconstruction ``(D, H, W)``.
        """
        if gd_reconstruction_masked is None:
            return vol_init

        cfg_kwargs = self.guidance_config.to_kwargs()
        recon = gd_reconstruction_masked(
            projs_vrc=projs_vrc, vecs=vecs, mask=mask,
            vol_init=vol_init, **cfg_kwargs,
        )
        return recon

    # ------------------------------------------------------------------
    # Full guided diffusion pipeline
    # ------------------------------------------------------------------
    def guided_diffusion_pipeline(
        self,
        device: torch.device,
        x_start: Optional[torch.Tensor],
        t_start: int,
        t_end: int,
        noise_scheduler,
        projs_vrc: torch.Tensor,
        vecs,
        num_steps: int = 25,
        buffer_size: int = 1,
        buffer_guidance: bool = False,
        mask=None,
        verbose: bool = False,
        fdk_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full guided diffusion reconstruction loop.

        Parameters
        ----------
        device : torch.device
            Computation device.
        x_start : torch.Tensor or None
            Starting volume ``(D, H, W)``. ``None`` initialises to zeros.
        t_start, t_end : int
            Start and end diffusion timesteps.
        noise_scheduler
            Noise scheduler instance.
        projs_vrc : torch.Tensor
            Projection data ``(V, R, C)``.
        vecs : array-like
            Geometry vectors.
        num_steps : int
            Number of diffusion steps.
        buffer_size : int
            Extra fine-grained steps at the end.
        buffer_guidance : bool
            Whether to apply guidance during buffer steps.
        mask : array-like, optional
            View selection mask.
        verbose : bool
            Show progress bar.
        fdk_conditioning : torch.Tensor, optional
            FDK prior for conditioning.

        Returns
        -------
        torch.Tensor
            Final reconstructed volume ``(D, H, W)``.
        """
        if x_start is None:
            x_start = torch.zeros(self.volume_shape, device=device)

        x_0_pred = x_start
        timesteps = torch.linspace(t_start, t_end + buffer_size + 1, num_steps, dtype=torch.long)
        timesteps = torch.cat([timesteps, torch.arange(buffer_size, 0, -1)])
        progress = tqdm(range(len(timesteps)), disable=not verbose)

        for i in progress:
            t = timesteps[i].item()

            with torch.no_grad():
                _, x_0_pred = self.diffusion_step(
                    device, t, noise_scheduler, x_0_pred, fdk_prior=fdk_conditioning,
                )
            torch.cuda.empty_cache()

            if t >= buffer_size or buffer_guidance:
                x_0_pred = self.sinogram_guidance(
                    projs_vrc=projs_vrc, vecs=vecs, vol_init=x_0_pred, mask=mask,
                )
                torch.cuda.empty_cache()

            if verbose:
                progress.set_postfix({"t": t})

        return x_0_pred


__all__ = ["TomographicCBCTDiffusion", "GuidanceConfig"]
