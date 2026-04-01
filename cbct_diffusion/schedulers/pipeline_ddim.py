# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DDIMPipeline: slice-wise diffusion for 3D CBCT volumes.

This pipeline applies a 2D UNet denoiser slice-by-slice across a 3D volume,
using the :class:`GuidedDDIMScheduler` to inject sinogram-consistency guidance
at every reverse-diffusion step.

The optional ``fdk_prior`` is concatenated as a second input channel (no noise
added to it). The ``normalize_fn`` / ``denormalize_fn`` callbacks allow the
guidance function to operate in the original data range while the UNet works
in normalised space.
"""

from typing import List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from cbct_diffusion.schedulers.scheduling_ddim import GuidedDDIMScheduler

try:
    from diffusers.utils.import_utils import is_torch_xla_available
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
        XLA_AVAILABLE = True
    else:
        XLA_AVAILABLE = False
except Exception:
    XLA_AVAILABLE = False


class DDIMPipeline(DiffusionPipeline):
    """Slice-wise DDIM diffusion pipeline for 3D CBCT reconstruction.

    Parameters
    ----------
    unet : nn.Module
        Slice-wise 2D UNet denoiser.
    scheduler : GuidedDDIMScheduler
        DDIM scheduler (optionally with guidance function).
    fdk_prior : torch.Tensor, optional
        FDK volume ``(D, H, W)`` used as conditioning channel.
    normalize_fn : callable, optional
        Data → normalised space.
    denormalize_fn : callable, optional
        Normalised → data space.
    slice_batch_size : int
        Number of slices processed per UNet forward pass.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet,
        scheduler: GuidedDDIMScheduler,
        fdk_prior: Optional[torch.Tensor] = None,
        normalize_fn: Optional[callable] = lambda x: x,
        denormalize_fn: Optional[callable] = lambda x: x,
        slice_batch_size: int = 2,
    ):
        super().__init__()
        scheduler = GuidedDDIMScheduler.from_config(scheduler.config)

        if fdk_prior is not None:
            if fdk_prior.dim() != 3:
                raise ValueError("fdk_prior must be (D, H, W)")
            fdk_prior = normalize_fn(fdk_prior)
        self.fdk_prior = fdk_prior
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn
        self.slice_batch_size = slice_batch_size

        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """Run the reverse diffusion loop.

        Parameters
        ----------
        batch_size : int
            Ignored when ``fdk_prior`` is provided (volume depth is used).
        num_inference_steps : int
            Number of DDIM denoising steps.
        eta : float
            Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM).

        Returns
        -------
        ImagePipelineOutput or tuple
            Contains the reconstructed volume ``(D, H, W)``.
        """
        if self.fdk_prior is not None:
            image_shape = self.fdk_prior.shape
        elif isinstance(self.unet.config.sample_size, int):
            image_shape = (batch_size, self.unet.config.sample_size, self.unet.config.sample_size)
        else:
            image_shape = (batch_size, *self.unet.config.sample_size)

        D, H, W = image_shape[0], image_shape[1], image_shape[2]
        device = self._execution_device
        fdk_prior = self.fdk_prior

        image = randn_tensor(image_shape, generator=generator, device=device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            noisy_slices = image

            if fdk_prior is not None:
                model_input = torch.stack([noisy_slices, fdk_prior.to(device)], dim=1)
            else:
                model_input = noisy_slices.unsqueeze(1)

            slice_idx = torch.arange(D, device=device)
            with torch.no_grad():
                noise_pred_slices = torch.empty((D, 1, H, W), device=device, dtype=model_input.dtype)
                for start in range(0, D, self.slice_batch_size):
                    end = min(start + self.slice_batch_size, D)
                    chunk = model_input[start:end]
                    chunk_idx = slice_idx[start:end]

                    if hasattr(self.unet.config, "num_class_embeds") and self.unet.config.num_class_embeds is not None:
                        pred = self.unet(chunk, t, class_labels=chunk_idx, return_dict=False)[0]
                    else:
                        pred = self.unet(chunk, t, return_dict=False)[0]
                    noise_pred_slices[start:end] = pred

            model_output = noise_pred_slices.permute(1, 0, 2, 3)[0].contiguous()

            image = self.scheduler.step(
                model_output, t, image, eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                normalize_fn=self.normalize_fn,
                denormalize_fn=self.denormalize_fn,
            ).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        image = self.denormalize_fn(image)
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
