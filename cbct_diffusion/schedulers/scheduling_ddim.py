# Copyright 2025 Stanford University Team and The HuggingFace Team. All rights reserved.
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

"""GuidedDDIMScheduler: DDIM with external guidance injection.

Extends the standard DDIM scheduler to apply an external guidance function
(e.g., sinogram-consistency gradient descent) to the predicted clean sample
at each denoising step.

The guidance is injected at step 4.5 of the DDIM algorithm, between the
prediction of x_0 and the computation of the variance / direction terms.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.schedulers.scheduling_ddim import (
    BaseOutput,
    ConfigMixin,
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    randn_tensor,
    register_to_config,
)


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """Output of :meth:`GuidedDDIMScheduler.step`.

    Attributes
    ----------
    prev_sample : torch.Tensor
        Denoised sample x_{t-1}.
    pred_original_sample : torch.Tensor, optional
        Predicted clean sample x_0.
    """
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    """Create a beta schedule from an alpha-bar transform function."""
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    """Rescale betas so the final timestep has zero SNR."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    a0 = alphas_bar_sqrt[0].clone()
    aT = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt = (alphas_bar_sqrt - aT) * a0 / (a0 - aT)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    return 1 - alphas


class GuidedDDIMScheduler(SchedulerMixin, ConfigMixin):
    """DDIM scheduler with external guidance injection.

    At each reverse step, after predicting the clean sample x_0, an optional
    ``guidance_function(x_0, timestep)`` is applied before computing the
    direction and noise terms. This enables sinogram-consistency enforcement.

    Parameters
    ----------
    num_train_timesteps : int
        Training diffusion steps.
    guidance_function : callable, optional
        ``f(x_0, t) -> x_0_guided``. Applied to the predicted clean image.
    beta_start, beta_end : float
        Beta schedule endpoints.
    beta_schedule : str
        ``"linear"``, ``"scaled_linear"``, or ``"squaredcos_cap_v2"``.
    clip_sample : bool
        Clip predicted x_0 to ``[-clip_sample_range, clip_sample_range]``.
    prediction_type : str
        ``"epsilon"``, ``"sample"``, or ``"v_prediction"``.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        guidance_function: callable = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} not implemented")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.guidance_function = guidance_function
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining))
        s = torch.quantile(sample.abs(), self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value).unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining).to(dtype)
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) > num_train_timesteps ({self.config.num_train_timesteps})"
            )
        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.config.timestep_spacing}")

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        normalize_fn: callable = None,
        denormalize_fn: callable = None,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """Reverse one DDIM step with optional guidance.

        Parameters
        ----------
        normalize_fn, denormalize_fn : callable, optional
            Applied around the guidance function so it operates in data space.
        """
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps() before step()")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.config.prediction_type}")

        # Clip / threshold
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range,
            )

        # ---- Guidance injection ----
        if self.guidance_function is not None:
            _denorm = denormalize_fn if denormalize_fn is not None else (lambda x: x)
            _norm = normalize_fn if normalize_fn is not None else (lambda x: x)
            pred_original_sample = _norm(self.guidance_function(_denorm(pred_original_sample), timestep))

        # Variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator,
                    device=model_output.device, dtype=model_output.dtype,
                )
            prev_sample = prev_sample + std_dev_t * variance_noise

        if not return_dict:
            return prev_sample, pred_original_sample
        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(self, original_samples, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha = sqrt_alpha.flatten()
        while len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)

        sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus = sqrt_one_minus.flatten()
        while len(sqrt_one_minus.shape) < len(original_samples.shape):
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def get_velocity(self, sample, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)
        sa = alphas_cumprod[timesteps] ** 0.5
        sa = sa.flatten()
        while len(sa.shape) < len(sample.shape):
            sa = sa.unsqueeze(-1)
        som = (1 - alphas_cumprod[timesteps]) ** 0.5
        som = som.flatten()
        while len(som.shape) < len(sample.shape):
            som = som.unsqueeze(-1)
        return sa * noise - som * sample

    def __len__(self):
        return self.config.num_train_timesteps
