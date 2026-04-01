"""Metrics for CBCT reconstruction evaluation.

Provides GPU-friendly PSNR and SSIM functions with consistent clamping and
normalisation across different CBCT datasets (dental, spine, walnut).

Key public API
--------------
- ``cbct_psnr``           – volume-level PSNR with dataset-specific clamping.
- ``cbct_ssim``           – sliding-window 2D/3D SSIM (matches ``scikit-image``).
- ``cbct_ssim_3d_full``   – full 3D SSIM (depth + height + width planes).
- ``cbct_ssim_3d_gaal``   – parallelised 3D SSIM via ``joblib`` (recommended).
- ``data_norm``           – min-max normalise a tensor to [0, 1].
- Clamp presets and helpers for per-dataset value ranges.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from joblib import Parallel, delayed, parallel_backend

# ---------------------------------------------------------------------------
# Clamp presets
# ---------------------------------------------------------------------------
DENTAL_CLAMP: Dict[str, float] = {"clamp_min": 0.0, "clamp_max": 0.09009}
SPINE_CLAMP: Dict[str, float] = {"clamp_min": 0.0, "clamp_max": 0.051744}
WALNUT_CLAMP: Dict[str, float] = {"clamp_min": 0.0, "clamp_max": 0.084}
ZERO_ONE_CLAMP: Dict[str, float] = {"clamp_min": 0.0, "clamp_max": 1.0}

_DATASET_CLAMP: Dict[str, float] = ZERO_ONE_CLAMP.copy()


def set_dataset_clamp(clamp_dict: Dict[str, float]) -> None:
    """Set the global dataset clamp values."""
    global _DATASET_CLAMP
    _DATASET_CLAMP = clamp_dict.copy()


def get_dataset_clamp() -> Dict[str, float]:
    """Return the current global dataset clamp."""
    return _DATASET_CLAMP.copy()


def get_clamp_by_name(name: str) -> Dict[str, float]:
    """Infer clamp values from a dataset name string."""
    n = name.lower()
    if "dental" in n:
        return DENTAL_CLAMP
    if "spine" in n:
        return SPINE_CLAMP
    if "walnut" in n:
        return WALNUT_CLAMP
    return ZERO_ONE_CLAMP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def data_norm(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalise *x* to [0, 1]."""
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)


def apply_circle_mask(x: torch.Tensor) -> torch.Tensor:
    """Zero out pixels outside the largest inscribed circle."""
    *batch, H, W = x.shape
    cy, cx = H / 2.0, W / 2.0
    r = min(cy, cx)
    yy = torch.arange(H, device=x.device).float() - cy + 0.5
    xx = torch.arange(W, device=x.device).float() - cx + 0.5
    dist = (yy[:, None] ** 2 + xx[None, :] ** 2).sqrt()
    mask = (dist <= r).float()
    return x * mask


def create_circle_filter(H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a boolean circle mask of shape ``(H, W)``."""
    cy, cx = H / 2.0, W / 2.0
    r = min(cy, cx)
    yy = torch.arange(H, device=device).float() - cy + 0.5
    xx = torch.arange(W, device=device).float() - cx + 0.5
    return (yy[:, None] ** 2 + xx[None, :] ** 2).sqrt() <= r


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def cbct_psnr(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range: Optional[float] = None,
    clamp_values: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Volume PSNR with dataset-specific clamping and normalisation.

    Both inputs are clamped, normalised to [0, 1], then PSNR is computed on
    the full volume.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    gt = gt.clamp(clamp["clamp_min"], clamp["clamp_max"])
    pred = pred.clamp(clamp["clamp_min"], clamp["clamp_max"])
    gt = data_norm(gt).to(device)
    pred = data_norm(pred).to(device)
    if data_range is None:
        data_range = gt.max() - gt.min()
    mse = torch.mean((gt - pred) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=gt.device)
    return 20 * torch.log10(data_range / torch.sqrt(mse))


# ---------------------------------------------------------------------------
# 2D SSIM helper (sliding-window, matches scikit-image)
# ---------------------------------------------------------------------------
def _ssim_2d(
    gt: torch.Tensor,
    pred: torch.Tensor,
    win_size: int,
    c1: float,
    c2: float,
    use_sample_covariance: bool = True,
) -> torch.Tensor:
    """Sliding-window 2D SSIM on a single ``(H, W)`` pair."""
    H, W = gt.shape
    pad = win_size // 2
    gt_p = F.pad(gt.unsqueeze(0).unsqueeze(0), (pad,) * 4, mode="reflect")
    pred_p = F.pad(pred.unsqueeze(0).unsqueeze(0), (pad,) * 4, mode="reflect")
    gt_w = gt_p.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(H, W, -1)
    pred_w = pred_p.unfold(2, win_size, 1).unfold(3, win_size, 1).contiguous().view(H, W, -1)
    N = win_size * win_size
    mu_gt = gt_w.mean(dim=2)
    mu_pred = pred_w.mean(dim=2)
    gt_c = gt_w - mu_gt.unsqueeze(2)
    pred_c = pred_w - mu_pred.unsqueeze(2)
    denom = (N - 1) if (use_sample_covariance and N > 1) else N
    sig_gt = (gt_c ** 2).sum(dim=2) / denom
    sig_pred = (pred_c ** 2).sum(dim=2) / denom
    sig_cross = (gt_c * pred_c).sum(dim=2) / denom
    num = (2 * mu_gt * mu_pred + c1) * (2 * sig_cross + c2)
    den = (mu_gt ** 2 + mu_pred ** 2 + c1) * (sig_gt + sig_pred + c2)
    return (num / (den + 1e-10)).mean()


# ---------------------------------------------------------------------------
# 2D / 3D SSIM (GPU, sliding-window)
# ---------------------------------------------------------------------------
def cbct_ssim(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    clamp_values: Optional[Dict[str, float]] = None,
    win_size: int = 7,
    use_sample_covariance: bool = True,
) -> torch.Tensor:
    """Sliding-window SSIM (matches ``skimage.metrics.structural_similarity``).

    Supports 2D ``(H, W)`` and 3D ``(D, H, W)`` inputs (slice-averaged for 3D).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    gt = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).to(device)
    pred = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).to(device)
    if data_range is None:
        data_range = gt.max() - gt.min()
    if win_size % 2 == 0:
        win_size += 1
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    if gt.dim() == 2:
        return _ssim_2d(gt, pred, win_size, c1, c2, use_sample_covariance)
    if gt.dim() == 3:
        vals = [_ssim_2d(gt[i], pred[i], win_size, c1, c2, use_sample_covariance) for i in range(gt.shape[0])]
        return torch.stack(vals).mean()
    raise ValueError(f"SSIM supports 2D/3D tensors, got {gt.dim()}D")


# ---------------------------------------------------------------------------
# Full 3D SSIM (GPU, three orthogonal planes)
# ---------------------------------------------------------------------------
def cbct_ssim_3d_full(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    clamp_values: Optional[Dict[str, float]] = None,
    win_size: int = 7,
    use_sample_covariance: bool = True,
) -> torch.Tensor:
    """True 3D SSIM: average SSIM across depth, height and width planes.

    Inputs are clamped and normalised before computation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    gt = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).to(device)
    pred = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).to(device)
    if data_range is None:
        data_range = gt.max() - gt.min()
    if win_size % 2 == 0:
        win_size += 1
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    if gt.dim() == 3:
        D, H, W = gt.shape
        ssim_d = torch.stack([_ssim_2d(gt[d], pred[d], win_size, c1, c2, use_sample_covariance) for d in range(D)]).mean()
        ssim_h = torch.stack([_ssim_2d(gt[:, h, :], pred[:, h, :], win_size, c1, c2, use_sample_covariance) for h in range(H)]).mean()
        ssim_w = torch.stack([_ssim_2d(gt[:, :, w], pred[:, :, w], win_size, c1, c2, use_sample_covariance) for w in range(W)]).mean()
        return (ssim_d + ssim_h + ssim_w) / 3.0
    raise ValueError("cbct_ssim_3d_full requires 3D (D,H,W) tensors")


# ---------------------------------------------------------------------------
# Parallelised 3D SSIM via joblib (recommended for large volumes)
# ---------------------------------------------------------------------------
def _compute_ssim_slice(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """SSIM for a single 2D slice pair via ``skimage``."""
    return structural_similarity(a, b, data_range=data_range)


def cbct_ssim_3d_gaal(
    arr1: torch.Tensor | np.ndarray,
    arr2: torch.Tensor | np.ndarray,
    size_average: bool = True,
    data_range: float = 1.0,
    n_jobs: int = -1,
    backend: str = "loky",
) -> float | np.ndarray:
    """Parallelised 3D SSIM (depth + height + width planes via ``joblib``).

    Inputs are clamped using the global dataset clamp and normalised to [0, 1]
    before SSIM computation.
    """
    clamp = get_dataset_clamp()
    if torch.is_tensor(arr1):
        arr1 = arr1.clamp(clamp["clamp_min"], clamp["clamp_max"])
        arr1 = data_norm(arr1).cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.clamp(clamp["clamp_min"], clamp["clamp_max"])
        arr2 = data_norm(arr2).cpu().detach().numpy()

    assert arr1.ndim == 3 and arr2.ndim == 3
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    D, H, W = arr1.shape

    with parallel_backend(backend, n_jobs=n_jobs):
        ssim_d = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_compute_ssim_slice)(arr1[d], arr2[d], data_range) for d in range(D)),
            dtype=np.float64,
        )
        ssim_h = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_compute_ssim_slice)(arr1[:, h, :], arr2[:, h, :], data_range) for h in range(H)),
            dtype=np.float64,
        )
        ssim_w = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_compute_ssim_slice)(arr1[:, :, w], arr2[:, :, w], data_range) for w in range(W)),
            dtype=np.float64,
        )

    ssim_avg = (ssim_d.mean() + ssim_h.mean() + ssim_w.mean()) / 3.0
    if size_average:
        return ssim_avg
    return np.array([ssim_avg])
