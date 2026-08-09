"""Metrics for CBCT reconstruction evaluation.

Provides GPU-friendly PSNR and SSIM functions with consistent clamping and
normalisation across different CBCT datasets (dental, spine, walnut).

Key public API
--------------
- ``cbct_psnr``           – volume-level PSNR with dataset-specific clamping.
- ``cbct_ssim``           – sliding-window 2D/3D SSIM (matches ``scikit-image``).
- ``cbct_ssim_3d_full``   – full 3D SSIM (depth + height + width planes).
- ``cbct_ssim_3d_gaal``   – parallelised 3D SSIM via ``joblib`` (recommended).
- ``cbct_psnr_per_axis``  – PSNR of each 2D slice family, averaged per axis.
- ``inter_slice_consistency`` – normalised adjacent-slice total variation.
- ``data_norm``           – min-max normalise a tensor to [0, 1].
- Clamp presets and helpers for per-dataset value ranges.

Axis convention
---------------
All volumes are indexed ``(D, H, W)`` where **axis 0 is the axial axis** — the
plane in which the 2D diffusion prior and the FDK/FBP denoiser operate. This
follows the ASTRA volume layout ``(Z, Y, X)`` with the cone-beam rotation axis
along Z, and matches ``SliceCBCTDataset``'s ``axis_map={"axial": 0, ...}`` and
``DDIMPipeline``'s use of dim 0 as the slice dimension.

Axes 1 and 2 are therefore the two *off-axis* (coronal / sagittal) directions,
in which a slice-by-slice 2D method has no explicit coherence mechanism. The
axial-vs-off-axis gap in SSIM is a direct measure of inter-slice inconsistency,
which the plain 3-axis average in ``cbct_ssim_3d_gaal`` averages away.
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
    return_per_axis: bool = False,
) -> float | np.ndarray | Dict[str, float]:
    """Parallelised 3D SSIM (depth + height + width planes via ``joblib``).

    Inputs are clamped using the global dataset clamp and normalised to [0, 1]
    before SSIM computation.

    Parameters
    ----------
    return_per_axis : bool
        If ``True``, return a dict with the per-axis means kept separate
        instead of collapsing them into the 3-axis average::

            {"ssim": <3-axis mean>,        # identical to the default return
             "ssim_axial": <axis 0 mean>,  # the plane the 2D model works in
             "ssim_coronal": <axis 1 mean>,
             "ssim_sagittal": <axis 2 mean>,
             "ssim_offaxis": <mean of axes 1 and 2>,
             "ssim_axial_gap": <axial minus off-axis>}

        ``ssim_axial_gap`` is the inter-slice inconsistency signal: a 2D
        slice-wise method scores well in the axial plane and pays for it in the
        other two, so a large positive gap means poor 3D coherence. See the
        module docstring for the axis convention.
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

    if return_per_axis:
        axial = float(ssim_d.mean())
        coronal = float(ssim_h.mean())
        sagittal = float(ssim_w.mean())
        offaxis = 0.5 * (coronal + sagittal)
        return {
            "ssim": float(ssim_avg),
            "ssim_axial": axial,
            "ssim_coronal": coronal,
            "ssim_sagittal": sagittal,
            "ssim_offaxis": offaxis,
            "ssim_axial_gap": axial - offaxis,
        }

    if size_average:
        return ssim_avg
    return np.array([ssim_avg])


# ---------------------------------------------------------------------------
# Per-axis PSNR
# ---------------------------------------------------------------------------
def cbct_psnr_per_axis(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range: Optional[float] = None,
    clamp_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Slice-wise PSNR averaged within each of the three orthogonal planes.

    The volume PSNR reported by :func:`cbct_psnr` is a single global number and
    so cannot expose an axis asymmetry. This computes PSNR per 2D slice and
    averages within each axis family, giving the PSNR counterpart of
    ``cbct_ssim_3d_gaal(..., return_per_axis=True)``.

    Note that per-slice PSNR is *not* comparable in absolute value to the
    volume PSNR of :func:`cbct_psnr` (each slice is normalised by its own MSE
    before averaging in the log domain); only the differences between axes are
    meaningful. Returns keys ``psnr_axial``/``psnr_coronal``/``psnr_sagittal``
    plus ``psnr_offaxis`` and ``psnr_axial_gap``.
    """
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    g = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    p = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    if g.dim() != 3:
        raise ValueError("cbct_psnr_per_axis requires 3D (D,H,W) tensors")
    if data_range is None:
        data_range = float(g.max() - g.min())

    out: Dict[str, float] = {}
    for axis, name in [(0, "axial"), (1, "coronal"), (2, "sagittal")]:
        # Per-slice MSE over the two in-plane dimensions.
        dims = tuple(d for d in range(3) if d != axis)
        mse = ((g - p) ** 2).mean(dim=dims)
        mse = mse.clamp_min(1e-12)
        out[f"psnr_{name}"] = float((20 * torch.log10(data_range / mse.sqrt())).mean())

    out["psnr_offaxis"] = 0.5 * (out["psnr_coronal"] + out["psnr_sagittal"])
    out["psnr_axial_gap"] = out["psnr_axial"] - out["psnr_offaxis"]
    return out


# ---------------------------------------------------------------------------
# Inter-slice consistency
# ---------------------------------------------------------------------------
def inter_slice_consistency(
    gt: torch.Tensor,
    pred: torch.Tensor,
    clamp_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Adjacent-slice total variation of *pred*, normalised by that of *gt*.

    For each axis, computes the mean absolute difference between adjacent
    slices and divides by the same quantity measured on the ground truth::

        ratio_a = mean|diff(pred, axis=a)| / mean|diff(gt, axis=a)|

    A ratio near 1 means the reconstruction has the same amount of variation
    along that axis as the reference. A ratio above 1 along **axis 0** (the
    axial / through-plane direction, see module docstring) means the volume
    varies more from slice to slice than it should — i.e. spurious inter-slice
    jitter, which is exactly the failure mode of applying a 2D prior
    independently per slice. Ratios below 1 indicate over-smoothing.

    The in-plane ratios (axes 1 and 2) act as a control: a method that is
    simply blurry has all three ratios below 1, whereas a method with an
    inter-slice coherence problem has ``tv_ratio_axial`` elevated relative to
    ``tv_ratio_inplane``.

    Returns ``tv_ratio_axial``, ``tv_ratio_coronal``, ``tv_ratio_sagittal``,
    ``tv_ratio_inplane`` (mean of axes 1 and 2) and ``tv_ratio_excess``
    (axial minus in-plane).
    """
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    g = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    p = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    if g.dim() != 3:
        raise ValueError("inter_slice_consistency requires 3D (D,H,W) tensors")

    out: Dict[str, float] = {}
    for axis, name in [(0, "axial"), (1, "coronal"), (2, "sagittal")]:
        tv_g = float(g.diff(dim=axis).abs().mean())
        tv_p = float(p.diff(dim=axis).abs().mean())
        out[f"tv_ratio_{name}"] = tv_p / max(tv_g, 1e-12)
        out[f"tv_abs_{name}"] = tv_p

    out["tv_ratio_inplane"] = 0.5 * (out["tv_ratio_coronal"] + out["tv_ratio_sagittal"])
    out["tv_ratio_excess"] = out["tv_ratio_axial"] - out["tv_ratio_inplane"]
    return out


# ---------------------------------------------------------------------------
# Low-frequency slice-bias jitter
# ---------------------------------------------------------------------------
def slice_bias_jitter(
    gt: torch.Tensor,
    pred: torch.Tensor,
    clamp_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Adjacent-*slice-mean* jitter, isolating the low-frequency component of
    inter-slice inconsistency from :func:`inter_slice_consistency`'s pixel-level
    TV ratio.

    ``inter_slice_consistency`` differences every voxel between adjacent axial
    slices, so its ratio is dominated by ordinary high-frequency anatomical
    texture (real tissue/material boundaries vary a lot from slice to slice,
    which is legitimate). It under-weights a different, more visually
    conspicuous failure: a *whole-slice* brightness/contrast shift between
    neighbouring slices, which reads as banding when the volume is viewed along
    a coronal or sagittal cut (that cut's "vertical" axis literally is the
    axial slice index, so a brightness step between axial slices becomes a
    visible horizontal band in that cut).

    This computes the same normalised-adjacent-difference ratio, but on the
    1D signal of *per-slice means* rather than on every voxel::

        bias_ratio = mean|diff(pred_slice_means)| / mean|diff(gt_slice_means)|

    Ground-truth slice means are physically smooth (real objects rarely jump in
    average brightness from one slice to the next), so the denominator is
    small and this ratio is far more sensitive to injected banding than the
    pixel-level TV ratio -- at the cost of being blind to genuine high-frequency
    texture inconsistency, which is exactly why it is reported alongside
    :func:`inter_slice_consistency` rather than instead of it.

    Only the axial axis is reported: it is the one relevant to slice-by-slice
    2D reconstruction, and it is what "banding in a coronal/sagittal view"
    refers to (see module docstring).
    """
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    g = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    p = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    if g.dim() != 3:
        raise ValueError("slice_bias_jitter requires 3D (D,H,W) tensors")

    g_means = g.mean(dim=(1, 2))
    p_means = p.mean(dim=(1, 2))
    tv_g = float(g_means.diff().abs().mean())
    tv_p = float(p_means.diff().abs().mean())
    return {
        "slice_bias_ratio": tv_p / max(tv_g, 1e-12),
        "slice_bias_abs": tv_p,
    }


# ---------------------------------------------------------------------------
# Perceptual metric (LPIPS)
# ---------------------------------------------------------------------------
_LPIPS_MODEL = None


def _get_lpips_model(device: torch.device):
    """Lazily construct and cache the AlexNet-backbone LPIPS network."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex")
    return _LPIPS_MODEL.to(device).eval()


def lpips_per_axis(
    gt: torch.Tensor,
    pred: torch.Tensor,
    device: Optional[torch.device] = None,
    clamp_values: Optional[Dict[str, float]] = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Per-axis LPIPS (AlexNet backbone), mirroring :func:`cbct_ssim_3d_gaal`'s
    axis convention: each 2D slice along an axis is a true ``(H, W)`` image
    (grayscale replicated to 3 channels), scored independently and averaged
    within that axis family -- not a marginalised approximation.

    LPIPS is a *distance* (lower = more perceptually similar), the opposite
    sense of SSIM. ``lpips_axial_gap`` keeps the same ``axial - offaxis``
    formula as :func:`cbct_ssim_3d_gaal` for naming consistency, but because it
    is a distance, a *negative* gap (axial distance lower than off-axis) is the
    signature of an unfair axial advantage -- the opposite sign convention from
    the SSIM gap.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp = clamp_values if clamp_values is not None else _DATASET_CLAMP
    g = data_norm(gt.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    p = data_norm(pred.clamp(clamp["clamp_min"], clamp["clamp_max"])).float()
    if g.dim() != 3:
        raise ValueError("lpips_per_axis requires 3D (D,H,W) tensors")

    model = _get_lpips_model(device)

    def axis_mean(g_stack: torch.Tensor, p_stack: torch.Tensor) -> float:
        # g_stack/p_stack: (N, H, W) in [0, 1], N = number of slices along this axis.
        vals = []
        with torch.no_grad():
            for s in range(0, g_stack.shape[0], batch_size):
                e = min(s + batch_size, g_stack.shape[0])
                a = g_stack[s:e].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                b = p_stack[s:e].unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                d = model(a, b, normalize=True).flatten()
                vals.append(d.cpu())
        return float(torch.cat(vals).mean())

    axial = axis_mean(g, p)
    coronal = axis_mean(g.permute(1, 0, 2), p.permute(1, 0, 2))
    sagittal = axis_mean(g.permute(2, 0, 1), p.permute(2, 0, 1))
    offaxis = 0.5 * (coronal + sagittal)
    return {
        "lpips": (axial + coronal + sagittal) / 3.0,
        "lpips_axial": axial,
        "lpips_coronal": coronal,
        "lpips_sagittal": sagittal,
        "lpips_offaxis": offaxis,
        "lpips_axial_gap": axial - offaxis,
    }
