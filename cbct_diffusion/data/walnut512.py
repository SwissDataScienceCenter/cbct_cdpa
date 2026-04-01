"""Walnut512: in-memory dataset for high-resolution Walnut CBCT data.

This dataset loads raw Walnut projection data from disk (TIFF images + ASTRA
geometry files), builds external ground-truth volumes from pre-reconstructed
slices, and generates sparse-view FDK reconstructions on the fly.

It is used for training and evaluation at the native 501³ resolution.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from astra_torch import fdk_reconstruction_masked
from cbct_diffusion.utils.external_slices import build_external_volume


def _discover_walnuts(root: str) -> List[int]:
    """Return sorted list of Walnut IDs under *root*."""
    ids = []
    for name in os.listdir(root):
        if name.startswith("Walnut"):
            try:
                ids.append(int(name.replace("Walnut", "")))
            except ValueError:
                pass
    return sorted(ids)


def _load_walnut_projections(
    data_path: str,
    walnut_id: int,
    orbit_id: int = 1,
    angular_sub_sampling: int = 1,
    voxel_per_mm: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """Load and preprocess raw Walnut projections.

    This function reads flat/dark fields and projection TIFFs, applies
    flat-field correction and negative-log transform, and returns the
    projection tensor in ``(V, R, C)`` layout together with ASTRA geometry
    vectors.

    Parameters
    ----------
    data_path : str
        Directory containing ``WalnutX/Projections/tubeV1/`` etc.
    walnut_id : int
        Walnut number.
    orbit_id : int
        Tube / orbit number.
    angular_sub_sampling : int
        Keep every *n*-th projection.
    voxel_per_mm : int
        Voxel density.
    device : torch.device, optional
        Target device (default: CUDA if available).
    verbose : bool
        Print progress bar.

    Returns
    -------
    projs_vrc : torch.Tensor
        ``(V, R, C)`` projection tensor.
    vecs : np.ndarray
        ``(V, 12)`` ASTRA geometry vectors.
    meta : dict
        ``{"voxel_per_mm", "voxel_size_mm"}``.
    """
    import imageio.v2 as imageio
    from concurrent.futures import ThreadPoolExecutor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tube_dir = os.path.join(data_path, f"Walnut{walnut_id}", "Projections", f"tubeV{orbit_id}")
    vecs_full = np.loadtxt(os.path.join(tube_dir, "scan_geom_corrected.geom"))
    vecs = vecs_full[range(0, 1200, angular_sub_sampling)]

    projs_idx = range(1200, 0, -angular_sub_sampling)
    proj_rows, proj_cols = 972, 768
    n_pro = vecs.shape[0]

    def _trafo(im_np):
        t = torch.from_numpy(im_np).to(device, dtype=torch.float32)
        return torch.transpose(torch.flipud(t), 0, 1)

    dark = _trafo(imageio.imread(os.path.join(tube_dir, "di000000.tif")))
    flat = torch.stack(
        [_trafo(imageio.imread(os.path.join(tube_dir, f"io00000{i}.tif"))) for i in range(2)],
        dim=0,
    ).mean(dim=0)

    projs = torch.zeros((n_pro, proj_rows, proj_cols), dtype=torch.float32, device=device)

    def _load_one(i):
        path = os.path.join(tube_dir, f"scan_{projs_idx[i]:06d}.tif")
        return i, _trafo(imageio.imread(path))

    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_load_one, i): i for i in range(n_pro)}
        for fut in tqdm(futures, total=n_pro, desc="Loading projections", disable=not verbose):
            i, t = fut.result()
            projs[i] = t

    denom = (flat - dark).clamp(min=1e-6)
    projs = -torch.log((projs - dark) / denom)
    projs = projs.permute(1, 0, 2).contiguous().permute(1, 0, 2)

    voxel_size_mm = 1.0 / voxel_per_mm
    return projs, vecs, {"voxel_per_mm": voxel_per_mm, "voxel_size_mm": voxel_size_mm}


def _uniform_mask(n: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean mask of length *n* with exactly *k* ``True`` entries."""
    if rng is None:
        rng = np.random.default_rng()
    if k > n:
        raise ValueError("k cannot exceed number of views")
    mask = np.zeros(n, dtype=bool)
    step = math.ceil(n / k)
    fixed = np.arange(0, n, step)
    mask[fixed] = True
    remaining = k - int(mask.sum())
    if remaining > 0:
        available = np.where(~mask)[0]
        mask[rng.choice(available, size=remaining, replace=False)] = True
    return mask


class Walnut512(Dataset):
    """In-memory Walnut dataset with sparse-view FDK reconstructions.

    Parameters
    ----------
    data_path : str
        Root directory (contains ``Train/`` or ``Test/``).
    split_subdir : str
        ``"Train"`` or ``"Test"``.
    orbit_id : int
        Orbit / tube index.
    angular_sub_sampling : int
        Sub-sampling factor for raw projections.
    voxel_per_mm : int
        Voxel density.
    device : torch.device
        Computation device.
    k : int, optional
        Number of views for sparse reconstruction (``None`` = full).
    seed : int
        RNG seed for reproducible view selection.
    axis : int
        Slice axis (0, 1, or 2).
    limit : int
        Max walnuts to load (``-1`` = all).
    normalize_factor : float
        Normalisation divisor.
    walnut_range : tuple, optional
        ``(start, end)`` indices into the sorted walnut-ID list.
    augment : bool
        Enable random augmentation.
    """

    def __init__(
        self,
        data_path: str,
        split_subdir: str = "Train",
        orbit_id: int = 1,
        angular_sub_sampling: int = 1,
        voxel_per_mm: int = 10,
        device: torch.device = torch.device("cpu"),
        k: Optional[int] = None,
        seed: int = 42,
        axis: int = 2,
        limit: int = -1,
        normalize_factor: float = 0.0,
        walnut_range: Optional[Tuple[int, int]] = (0, -1),
        augment: bool = False,
        recon_dir_name: str = "Reconstructions",
        recon_prefix: str = "full_AGD_50_",
        recon_zfill: int = 6,
        recon_exts: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.split_subdir = split_subdir
        self.orbit_id = orbit_id
        self.angular_sub_sampling = angular_sub_sampling
        self.voxel_per_mm = voxel_per_mm
        self.voxel_size_mm = 1.0 / voxel_per_mm
        self.device = device
        self.k = k
        self.seed = seed
        self.axis = axis
        self.normalize_factor = normalize_factor
        self.augment = augment
        self.limit = limit
        self.walnut_range = walnut_range or (0, -1)

        self.recon_dir_name = recon_dir_name
        self.recon_prefix = recon_prefix
        self.recon_zfill = recon_zfill
        self.recon_exts = list(recon_exts) if recon_exts else [".tiff", ".tif", ".png", ".npy"]

        self.rng = np.random.default_rng(seed)

        # Storage
        self.walnut_ids: List[int] = []
        self.projections: list = []
        self.vecs: list = []
        self.external_volumes: list = []
        self.reconstructions: list = []
        self.masks: list = []

        self._load_all_data()
        self.total_slices = sum(v.shape[self.axis] for v in self.reconstructions)
        self.rebuild_dataset(k=self.k)

    # ------------------------------------------------------------------
    def _load_all_data(self):
        root = os.path.join(self.data_path, self.split_subdir)
        self.walnut_ids = _discover_walnuts(root)[self.walnut_range[0] : self.walnut_range[1]]
        if not self.walnut_ids:
            raise RuntimeError(f"No Walnut* folders in {root}")

        print(f"Loading {len(self.walnut_ids)} walnuts …")
        for wid in tqdm(self.walnut_ids, desc="Loading"):
            if 0 < self.limit <= len(self.projections):
                break

            projs, vecs, _ = _load_walnut_projections(
                root, walnut_id=wid, orbit_id=self.orbit_id,
                angular_sub_sampling=self.angular_sub_sampling,
                voxel_per_mm=self.voxel_per_mm,
                device=torch.device("cpu"), verbose=False,
            )
            self.projections.append(projs.contiguous())
            self.vecs.append(vecs.copy())

            num_slices = projs.shape[1]
            ext_vol, _ = build_external_volume(
                walnut_id=wid, num_slices=num_slices,
                data_root=self.data_path,
                split_candidates=(self.split_subdir,),
                prefix=self.recon_prefix, zfill=self.recon_zfill,
                exts=self.recon_exts, recon_dir_name=self.recon_dir_name,
                device=torch.device("cpu"),
            )
            self.external_volumes.append(ext_vol)

    # ------------------------------------------------------------------
    def rebuild_dataset(self, k: Optional[int] = None):
        """Re-generate sparse masks and FDK reconstructions for *k* views."""
        self.reconstructions.clear()
        self.masks.clear()
        rng = np.random.default_rng(self.seed)

        for idx in range(len(self.projections)):
            n_views = self.vecs[idx].shape[0]
            if k is not None and k < n_views:
                mask = _uniform_mask(n_views, k, rng)
            else:
                mask = np.ones(n_views, dtype=bool)
            self.masks.append(mask)

            fdk_vol = fdk_reconstruction_masked(
                projs_vrc=self.projections[idx].to(self.device),
                vecs=self.vecs[idx],
                mask=mask,
                voxel_per_mm=self.voxel_per_mm,
                voxel_size_mm=self.voxel_size_mm,
                device=self.device,
            ).cpu()
            self.reconstructions.append(fdk_vol)

        self.total_slices = sum(v.shape[self.axis] for v in self.reconstructions)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.total_slices

    def __getitem__(self, global_idx: int):
        # Map global index to (walnut_idx, slice_idx)
        offset = 0
        for w_idx, vol in enumerate(self.reconstructions):
            n = vol.shape[self.axis]
            if global_idx < offset + n:
                local = global_idx - offset
                break
            offset += n
        else:
            raise IndexError(global_idx)

        recon_vol = self.reconstructions[w_idx]
        ext_vol = self.external_volumes[w_idx]

        def _get_slice(vol, idx, ax):
            if ax == 0:
                return vol[idx, :, :]
            elif ax == 1:
                return vol[:, idx, :]
            else:
                return vol[:, :, idx]

        slice_recon = _get_slice(recon_vol, local, self.axis).clone()
        slice_ext = _get_slice(ext_vol, local, self.axis).clone()

        if self.normalize_factor > 0:
            slice_recon = slice_recon / self.normalize_factor
            slice_ext = slice_ext / self.normalize_factor

        k_used = int(self.masks[w_idx].sum())
        return slice_recon, slice_ext, k_used, local
