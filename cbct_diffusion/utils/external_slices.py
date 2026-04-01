"""Load external reconstruction slices and assemble ground-truth volumes.

This module provides helpers for loading Walnut reconstruction slices stored as
individual TIFF/PNG/NPY files and stacking them into a 3D volume.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None  # type: ignore


def external_slice_path_candidates(
    data_root: Path | str,
    split_subdir: str,
    walnut_id: int,
    slice_idx: int,
    prefix: str,
    zfill: int,
    exts: Sequence[str],
    recon_dir_name: str = "Reconstructions",
) -> List[Path]:
    """Return candidate file paths for a given slice index."""
    base = f"{prefix}{slice_idx:0{zfill}d}"
    folder = Path(data_root) / split_subdir / f"Walnut{walnut_id}" / recon_dir_name
    return [folder / f"{base}{ext}" for ext in exts]


def load_external_slice(
    data_root: Path | str,
    split_subdir: str,
    walnut_id: int,
    slice_idx: int,
    prefix: str = "full_AGD_50_",
    zfill: int = 6,
    exts: Sequence[str] = (".tiff", ".tif", ".png", ".npy"),
    recon_dir_name: str = "Reconstructions",
    external_dtype: str | None = None,
    missing_ok: bool = False,
) -> torch.Tensor | None:
    """Load a single external reconstruction slice as ``torch.float32``.

    Returns ``None`` when *missing_ok* is ``True`` and no file is found.
    """
    for cand in external_slice_path_candidates(
        data_root, split_subdir, walnut_id, slice_idx, prefix, zfill, exts, recon_dir_name
    ):
        if cand.is_file():
            if cand.suffix == ".npy":
                arr = np.load(cand)
            else:
                arr = None
                if Image is not None:
                    try:
                        with Image.open(cand) as im:
                            arr = np.array(im)
                    except Exception:
                        pass
                if arr is None and iio is not None:
                    arr = iio.imread(cand)
            if arr is None:
                raise RuntimeError(f"Failed to load slice {cand}")
            if external_dtype is not None:
                arr = arr.astype(external_dtype)
            return torch.from_numpy(arr).to(torch.float32)
    if missing_ok:
        return None
    raise FileNotFoundError(
        f"No external slice found for Walnut{walnut_id} slice {slice_idx}"
    )


def build_external_volume(
    walnut_id: int,
    num_slices: int,
    data_root: Path | str,
    split_candidates: Sequence[str] = ("Test", "Train"),
    prefix: str = "full_AGD_50_",
    zfill: int = 6,
    exts: Sequence[str] = (".tiff", ".tif", ".png", ".npy"),
    recon_dir_name: str = "Reconstructions",
    fallback_volume: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, list[int]]:
    """Assemble a 3D volume by stacking per-slice images.

    Parameters
    ----------
    walnut_id : int
        Walnut sample number.
    num_slices : int
        Number of slices (depth) to assemble.
    data_root : path-like
        Root directory containing ``Train/`` and/or ``Test/`` sub-folders.
    fallback_volume : Tensor, optional
        Volume whose slices are used when the external file is missing.

    Returns
    -------
    (volume, missing_indices)
        ``volume`` has shape ``(D, H, W)``; *missing_indices* lists slices
        that were not found on disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)
    missing: list[int] = []
    slices: list[torch.Tensor] = []

    for z in tqdm(range(num_slices), desc=f"Loading Walnut{walnut_id} slices"):
        loaded = None
        for split in split_candidates:
            loaded = load_external_slice(
                data_root=data_root,
                split_subdir=split,
                walnut_id=walnut_id,
                slice_idx=z,
                prefix=prefix,
                zfill=zfill,
                exts=exts,
                recon_dir_name=recon_dir_name,
                missing_ok=True,
            )
            if loaded is not None:
                break
        if loaded is None:
            missing.append(z)
            if fallback_volume is not None:
                arr = fallback_volume[:, :, z].detach().cpu().numpy().astype(np.float32)
                loaded = torch.from_numpy(arr)
            else:
                loaded = (
                    torch.zeros_like(slices[0])
                    if slices
                    else torch.zeros(1, 1)
                )
        slices.append(loaded)

    vol = torch.stack(slices, dim=0).to(device)
    return vol, missing
