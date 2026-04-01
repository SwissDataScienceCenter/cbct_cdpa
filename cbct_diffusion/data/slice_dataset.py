"""SliceCBCTDataset: 2D slice-based dataset for CBCT reconstruction training.

This dataset loads 3D CBCT volumes via :class:`CBCTDataset`, computes FDK
reconstructions from their projections, and serves paired 2D slices:

- **FDK reconstruction slice** (low-quality input)
- **Ground-truth volume slice** (supervision target)

The paired slices can be used for training both UNet and diffusion models.

Each ``__getitem__`` call returns a tuple::

    (slice_fdk, slice_gt, k, slice_idx)

where ``k`` is the number of projection views and ``slice_idx`` is the
position of the slice within the 3D volume.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from astra_torch import fdk_reconstruction_masked
from cbct_diffusion.data.cbct_dataset import CBCTDataset


def create_cbct_args(
    datadir: str,
    start: float = 0,
    end: float = 360,
    nviews: int = 20,
    angle_sampling: str = "uniform",
    device: Optional[torch.device] = None,
):
    """Create an arguments namespace for :class:`CBCTDataset`.

    Parameters
    ----------
    datadir : str
        Root directory containing the CBCT data.
    start, end : float
        Angular range in degrees.
    nviews : int
        Number of projection views.
    angle_sampling : str
        Sampling strategy (only ``"uniform"`` is supported).
    device : torch.device, optional
        Device for tensor allocation (default: CUDA if available).
    """

    class Args:
        def __init__(self):
            self.datadir = datadir
            self.datatype = os.path.basename(datadir)
            self.start = start
            self.end = end
            self.nviews = nviews
            self.angle_sampling = angle_sampling
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    return Args()


class SliceCBCTDataset(Dataset):
    """2D-slice dataset pairing FDK reconstructions with ground truth.

    Parameters
    ----------
    args : object
        Arguments created via :func:`create_cbct_args`.
    stage : str
        Dataset split: ``"train"``, ``"val"``, ``"test"``.
    slice_axis : str
        Axis along which to extract slices: ``"axial"``, ``"coronal"``, or
        ``"sagittal"``.
    device : torch.device, optional
        Computation device.
    preload_all : bool
        If ``True``, load all volumes and compute FDK at initialisation.
    normalize_factor : float
        Divide all slices by this value (set to ``clamp_max`` for [0, 1] range).
    augment : bool
        Apply random augmentation during training.
    limit : int
        Maximum number of volumes to load (``-1`` = no limit).
    """

    def __init__(
        self,
        args,
        stage: str = "train",
        slice_axis: Literal["axial", "coronal", "sagittal"] = "axial",
        device: Optional[torch.device] = None,
        preload_all: bool = True,
        normalize_factor: float = 1.0,
        augment: bool = False,
        limit: int = -1,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.args = args
        self.args.device = device
        self.stage = stage
        self.slice_axis = slice_axis

        dataset_prefix = os.path.basename(args.datadir)
        self.voxel_per_mm = 10.0 / 1.957
        if dataset_prefix == "walnut":
            self.voxel_size_mm = 1.0 / self.voxel_per_mm
        elif dataset_prefix == "dental":
            self.voxel_size_mm = 1.61 / self.voxel_per_mm
        else:
            self.voxel_size_mm = 2.0

        self.normalize_factor = normalize_factor
        self.augment = augment and stage == "train"
        self.limit = limit

        self.cbct_dataset = CBCTDataset(args, stage=stage)

        axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
        if slice_axis not in axis_map:
            raise ValueError(f"slice_axis must be one of {list(axis_map.keys())}")
        self.axis = axis_map[slice_axis]

        self.fdk_volumes: list = []
        self.gt_volumes: list = []
        self.projections: list = []
        self.vecs: list = []
        self.k_values: list = []
        self.volume_shapes: list = []

        if preload_all:
            self._preload_all_data()

        self._calculate_slice_indices()

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------
    def _preload_all_data(self):
        """Load all volumes and compute FDK reconstructions."""
        print(f"Preloading {len(self.cbct_dataset)} volumes …")

        for i in tqdm(range(len(self.cbct_dataset)), desc="Loading volumes"):
            if 0 < self.limit <= i:
                print(f"Limiting to {self.limit} volumes.")
                break

            sample = self.cbct_dataset[i]
            gt_volume = sample["3Dvolume"]
            self.gt_volumes.append(gt_volume)
            self.volume_shapes.append(gt_volume.shape)

            projs_vrc = sample["images"].squeeze(1)
            vecs_np = sample["poses"].detach().cpu().numpy().astype(np.float32)
            k = projs_vrc.shape[0]
            self.k_values.append(k)
            self.projections.append(projs_vrc)
            self.vecs.append(vecs_np)

            try:
                fdk_volume = fdk_reconstruction_masked(
                    projs_vrc=projs_vrc,
                    vecs=vecs_np,
                    mask=None,
                    voxel_per_mm=self.voxel_per_mm,
                    voxel_size_mm=self.voxel_size_mm,
                    device=self.device,
                )
                # Resize if shape mismatch
                if fdk_volume.shape != gt_volume.shape:
                    fdk_volume = F.interpolate(
                        fdk_volume.unsqueeze(0).unsqueeze(0),
                        size=gt_volume.shape,
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                self.fdk_volumes.append(fdk_volume)
            except Exception as exc:
                print(f"FDK failed for volume {i}: {exc}. Using GT as fallback.")
                self.fdk_volumes.append(gt_volume.clone())

        print(
            f"Loaded {len(self.fdk_volumes)} FDK and "
            f"{len(self.gt_volumes)} GT volumes."
        )

    def _calculate_slice_indices(self):
        """Build mapping from global slice index to ``(volume_idx, local_slice_idx)``."""
        self.slice_map = []
        total = 0
        for vol_idx, shape in enumerate(self.volume_shapes):
            n = shape[self.axis]
            for s in range(n):
                self.slice_map.append((vol_idx, s))
            total += n
        self._total_slices = total
        print(f"Total slices ({self.slice_axis}): {total}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total_slices

    def __getitem__(self, global_idx: int):
        vol_idx, slice_idx = self.slice_map[global_idx]
        fdk_vol = self.fdk_volumes[vol_idx].clone()
        gt_vol = self.gt_volumes[vol_idx].clone()
        k = self.k_values[vol_idx]

        if self.axis == 0:
            slice_fdk = fdk_vol[slice_idx, :, :]
            slice_gt = gt_vol[slice_idx, :, :]
        elif self.axis == 1:
            slice_fdk = fdk_vol[:, slice_idx, :]
            slice_gt = gt_vol[:, slice_idx, :]
        else:
            slice_fdk = fdk_vol[:, :, slice_idx]
            slice_gt = gt_vol[:, :, slice_idx]

        slice_fdk = self.normalize(slice_fdk.clone())
        slice_gt = self.normalize(slice_gt.clone())

        if self.augment:
            slice_fdk, slice_gt = self._apply_augmentation(slice_fdk, slice_gt)

        return slice_fdk, slice_gt, k, slice_idx

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def _apply_augmentation(
        self, slice_fdk: torch.Tensor, slice_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random rotation, scaling, and flips to both slices."""
        combined = torch.stack([slice_fdk, slice_gt]).unsqueeze(0)  # (1,2,H,W)

        if torch.rand(1) > 0.5:
            angle = torch.rand(1) * 180
            combined = self._rotate_tensor(combined, angle.item())
        if torch.rand(1) > 0.5:
            scale = 0.9 + torch.rand(1) * 0.2
            combined = self._scale_tensor(combined, scale.item())
        if torch.rand(1) > 0.5:
            combined = torch.flip(combined, dims=[-1])
        if torch.rand(1) > 0.5:
            combined = torch.flip(combined, dims=[-2])

        return combined[0, 0], combined[0, 1]

    @staticmethod
    def _rotate_tensor(tensor: torch.Tensor, angle: float) -> torch.Tensor:
        rad = torch.tensor(angle * 3.14159 / 180.0)
        cos_a, sin_a = torch.cos(rad), torch.sin(rad)
        mat = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=tensor.dtype, device=tensor.device,
        ).unsqueeze(0)
        grid = F.affine_grid(mat, tensor.size(), align_corners=False)
        return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    @staticmethod
    def _scale_tensor(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        mat = torch.tensor(
            [[scale, 0, 0], [0, scale, 0]],
            dtype=tensor.dtype, device=tensor.device,
        ).unsqueeze(0)
        grid = F.affine_grid(mat, tensor.size(), align_corners=False)
        return F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Divide by ``normalize_factor``."""
        if self.normalize_factor > 0:
            tensor = tensor / self.normalize_factor
        return tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Multiply by ``normalize_factor``."""
        if self.normalize_factor > 0:
            tensor = tensor * self.normalize_factor
        return tensor

    def get_volume_info(self) -> dict:
        return {
            "num_volumes": len(self.fdk_volumes),
            "total_slices": self._total_slices,
            "slice_axis": self.slice_axis,
            "k_values": self.k_values,
            "volume_shapes": self.volume_shapes,
        }
