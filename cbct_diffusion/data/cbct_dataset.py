"""CBCTDataset: loads CBCT projections and ground-truth volumes.

This dataset reads data in the format used by the Geometry-Aware-Attenuation-Learning
project. Each sample directory contains:

- ``transforms.json``  — acquisition geometry (source/detector poses per view)
- ``gt_volume.nii.gz``  — ground-truth 3D volume
- ``proj.nii.gz``       — projection images

A JSON split file (``<datatype>_split.json``) specifies which sample directories
belong to each stage (train / val / test).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class CBCTDataset(Dataset):
    """PyTorch dataset for CBCT projection data and ground-truth volumes.

    Parameters
    ----------
    args : object
        Must have attributes: ``datadir``, ``datatype``, ``start``, ``end``,
        ``nviews``, ``angle_sampling``, ``device``.
    stage : str
        One of ``"train"``, ``"val"``, ``"test"``, ``"visual"``.
    """

    def __init__(self, args, stage: str = "train"):
        super().__init__()
        self.args = args
        self.stage = stage
        self.angle_sampling = args.angle_sampling
        self.device = args.device
        self.datadir = args.datadir

        # Locate the split file
        split_paths = [
            os.path.join(args.datadir, "dataset_split", args.datatype + "_split.json"),
            os.path.join(os.path.dirname(args.datadir), "dataset_split", args.datatype + "_split.json"),
            os.path.join(
                os.path.dirname(os.path.dirname(args.datadir)),
                "dataset_split", args.datatype + "_split.json",
            ),
        ]

        dataset_split_json = None
        for path in split_paths:
            if os.path.exists(path):
                dataset_split_json = path
                break

        if dataset_split_json is None:
            raise FileNotFoundError(
                f"Could not find split JSON for '{args.datatype}'. Tried: {split_paths}"
            )

        with open(dataset_split_json, "r") as fh:
            json_data = json.load(fh)
        self.dataset_split = json_data[stage]
        print(
            f"CBCTDataset: {self.datadir}, stage={stage}, "
            f"samples={len(self.dataset_split)}"
        )

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, index: int):
        if sitk is None:
            raise ImportError(
                "SimpleITK is required. Install with: pip install SimpleITK"
            )

        # Geometry
        paras_json = os.path.join(self.datadir, self.dataset_split[index], "transforms.json")
        with open(paras_json, "r") as fh:
            paras = json.load(fh)

        # Ground-truth volume
        volume_path = os.path.join(self.datadir, self.dataset_split[index], "gt_volume.nii.gz")
        volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
        volume = np.clip(volume, 0, volume.max())
        volume_tensor = torch.tensor(volume, dtype=torch.float32, device=self.device)

        # Projections
        img_path = os.path.join(self.datadir, self.dataset_split[index], "proj.nii.gz")
        proj = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        proj = np.clip(proj, 0, proj.max())

        # Select views uniformly
        start, end, nviews = self.args.start, self.args.end, self.args.nviews
        angle_per_view = paras["angle_per_view"]
        start_index = int(np.round(start / angle_per_view))
        end_index = int(np.round(end / angle_per_view))
        indices = np.linspace(start_index, end_index, nviews, endpoint=False, dtype=int)

        all_imgs = torch.tensor(
            proj[indices], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        vecs = []
        for i in indices:
            vec = torch.tensor(paras["frames"][i]["vec"], dtype=torch.float32, device=self.device)
            vecs.append(vec)
        vecs = torch.stack(vecs)

        return {
            "paras": paras,
            "3Dvolume": volume_tensor,
            "images": all_imgs,
            "poses": vecs,
            "obj_index": paras["obj_index"],
        }
