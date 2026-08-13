#!/usr/bin/env python3
"""Coronal/sagittal comparison figure: GT vs DPA vs CDPA vs mu(CDPA) vs DPA+ADMM-TV,
for the walnut 501^3/N=30-views test volumes.

Unlike the 256^3 figures in chip-project/notebooks/cbct_analysis/cbct_visualizations.ipynb
(which pull pre-rendered PNGs logged to W&B during the original reconstruction runs),
the ADMM-TV volumes here were produced by evaluate_volume.py directly to
{output_dir}/volumes/*.npz with --no_wandb, so there is nothing to fetch from W&B --
this script loads the saved volumes directly and slices them itself. Pure
numpy/matplotlib, no torch/astra dependency, since every volume it needs is
already a plain .npz on disk (see scripts/save_gt_volumes note in
evaluate_volume.py's run_save_gt for why GT specifically had to be added there).

Coronal/sagittal views are used rather than axial because both display the
axial axis (axis 0) as one of their two dimensions -- exactly where DPA's
inter-slice jitter and the ADMM-TV variants' axial over-smoothing are visible,
neither of which shows up in an axial slice itself (see
cbct_diffusion.utils.metrics's axis convention: axis 0 = axial, axis 1 =
coronal, axis 2 = sagittal).

Usage
-----
python scripts/plot_admmtv_walnut501.py \\
    --volumes_dir /mydata/sdate/shared/results/cbct501/volumes \\
    --output_dir /myhome/CBCT_conditional_reconstructions/images \\
    --cbct_id 0 1 2 3 4 --view coronal sagittal axial
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

WALNUT_CLAMP_MIN = 0.0
WALNUT_CLAMP_MAX = 0.084

METHODS = [
    ("dpa", "DPA"),
    ("cdpa", "CDPA"),
    ("mu_cdpa_n10", r"$\mu$(CDPA)"),
    ("dpa_admmtv", "DPA+ADMM-TV"),
    ("gt", "Ground Truth"),
]


def load_volume(volumes_dir: Path, dataset: str, nviews: int, cbct_id: int, method: str) -> np.ndarray:
    path = volumes_dir / f"{dataset}_n{nviews}_id{cbct_id}_{method}.npz"
    return np.load(path)["volume"]


def slice_view(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """(D, H, W) volume -> a 2D slice. Coronal/sagittal show the axial axis
    (axis 0) as one of their two dimensions; axial does not (see module
    docstring) -- kept for a complementary, non-inconsistency-revealing view.
    """
    if view == "coronal":
        return volume[:, index, :]
    if view == "sagittal":
        return volume[:, :, index]
    if view == "axial":
        return volume[index, :, :]
    raise ValueError(f"unsupported view {view!r}, expected 'coronal', 'sagittal' or 'axial'")


def plot_comparison_row(
    slices: list[np.ndarray], titles: list[str],
    zoom_size: int = 70, saving_path: Path | None = None,
) -> None:
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))

    h, w = slices[0].shape
    zoom_center = (w // 2, h // 2)

    for ax, sl, title in zip(axes, slices, titles):
        ax.imshow(sl, cmap="gray", vmin=WALNUT_CLAMP_MIN, vmax=WALNUT_CLAMP_MAX)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

        cx, cy = zoom_center
        cx = max(zoom_size, min(w - zoom_size, cx))
        cy = max(zoom_size, min(h - zoom_size, cy))
        ax.add_patch(Rectangle(
            (cx - zoom_size, cy - zoom_size), 2 * zoom_size, 2 * zoom_size,
            fill=False, edgecolor="orange", linewidth=1.5,
        ))
        zoom = sl[cy - zoom_size:cy + zoom_size, cx - zoom_size:cx + zoom_size]
        inset_ax = inset_axes(ax, width="35%", height="35%", loc="lower left", borderpad=0)
        inset_ax.imshow(zoom, cmap="gray", vmin=WALNUT_CLAMP_MIN, vmax=WALNUT_CLAMP_MAX)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("orange")
            spine.set_linewidth(1.5)

    plt.tight_layout()

    if saving_path is not None:
        saving_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"  saved {saving_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--volumes_dir", required=True)
    p.add_argument("--dataset_label", default="walnut501")
    p.add_argument("--nviews", type=int, default=30)
    p.add_argument("--cbct_id", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--view", nargs="+", default=["coronal", "sagittal", "axial"],
                    choices=["coronal", "sagittal", "axial"])
    p.add_argument("--slice_index", type=int, default=None,
                    help="axis-1 (coronal) / axis-2 (sagittal) / axis-0 (axial) index; "
                         "default: volume midpoint")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    volumes_dir = Path(args.volumes_dir)
    output_dir = Path(args.output_dir)

    for cbct_id in args.cbct_id:
        volumes = {}
        for method, _ in METHODS:
            volumes[method] = load_volume(volumes_dir, args.dataset_label, args.nviews, cbct_id, method)
        d, h, w = volumes["gt"].shape

        for view in args.view:
            if args.slice_index is not None:
                index = args.slice_index
            elif view == "coronal":
                index = h // 2
            elif view == "sagittal":
                index = w // 2
            else:
                index = d // 2
            slices = [slice_view(volumes[m], view, index) for m, _ in METHODS]
            titles = [t for _, t in METHODS]
            out_path = output_dir / f"walnut501_admmtv_{view}_id{cbct_id}.pdf"
            print(f"id={cbct_id} view={view} index={index}")
            plot_comparison_row(slices, titles, saving_path=out_path)


if __name__ == "__main__":
    main()
