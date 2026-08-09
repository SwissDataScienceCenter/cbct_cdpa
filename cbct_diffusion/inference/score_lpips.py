#!/usr/bin/env python3
"""Post-hoc LPIPS scoring over ALREADY-RECONSTRUCTED, saved volumes.

``evaluate_volume.py`` persists every reconstruction to
``{output_dir}/volumes/{dataset}_n{nviews}_id{cbct_id}_{method}.npz`` precisely
so that a new metric does not require re-running inference (see its module
docstring, reason 1). This script is that offline post-processing step for
LPIPS: it reloads the ground truth via the same ``load_volume`` path used
originally (so the reference is identical to the one PSNR/SSIM were scored
against), reloads each saved reconstruction, and writes per-(volume, method)
LPIPS results to a separate ``{output_dir}/lpips/`` directory -- it never
touches the existing ``metrics/`` JSONs.

Resumable like the rest of the suite: each (dataset, id, method) result is
skipped if its JSON already exists.

Usage
-----
python -m cbct_diffusion.inference.score_lpips \\
    --data_path /mydata/sdate/shared/data/huggingface/walnut \\
    --cbct_id 0 1 2 3 4 --nviews 20 --output_dir /mydata/sdate/shared/results/cbct256

python -m cbct_diffusion.inference.score_lpips \\
    --data_path /mydata/sdate/shared/data/walnut --high_resolution \\
    --dataset_label walnut501 --cbct_id 0 1 2 3 4 --nviews 30 \\
    --output_dir /mydata/sdate/shared/results/cbct501
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

import cbct_diffusion  # noqa: F401  (import-order guard, see cbct_diffusion/__init__.py)

from cbct_diffusion.inference.evaluate_volume import load_volume
from cbct_diffusion.utils.metrics import lpips_per_axis, set_dataset_clamp, get_clamp_by_name

# Volumes that are not reconstructions to score against GT: the per-voxel
# posterior STD maps saved alongside mu(DPA)/mu(CDPA) for the uncertainty
# analysis (see evaluate_volume.run_diffusion).
_SKIP_METHOD_PREFIXES = ("std_",)


def _result_path(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str) -> Path:
    return output_dir / "lpips" / f"{dataset}_n{nviews}_id{cbct_id}_{method}.json"


def done(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str) -> bool:
    p = _result_path(output_dir, dataset, cbct_id, nviews, method)
    if not p.is_file():
        return False
    try:
        json.loads(p.read_text())
        return True
    except json.JSONDecodeError:
        print(f"  (corrupt result {p.name} — recomputing)")
        return False


def emit(output_dir: Path, dataset: str, cbct_id: int, nviews: int, method: str,
         result: dict) -> None:
    rp = _result_path(output_dir, dataset, cbct_id, nviews, method)
    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        **result, "dataset": dataset, "cbct_id": cbct_id, "nviews": nviews, "method": method,
    }, indent=2))
    tmp.replace(rp)


def methods_for(output_dir: Path, dataset: str, cbct_id: int, nviews: int) -> list[str]:
    """Every method with a saved reconstruction volume for this (dataset, id)."""
    pattern = str(output_dir / "volumes" / f"{dataset}_n{nviews}_id{cbct_id}_*.npz")
    out = []
    for f in sorted(glob.glob(pattern)):
        method = Path(f).stem[len(f"{dataset}_n{nviews}_id{cbct_id}_"):]
        if not method.startswith(_SKIP_METHOD_PREFIXES):
            out.append(method)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc LPIPS scoring over saved reconstructions")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--cbct_id", type=int, nargs="+", required=True)
    p.add_argument("--nviews", type=int, default=20)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--high_resolution", action="store_true")
    p.add_argument("--dataset_label", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32,
                   help="slices per LPIPS forward-pass batch")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    dataset = args.dataset_label or Path(args.data_path).name
    set_dataset_clamp(get_clamp_by_name(dataset))
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    ids = list(args.cbct_id)
    print(f"=== LPIPS: {dataset} ids={ids} nviews={args.nviews} device={device}")

    t_all = time.time()
    for cid in ids:
        methods = methods_for(output_dir, dataset, cid, args.nviews)
        pending = [m for m in methods if not done(output_dir, dataset, cid, args.nviews, m)]
        if not pending:
            print(f"--- id={cid}: all {len(methods)} method(s) already scored")
            continue

        args.cbct_id = cid  # load_volume expects the current id, matching evaluate_volume's loop
        t0 = time.time()
        d = load_volume(args, device)
        gt = d["gt"]
        print(f"--- id={cid}: loaded gt{tuple(gt.shape)} in {time.time()-t0:.1f}s, "
              f"scoring {len(pending)}/{len(methods)} method(s)")

        for method in pending:
            vp = output_dir / "volumes" / f"{dataset}_n{args.nviews}_id{cid}_{method}.npz"
            pred = torch.from_numpy(np.load(vp)["volume"]).to(gt.device)
            t1 = time.time()
            result = lpips_per_axis(gt, pred, device=device, batch_size=args.batch_size)
            dt = time.time() - t1
            emit(output_dir, dataset, cid, args.nviews, method, {**result, "scoring_time_s": dt})
            print(f"  [{method:22s}] LPIPS {result['lpips']:.4f} "
                  f"(ax {result['lpips_axial']:.4f} / off {result['lpips_offaxis']:.4f}, "
                  f"gap {result['lpips_axial_gap']:+.4f})  ({dt:.1f}s)", flush=True)
            del pred

    print(f"=== all {len(ids)} volume(s) in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
