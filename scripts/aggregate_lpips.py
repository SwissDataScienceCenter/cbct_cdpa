#!/usr/bin/env python3
"""Aggregate LPIPS results written by score_lpips.py.

Mirrors aggregate_cbct_256.py's load()/agg() pattern, adapted to the separate
lpips/ output directory and its own (dataset, nviews) pairs, since the 256^3
suite and the 501^3/30-view walnut suite live under different output roots.

Usage
-----
python scripts/aggregate_lpips.py
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# (output_dir, dataset, nviews, expected n)
TARGETS = [
    ("/mydata/sdate/shared/results/cbct256", "dental", 20, 20),
    ("/mydata/sdate/shared/results/cbct256", "spine", 20, 20),
    ("/mydata/sdate/shared/results/cbct256", "walnut", 20, 5),
    ("/mydata/sdate/shared/results/cbct501", "walnut501", 30, 5),
]

N_SAMPLES = 10
METHOD_ORDER = [
    ("fdk", "FDK"),
    ("dpa", "DPA"),
    ("cdpa", "CDPA"),
    ("unet", "FDK-denoiser"),
    (f"mu_dpa_n{N_SAMPLES}", "mu(DPA)"),
    (f"mu_cdpa_n{N_SAMPLES}", "mu(CDPA)"),
    ("unet_ft", "FDK-denoiser + FT"),
]

FNAME_RE = re.compile(r"^(?P<ds>\w+?)_n(?P<nv>\d+)_id(?P<id>\d+)_(?P<method>.+)$")


def load(output_dir: str, dataset: str, nviews: int) -> dict:
    out: dict = defaultdict(dict)
    for f in sorted(glob.glob(f"{output_dir}/lpips/*.json")):
        m = FNAME_RE.match(Path(f).stem)
        if not m or m["ds"] != dataset or int(m["nv"]) != nviews:
            continue
        try:
            d = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            print(f"  ! skipping unreadable {Path(f).name}")
            continue
        out[m["method"]][int(m["id"])] = d
    return out


def agg(per_id: dict, key: str) -> tuple[float, float, int]:
    v = np.array([per_id[i][key] for i in sorted(per_id) if key in per_id[i]], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def main() -> None:
    for output_dir, dataset, nviews, expected in TARGETS:
        data = load(output_dir, dataset, nviews)
        if not data:
            print(f"\n### {dataset} (n{nviews}): no LPIPS results yet under {output_dir}/lpips")
            continue
        print(f"\n{'='*104}\n### {dataset}  ({nviews} views, expected n={expected})\n{'='*104}")
        hdr = (f"{'method':20s} {'n':>3s} {'LPIPS':>15s} "
               f"{'ax':>8s} {'cor':>8s} {'sag':>8s} {'gap':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for meth, label in METHOD_ORDER:
            per_id = data.get(meth)
            if not per_id:
                print(f"{label:20s} {'--':>3s}  (not available)")
                continue
            n = len(per_id)
            m, s, _ = agg(per_id, "lpips")
            f = lambda k: agg(per_id, k)[0]
            flag = "" if n == expected else f"  <-- {n}/{expected}"
            print(f"{label:20s} {n:3d} {m:7.4f}±{s:.4f} "
                  f"{f('lpips_axial'):8.4f} {f('lpips_coronal'):8.4f} "
                  f"{f('lpips_sagittal'):8.4f} {f('lpips_axial_gap'):+8.4f}{flag}")


if __name__ == "__main__":
    main()
