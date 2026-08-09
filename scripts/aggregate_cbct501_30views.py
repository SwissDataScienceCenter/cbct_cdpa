#!/usr/bin/env python3
"""Aggregate the 501^3/30-view walnut sweep: per-axis SSIM, TV-ratio, slice-bias-jitter.

Same load()/agg() approach as scripts/aggregate_cbct_256.py, adapted to this
sweep's output dir, dataset label ("walnut501"), and view count (30). Exists
mainly to answer one question against the 20-view walnut results already in
hand: does the extra view density change the DPA-vs-CDPA inter-slice-
consistency story, given the paper already flags 20-view walnut as
under-determined for every method?

Usage
-----
python scripts/aggregate_cbct501_30views.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTPUT_DIR = "/mydata/sdate/shared/results/cbct501"
NVIEWS = 30
N_SAMPLES = 10
DATASET = "walnut501"
EXPECTED_VOLUMES = 5

METHOD_ORDER = [
    ("fdk", "FDK"),
    ("gd_zero", r"GD$_{zero}$"),
    ("gd_fdk", r"GD$_{FDK}$"),
    ("dpa", "DPA"),
    ("cdpa", "CDPA"),
    ("unet", "FDK-denoiser (ours)"),
    (f"mu_dpa_n{N_SAMPLES}", r"$\mu$(DPA)"),
    (f"mu_cdpa_n{N_SAMPLES}", r"$\mu$(CDPA)"),
    ("unet_ft", "FDK-denoiser + FT (ours)"),
]

FNAME_RE = re.compile(r"^(?P<ds>\w+?)_n(?P<nv>\d+)_id(?P<id>\d+)_(?P<method>.+)$")


def load() -> dict:
    """-> {method: {cbct_id: metrics_dict}}"""
    out: dict = defaultdict(dict)
    for f in sorted(glob.glob(f"{OUTPUT_DIR}/metrics/*.json")):
        m = FNAME_RE.match(Path(f).stem)
        if not m or m["ds"] != DATASET or int(m["nv"]) != NVIEWS:
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


def wilcoxon(a: dict, b: dict, key: str) -> tuple[float, int]:
    ids = sorted(set(a) & set(b))
    x = np.array([a[i][key] for i in ids], dtype=float)
    y = np.array([b[i][key] for i in ids], dtype=float)
    if len(ids) < 3 or np.allclose(x, y):
        return float("nan"), len(ids)
    try:
        from scipy.stats import wilcoxon as w
        return float(w(x, y).pvalue), len(ids)
    except Exception:
        return float("nan"), len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=f"mu_cdpa_n{N_SAMPLES}")
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit(f"No results found under {OUTPUT_DIR}/metrics for {DATASET} n{NVIEWS}")

    print(f"\n{'='*130}\n### walnut  (501^3, {NVIEWS} views, expected n={EXPECTED_VOLUMES} test volumes)\n{'='*130}")
    hdr = (f"{'method':26s} {'n':>3s} {'PSNR':>13s} {'SSIM':>14s} "
           f"{'SSIM_ax':>8s} {'SSIM_cor':>9s} {'SSIM_sag':>9s} {'gap':>8s} "
           f"{'tv_ax':>7s} {'tv_cor':>7s} {'tv_sag':>7s} {'tv_inpl':>8s} {'excess':>8s} "
           f"{'bias_ratio':>10s}")
    print(hdr)
    print("-" * len(hdr))
    incomplete = []
    for meth, label in METHOD_ORDER:
        per_id = data.get(meth)
        if not per_id:
            incomplete.append((meth, 0))
            print(f"{label:26s} {'--':>3s}  (not yet available)")
            continue
        n = len(per_id)
        if n != EXPECTED_VOLUMES:
            incomplete.append((meth, n))
        pm, ps, _ = agg(per_id, "psnr")
        sm, ss, _ = agg(per_id, "ssim")
        f = lambda k: agg(per_id, k)[0]
        flag = "" if n == EXPECTED_VOLUMES else f"  <-- {n}/{EXPECTED_VOLUMES}"
        print(f"{label:26s} {n:3d} {pm:7.2f}±{ps:4.2f} {sm:8.4f}±{ss:.4f} "
              f"{f('ssim_axial'):8.4f} {f('ssim_coronal'):9.4f} {f('ssim_sagittal'):9.4f} "
              f"{f('ssim_axial_gap'):+8.4f} {f('tv_ratio_axial'):7.3f} "
              f"{f('tv_ratio_coronal'):7.3f} {f('tv_ratio_sagittal'):7.3f} "
              f"{f('tv_ratio_inplane'):8.3f} {f('tv_ratio_excess'):+8.3f} "
              f"{f('slice_bias_ratio'):10.3f}{flag}")

    ref = data.get(args.reference)
    if ref:
        print(f"\n  Paired Wilcoxon signed-rank vs {args.reference} (PSNR / SSIM / tv_ratio_axial / slice_bias_ratio):")
        for meth, label in METHOD_ORDER:
            if meth == args.reference or meth not in data:
                continue
            pp, npair = wilcoxon(data[meth], ref, "psnr")
            sp, _ = wilcoxon(data[meth], ref, "ssim")
            tp, _ = wilcoxon(data[meth], ref, "tv_ratio_axial")
            bp, _ = wilcoxon(data[meth], ref, "slice_bias_ratio")
            def fmt(p):
                return "  n/a " if np.isnan(p) else f"{p:.4f}" + ("*" if p < 0.05 else " ")
            print(f"    {label:26s} n={npair:2d}  p_psnr={fmt(pp)}  p_ssim={fmt(sp)}  "
                  f"p_tv_ax={fmt(tp)}  p_bias={fmt(bp)}")

    print(f"\n{'='*130}\n### DPA vs CDPA slice-bias-jitter ratio, 30 views  (paired per volume)\n{'='*130}")
    for m_uncond, m_cond, lbl_uncond, lbl_cond in [
        ("dpa", "cdpa", "DPA", "CDPA"),
        (f"mu_dpa_n{N_SAMPLES}", f"mu_cdpa_n{N_SAMPLES}", r"mu(DPA)", r"mu(CDPA)"),
    ]:
        a, b = data.get(m_uncond), data.get(m_cond)
        if not a or not b:
            print(f"  {lbl_uncond} vs {lbl_cond}: missing data")
            continue
        ids = sorted(set(a) & set(b))
        am, astd, _ = agg(a, "slice_bias_ratio")
        bm, bstd, _ = agg(b, "slice_bias_ratio")
        d = np.array([a[i]["slice_bias_ratio"] - b[i]["slice_bias_ratio"] for i in ids])
        p, n = wilcoxon(a, b, "slice_bias_ratio")
        cdpa_worse = int((d < 0).sum())  # cdpa ratio > dpa ratio, i.e. cdpa banding worse
        p_str = "  n/a " if np.isnan(p) else f"{p:.4f}"
        print(f"  {lbl_uncond:>10s} {am:.3f}±{astd:.3f} (n={len(a)})   "
              f"{lbl_cond:>10s} {bm:.3f}±{bstd:.3f} (n={len(b)})   "
              f"paired Δ(uncond-cond)={d.mean():+.3f}   "
              f"CDPA-worse-on {cdpa_worse}/{n} volumes   p={p_str}")

    if incomplete:
        print(f"\n{'!'*130}")
        print(f"INCOMPLETE: {len(incomplete)} methods missing volumes: {incomplete}")
        print("!" * 130)


if __name__ == "__main__":
    main()
