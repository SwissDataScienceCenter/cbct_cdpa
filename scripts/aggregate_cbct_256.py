#!/usr/bin/env python3
"""Aggregate the 256^3 suite into per-dataset tables and LaTeX for the paper.

Reads the per-(volume, method) result JSONs written by
``cbct_diffusion.inference.evaluate_volume`` and reports, for every method and
dataset, mean +/- std across test volumes of:

- PSNR (volume) and SSIM (3-axis mean) -- the two columns of the current Table I
- SSIM per orthogonal plane, with the axial-vs-off-axis gap
- the adjacent-slice TV ratio, with the axial-vs-in-plane excess

and runs a paired Wilcoxon signed-rank test of each method against a chosen
reference, since the headline comparisons are over 5-20 volumes.

Usage
-----
python scripts/aggregate_cbct_256.py                    # all datasets, text
python scripts/aggregate_cbct_256.py --latex            # + LaTeX for Table I
python scripts/aggregate_cbct_256.py --datasets walnut  # one dataset
python scripts/aggregate_cbct_256.py --require_complete # refuse partial results
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTPUT_DIR = "/mydata/sdate/shared/results/cbct256"
NVIEWS = 20
N_SAMPLES = 10  # must match orchestrate_cbct_256.py's N_SAMPLES

# Display order and labels, matching the paper's Table I row order.
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
EXPECTED_VOLUMES = {"walnut": 5, "dental": 20, "spine": 20}

FNAME_RE = re.compile(r"^(?P<ds>\w+?)_n(?P<nv>\d+)_id(?P<id>\d+)_(?P<method>.+)$")


def load(datasets: list[str]) -> dict:
    """-> {dataset: {method: {cbct_id: metrics_dict}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(f"{OUTPUT_DIR}/metrics/*.json")):
        m = FNAME_RE.match(Path(f).stem)
        if not m or m["ds"] not in datasets or int(m["nv"]) != NVIEWS:
            continue
        try:
            d = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            print(f"  ! skipping unreadable {Path(f).name}")
            continue
        out[m["ds"]][m["method"]][int(m["id"])] = d
    return out


def agg(per_id: dict, key: str) -> tuple[float, float, int]:
    v = np.array([per_id[i][key] for i in sorted(per_id) if key in per_id[i]], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def wilcoxon(a: dict, b: dict, key: str) -> tuple[float, int]:
    """Paired Wilcoxon signed-rank p-value over the volumes both methods share."""
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


def paired_diff(a: dict, b: dict, key: str) -> np.ndarray:
    """Per-volume (a[key] - b[key]) over the volumes both methods share, in id order."""
    ids = sorted(set(a) & set(b))
    return np.array([a[i][key] - b[i][key] for i in ids], dtype=float)


def print_axis_gap_comparison(data: dict, datasets: list[str]) -> None:
    """DPA vs CDPA, paired per volume: does conditioning shrink the axial gap?

    The dataset-level mean gap (printed in the main table) is small for both
    methods and easy to dismiss. Pairing per volume answers a sharper question:
    for the *same* volume, is DPA's axial-vs-off-axis SSIM gap consistently
    larger than CDPA's? A method-level mean can hide a large, consistent
    per-volume effect if the gap's sign also varies across volumes -- pairing
    removes that confound.

    Reports both SSIM_axial_gap (axial - offaxis) and PSNR_axial_gap for the
    same pairing, since they can disagree in sign (see walnut CDPA, where SSIM
    barely favours the axial plane but PSNR favours the off-axis planes) --
    that disagreement is itself informative about what SSIM's local-window
    formulation is and is not sensitive to.
    """
    pairs = [("dpa", "cdpa", "DPA", "CDPA"),
             (f"mu_dpa_n{N_SAMPLES}", f"mu_cdpa_n{N_SAMPLES}", r"$\mu$(DPA)", r"$\mu$(CDPA)")]
    print(f"\n{'='*104}\n### Per-axis SSIM/PSNR gap: DPA vs CDPA, paired per volume\n{'='*104}")
    print("gap := axial - offaxis (i.e. mean of coronal, sagittal). Positive means the plane")
    print("the 2D model runs in scores better than the two it doesn't -- the axial advantage")
    print("a slice-by-slice method should NOT have if it were truly 3D-consistent.\n")

    for ds in datasets:
        if ds not in data:
            continue
        print(f"--- {ds} ---")
        for m_uncond, m_cond, lbl_uncond, lbl_cond in pairs:
            a, b = data[ds].get(m_uncond), data[ds].get(m_cond)
            if not a or not b:
                print(f"  {lbl_uncond} vs {lbl_cond}: missing data")
                continue
            ids = sorted(set(a) & set(b))
            n = len(ids)

            for metric, name in [("ssim_axial_gap", "SSIM gap"), ("psnr_axial_gap", "PSNR gap")]:
                am, astd, _ = agg(a, metric)
                bm, bstd, _ = agg(b, metric)
                d = paired_diff(a, b, metric)  # uncond - cond, per volume, paired
                dstd = d.std(ddof=1) if n > 1 else 0.0
                p, _ = wilcoxon(a, b, metric)
                pct = 100.0 * d.mean() / am if abs(am) > 1e-9 else float("nan")
                consistent = int((d > 0).sum())  # volumes where uncond gap > cond gap
                p_str = "  n/a " if np.isnan(p) else f"{p:.4f}" + ("*" if p < 0.05 else " ")
                print(f"  {name:9s} {lbl_uncond:>10s} {am:+.4f}±{astd:.4f}   "
                      f"{lbl_cond:>10s} {bm:+.4f}±{bstd:.4f}   "
                      f"paired Δ={d.mean():+.4f}±{dstd:.4f} "
                      f"({pct:+5.0f}% reduction, {consistent}/{n} volumes)  p={p_str}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dental", "spine", "walnut"])
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--axis_gap", action="store_true",
                    help="paired DPA-vs-CDPA (and mu variants) axial-gap comparison")
    ap.add_argument("--reference", default=f"mu_cdpa_n{N_SAMPLES}",
                    help="method the Wilcoxon tests compare against")
    ap.add_argument("--require_complete", action="store_true",
                    help="exit non-zero if any (dataset, method) is missing volumes")
    args = ap.parse_args()

    data = load(args.datasets)
    if not data:
        raise SystemExit(f"No results found under {OUTPUT_DIR}/metrics")

    incomplete = []
    for ds in args.datasets:
        if ds not in data:
            print(f"\n### {ds}: no results yet")
            continue
        exp = EXPECTED_VOLUMES[ds]
        print(f"\n{'='*104}\n### {ds}  (expected n={exp} test volumes, {NVIEWS} views)\n{'='*104}")
        hdr = (f"{'method':26s} {'n':>3s} {'PSNR':>13s} {'SSIM':>14s} "
               f"{'SSIM_ax':>8s} {'SSIM_cor':>9s} {'SSIM_sag':>9s} {'gap':>8s} "
               f"{'tv_ax':>7s} {'tv_inpl':>8s} {'excess':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for meth, label in METHOD_ORDER:
            per_id = data[ds].get(meth)
            if not per_id:
                incomplete.append((ds, meth, 0, exp))
                print(f"{label:26s} {'--':>3s}  (not yet available)")
                continue
            n = len(per_id)
            if n != exp:
                incomplete.append((ds, meth, n, exp))
            pm, ps, _ = agg(per_id, "psnr")
            sm, ss, _ = agg(per_id, "ssim")
            f = lambda k: agg(per_id, k)[0]
            flag = "" if n == exp else f"  <-- {n}/{exp}"
            print(f"{label:26s} {n:3d} {pm:7.2f}±{ps:4.2f} {sm:8.4f}±{ss:.4f} "
                  f"{f('ssim_axial'):8.4f} {f('ssim_coronal'):9.4f} {f('ssim_sagittal'):9.4f} "
                  f"{f('ssim_axial_gap'):+8.4f} {f('tv_ratio_axial'):7.3f} "
                  f"{f('tv_ratio_inplane'):8.3f} {f('tv_ratio_excess'):+8.3f}{flag}")

        ref = data[ds].get(args.reference)
        if ref:
            print(f"\n  Paired Wilcoxon signed-rank vs {args.reference} (PSNR / SSIM):")
            for meth, label in METHOD_ORDER:
                if meth == args.reference or meth not in data[ds]:
                    continue
                pp, npair = wilcoxon(data[ds][meth], ref, "psnr")
                sp, _ = wilcoxon(data[ds][meth], ref, "ssim")
                def fmt(p):
                    return "  n/a " if np.isnan(p) else f"{p:.4f}" + ("*" if p < 0.05 else " ")
                print(f"    {label:26s} n={npair:2d}  p_psnr={fmt(pp)}  p_ssim={fmt(sp)}")

    if args.axis_gap:
        print_axis_gap_comparison(data, args.datasets)

    if args.latex:
        print(f"\n\n{'='*104}\n### LaTeX (Table I body, three SSIM columns per dataset)\n{'='*104}")
        for ds in args.datasets:
            if ds not in data:
                continue
            print(f"% ---- {ds} ----")
            for meth, label in METHOD_ORDER:
                per_id = data[ds].get(meth)
                if not per_id:
                    continue
                pm, ps, n = agg(per_id, "psnr")
                sm, ss, _ = agg(per_id, "ssim")
                am, as_, _ = agg(per_id, "ssim_axial")
                om, os_, _ = agg(per_id, "ssim_offaxis")
                print(f"{label} & {pm:.2f}±{ps:.2f} & {sm:.3f}±{ss:.3f} & "
                      f"{am:.3f}±{as_:.3f} & {om:.3f}±{os_:.3f} \\\\  % n={n}")
            print()

    if incomplete:
        print(f"\n{'!'*104}")
        print(f"INCOMPLETE: {len(incomplete)} (dataset, method) cells are missing volumes:")
        for ds, meth, n, exp in incomplete:
            print(f"  {ds:8s} {meth:22s} {n}/{exp}")
        print("Do not put these numbers in the paper yet. Check progress with:")
        print("  python scripts/orchestrate_cbct_256.py --status")
        print("!" * 104)
        if args.require_complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
