#!/usr/bin/env python3
"""Orchestrate the 501^3 / 30-view walnut suite on RunAI.

Checks whether conditioning's inter-slice-consistency benefit (established at
256^3/20-views on dental+spine, see scripts/aggregate_cbct_256.py --axis_gap)
holds up on REAL CBCT measurements once the view count matches where the paper
already claims things work well (N=30, the sparsity level fine-tuned for direct
comparison against S-STAR Net). 20-view walnut is acknowledged in the paper
itself as under-determined for every method; 30-view is the fairer real-data
test.

Produces, for 5 test walnuts (Walnut1..Walnut5 under Test/, raw high-scan TIFF
data, orbit_id=1 as in the two release inference scripts and the archived
W&B config for the published walnut_metrics table):
  classical  FDK, GD_zero, GD_FDK, FDK-denoiser, FDK-denoiser+FT  (one job/volume set)
  dpa        unconditional diffusion: single sample + mu(DPA)_n10
  cdpa       conditional diffusion:   single sample + mu(CDPA)_n10

Two real bugs in the released repo were found and fixed to make this path work
at all (see cbct_diffusion/data/walnut512.py):
  1. Walnut512 was missing normalize()/denormalize(), which
     reconstruct_diffusion.py's own --high_resolution path calls.
  2. The external (ground-truth) volume loader used the raw detector row count
     (972) instead of 501 (the FDK grid's slice count) for how many archived
     GT slices to load, so GT and every reconstruction had mismatched shapes.
     Fixed to hardcode 501, matching the original chip-project implementation.

Resource sizing (measured via single-sample profiling probes on this cluster,
not guessed):
  - DPA:  peak 9.19 GB allocated, slice_batch_size=8, guidance_max_epochs=20 @ lr=2e-3
  - CDPA: peak 10.70 GB allocated, slice_batch_size=8, guidance_max_epochs=5 @ lr=5e-4
    (confirmed via cbct501-probe-cdpa-r2)
  - classical: completed inside a 0.3-portion (24 GB) container without issue;
    its ops (single-batch UNet forward + one GD fine-tune stage) are lighter
    than diffusion sampling, so it is given a smaller portion.

Timing (measured on THIS run, cluster at 14/14 GPUs allocated -- i.e. already
contended, not a clean baseline):
  - classical, all 5 methods, 1 volume: 787 s
  - DPA, 1 sample:  993 s
  - CDPA, 1 sample: 808 s (confirmed via cbct501-probe-cdpa-r2; slower guidance
    schedule than DPA's epoch count would suggest, but still cheaper overall --
    likely dominated by the extra FDK-conditioning channel's I/O/copy cost, not
    epoch count)

Usage
-----
python scripts/orchestrate_cbct501_30views.py --dry_run
python scripts/orchestrate_cbct501_30views.py --submit
python scripts/orchestrate_cbct501_30views.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "/mydata/sdate/shared/data/walnut"          # raw TIFF root (Train/, Test/)
OUTPUT_DIR = "/mydata/sdate/shared/results/cbct501"       # final results (NOT the *_probe dir)
CKPT_DIR = "/mydata/sdate/shared/checkpoints/cbct501"

REPO = "/myhome/cbct-diffusion"
LAUNCHER = f"bash {REPO}/scripts/cbct_launcher.sh"
WORKER = "python -m cbct_diffusion.inference.evaluate_volume"

DATASET_LABEL = "walnut501"   # distinct from "walnut" (the 256^3 sweep) in filenames
NVIEWS = 30
N_VOLUMES = 5                 # Test/Walnut1..Walnut5
N_SAMPLES = 10                # matches the archived W&B config (diffusion_runs=10)

DPA_CKPT = "Diffusion_Walnut_CBCT_501_30.pt"
CDPA_CKPT = "Diffusion_Walnut_CBCT_501_cond_30.pt"
UNET_CKPT = "Unet_Walnut_CBCT_501_30.pt"

# Guidance settings recovered from the archived W&B config for the published
# walnut_metrics table (CDPA) / carried over unchanged from the 256^3 DPA
# schedule (DPA, no archived 501/N=30 unconditional run exists to calibrate
# against -- this sweep is the first time it has been run).
GUIDANCE = {
    "cdpa": dict(max_epochs=[5], lr=[5e-4]),
    "dpa": dict(max_epochs=[20], lr=[2e-3]),
}
FT_EPOCHS, FT_LR, FT_BATCH = [10, 10], [5e-3, 5e-4], 20
GD_EPOCHS, GD_LR, GD_BATCH = [100, 50, 10], [5e-3, 5e-4, 1e-4], 30

# ---------------------------------------------------------------------------
# RunAI settings
# ---------------------------------------------------------------------------
IMAGE = "lfbarba/sdsc_image:1.0.1"
PROJECT = "sdate-luisb"
GPU_PORTION_DIFFUSION = 0.2   # 16 GB vs the larger of DPA (9.19 GB) / CDPA (10.70 GB) measured peaks (~1.5x headroom)
GPU_PORTION_CLASSICAL = 0.15  # 12 GB; classical completed fine inside 24 GB, well under this comfortably
SLICE_BATCH_SIZE = 8          # profiled value at 512x512-padded slices; do not raise without re-profiling memory
UNET_BATCH_SIZE = 8

RESOURCES_DIFFUSION = (
    f"--gpu-request-type portion --gpu-portion-request {GPU_PORTION_DIFFUSION} "
    "--node-type A100 --large-shm "
    "--cpu-core-request 4 --cpu-core-limit 8 "
    "--cpu-memory-request 24G --cpu-memory-limit 32G"
)
RESOURCES_CLASSICAL = (
    f"--gpu-request-type portion --gpu-portion-request {GPU_PORTION_CLASSICAL} "
    "--node-type A100 --large-shm "
    "--cpu-core-request 4 --cpu-core-limit 8 "
    "--cpu-memory-request 24G --cpu-memory-limit 32G"
)
SSIM_JOBS = 4

GROUP_METHODS = {
    "classical": ["fdk", "gd_zero", "gd_fdk", "unet", "unet_ft"],
    "dpa": ["dpa", f"mu_dpa_n{N_SAMPLES}"],
    "cdpa": ["cdpa", f"mu_cdpa_n{N_SAMPLES}"],
}

# Measured per-unit wall-clock for the estimate printout -- see module
# docstring; these are CONTENDED figures from the actual profiling runs on
# this cluster, not clean single-tenant numbers, so they already include
# whatever slowdown a co-resident cluster causes today. Real throughput will
# still vary with how many other jobs are scheduled at the time.
SECONDS_PER_SAMPLE = {"cdpa": 808, "dpa": 993}  # both confirmed via profiling probes
SECONDS_CLASSICAL_PER_VOLUME = 787


# ---------------------------------------------------------------------------
def metrics_dir() -> Path:
    return Path(OUTPUT_DIR) / "metrics"


def is_done(cbct_id: int, method: str) -> bool:
    p = metrics_dir() / f"{DATASET_LABEL}_n{NVIEWS}_id{cbct_id}_{method}.json"
    if not p.is_file():
        return False
    try:
        json.loads(p.read_text())
        return True
    except (json.JSONDecodeError, OSError):
        return False


def pending_ids(group: str) -> list[int]:
    return [i for i in range(N_VOLUMES) if not all(is_done(i, m) for m in GROUP_METHODS[group])]


def chunk(xs: list[int], size: int) -> list[list[int]]:
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def job_name(group: str, ids: list[int]) -> str:
    span = f"{ids[0]}" if len(ids) == 1 else f"{ids[0]}-{ids[-1]}"
    return f"cbct501-{group}-{span}".lower()


def build_command(group: str, ids: list[int]) -> str:
    parts = [
        WORKER,
        f"--data_path {DATA_ROOT}",
        f"--cbct_id {' '.join(str(i) for i in ids)}",
        f"--nviews {NVIEWS}",
        "--high_resolution",
        f"--dataset_label {DATASET_LABEL}",
        f"--method_group {group}",
        f"--output_dir {OUTPUT_DIR}",
        f"--ssim_jobs {SSIM_JOBS}",
    ]
    if group == "classical":
        parts += [
            f"--unet_checkpoint {CKPT_DIR}/{UNET_CKPT}",
            f"--unet_batch_size {UNET_BATCH_SIZE}",
            f"--ft_epochs {' '.join(str(e) for e in FT_EPOCHS)}",
            f"--ft_lr {' '.join(str(l) for l in FT_LR)}",
            f"--ft_batch_size {FT_BATCH}",
            f"--gd_epochs {' '.join(str(e) for e in GD_EPOCHS)}",
            f"--gd_lr {' '.join(str(l) for l in GD_LR)}",
            f"--gd_batch_size {GD_BATCH}",
        ]
    else:
        ckpt = CDPA_CKPT if group == "cdpa" else DPA_CKPT
        g = GUIDANCE[group]
        parts += [
            f"--diffusion_checkpoint {CKPT_DIR}/{ckpt}",
            f"--n_samples {N_SAMPLES}",
            f"--slice_batch_size {SLICE_BATCH_SIZE}",
            f"--guidance_max_epochs {' '.join(str(e) for e in g['max_epochs'])}",
            f"--guidance_lr {' '.join(str(l) for l in g['lr'])}",
        ]
    return " ".join(parts)


def build_submit(group: str, ids: list[int]) -> tuple[str, str]:
    name = job_name(group, ids)
    resources = RESOURCES_CLASSICAL if group == "classical" else RESOURCES_DIFFUSION
    # `training submit`, not `workspace submit` -- Workspaces default to
    # non-preemptible and sit in Pending forever over quota (see the 256^3
    # sweep's history for the same lesson). This project's GPU quota (0.20) is
    # far below what this sweep needs at any concurrency, so preemptible
    # Training submission is required, not optional.
    cmd = (
        f"runai training submit {name} "
        f"-i {IMAGE} -p {PROJECT} {resources} "
        f"--preemptibility preemptible "
        f"--command -- {LAUNCHER} {build_command(group, ids)}"
    )
    return name, cmd


# ---------------------------------------------------------------------------
def print_status(groups: list[str]) -> None:
    print(f"Results: {OUTPUT_DIR}/metrics\n")
    header = f"{'group':10s} {'done':>9s}  {'volumes still pending'}"
    print(header)
    print("-" * len(header))
    grand_done = grand_total = 0
    for g in groups:
        pend = pending_ids(g)
        done = N_VOLUMES - len(pend)
        grand_done += done
        grand_total += N_VOLUMES
        print(f"{g:10s} {done:4d}/{N_VOLUMES:<4d}  {pend if pend else '-'}")
    print("-" * len(header))
    pct = 100.0 * grand_done / grand_total if grand_total else 0.0
    print(f"{'TOTAL':10s} {grand_done:4d}/{grand_total:<4d}  ({pct:.1f}% complete)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrate the 501^3/30-view walnut suite on RunAI")
    ap.add_argument("--groups", nargs="+", default=["classical", "cdpa", "dpa"],
                    choices=["classical", "cdpa", "dpa"])
    ap.add_argument("--volumes_per_job", type=int, default=None,
                    help="default: all 5 for classical, 1 for cdpa/dpa (a preemption "
                         "then costs at most one posterior sample, not a whole volume)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        print_status(args.groups)
        return

    jobs: list[tuple[str, list[int]]] = []
    for g in args.groups:
        pend = pending_ids(g)
        if not pend:
            continue
        per = args.volumes_per_job or (len(pend) if g == "classical" else 1)
        jobs.extend((g, ids) for ids in chunk(pend, per))

    if not jobs:
        print("Nothing pending — the suite is complete.")
        print_status(args.groups)
        return

    gpu_seconds = sum(
        SECONDS_CLASSICAL_PER_VOLUME * len(ids) if g == "classical"
        else SECONDS_PER_SAMPLE[g] * N_SAMPLES * len(ids)
        for g, ids in jobs
    )
    print(f"# {len(jobs)} job(s) to submit")
    print(f"# estimated GPU time: {gpu_seconds/3600:.1f} GPU-hours at the measured "
          f"(already-contended) per-unit times above")
    print(f"# volumes persisted to {OUTPUT_DIR}/volumes "
          f"(~600 MB/volume-id at 501^3 x {len({i for _, ids in jobs for i in ids})} volumes)")
    print()

    submitted, failed = [], []
    for g, ids in jobs:
        name, cmd = build_submit(g, ids)
        if args.submit:
            print(f"--> {name}", flush=True)
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                submitted.append(name)
            else:
                failed.append((name, (r.stderr or r.stdout).strip()[:300]))
                print(f"    FAILED: {(r.stderr or r.stdout).strip()[:300]}")
        else:
            print(cmd)
            print()

    if args.submit:
        print(f"\nSubmitted {len(submitted)}/{len(jobs)}")
        if failed:
            print(f"Failed {len(failed)}:")
            for n, err in failed:
                print(f"  {n}: {err}")
            sys.exit(1)
        print(f"\nMonitor with:  runai training list -p {PROJECT}")
        print(f"Progress with: python {__file__} --status")


if __name__ == "__main__":
    main()
