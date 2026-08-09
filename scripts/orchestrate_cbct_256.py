#!/usr/bin/env python3
"""Orchestrate the 256^3 CBCT evaluation suite on RunAI.

Regenerates every cell of the paper's Table I from a single, consistent
configuration, adding per-axis SSIM/PSNR and the adjacent-slice TV ratio, and
persisting every reconstructed volume so future metrics need no new inference.

Why a full re-run rather than patching individual cells: the published Table I
was assembled from two W&B projects (``cbct-reconstructions`` and
``cbct-reconstructions-ddim``) at two different test-set sizes (dental/spine
DPA/CDPA rows average 15 volumes, the mu(.) rows average 20), and the
FDK-denoiser + FT rows come from a fine-tuning configuration that differs per
volume within the same dataset. No single command reproduces it.

Follows the submission pattern of /myhome/smartt/scripts/orchestrate_benchmark.py
(RunAI workspace submit, preemptible, launcher + worker), with two differences:
it can submit directly instead of only printing, and it derives the work list
from result files already on the shared volume so a relaunch is idempotent.

Examples
--------
# What is missing, and what would be submitted (no side effects):
python scripts/orchestrate_cbct_256.py --dry_run

# Submit everything that is missing:
python scripts/orchestrate_cbct_256.py --submit

# Just one dataset / one method group:
python scripts/orchestrate_cbct_256.py --datasets walnut --groups cdpa --submit

# Progress of the whole sweep:
python scripts/orchestrate_cbct_256.py --status
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
DATA_ROOT = "/mydata/sdate/shared/data/huggingface"
OUTPUT_DIR = "/mydata/sdate/shared/results/cbct256"
CKPT_DIR = "/mydata/sdate/shared/checkpoints/cbct256"

REPO = "/myhome/cbct-diffusion"
LAUNCHER = f"bash {REPO}/scripts/cbct_launcher.sh"
WORKER = "python -m cbct_diffusion.inference.evaluate_volume"

# ---------------------------------------------------------------------------
# RunAI settings
#
# GPU sizing: a measured CDPA run at 256^3 with slice_batch_size=40 peaks at
# 8.06 GB allocated (torch.max_memory_allocated). 0.15 of an 80 GB A100 = 12 GB,
# i.e. ~1.5x headroom, which is why this asks for 0.15 rather than the 0.2 the
# earlier scripts used. The classical group's peak is lower (UNet at batch 10
# plus GD on one 256^3 volume), so it shares the same request.
#
# Everything is submitted preemptible: the sdate-luisb GPU quota is 0.20, far
# below what this sweep needs, and over-quota submission requires preemptibility.
# The worker is resumable per (volume, method), so a preemption costs at most
# one posterior sample.
# ---------------------------------------------------------------------------
IMAGE = "lfbarba/sdsc_image:1.0.1"
PROJECT = "sdate-luisb"
GPU_PORTION = 0.15
RESOURCES = (
    f"--gpu-request-type portion --gpu-portion-request {GPU_PORTION} "
    "--node-type A100 --large-shm "
    "--cpu-core-request 4 --cpu-core-limit 8 "
    "--cpu-memory-request 24G --cpu-memory-limit 32G"
)
SSIM_JOBS = 4  # <= cpu-core-limit; joblib oversubscription just adds contention

NVIEWS = 20
# Posterior samples for every mu(.) row. The paper's Table I caption says 20,
# but the logged runs that actually produced it used 10 for DPA (and for
# mu(DPA) walnut) and 20 only for the conditional walnut runs -- see the
# provenance note in the module docstring. Standardising on 10 here trades
# the caption's stated number for internal consistency across every dataset
# and both DPA/CDPA, at half the compute of 20.
N_SAMPLES = 10

# ---------------------------------------------------------------------------
# Per-dataset configuration
#
# n_volumes is the size of the *test* split (dental/spine 20, walnut 5) -- the
# published DPA/CDPA rows for dental and spine only covered ids 5-19 because the
# launch script had `for id in {5..19}` with `{0..19}` commented out.
#
# ft_* are the FDK-denoiser data-consistency fine-tuning settings. These are
# recovered from the W&B config of the runs Table I actually used (the
# 2025-09-30 batch) and verified to reproduce the logged values bit-for-bit on
# walnut id 0 (29.217 dB / 0.7463 SSIM). Note they differ from the defaults now
# in reconstruct_unet.py, whose much smaller LR gives ~0.82 SSIM instead.
# ---------------------------------------------------------------------------
DATASETS = {
    "walnut": dict(
        n_volumes=5,
        unet_ckpt="Unet_Walnut_CBCT_256.pt",
        dpa_ckpt="Diffusion_Walnut_CBCT_256_ft20.pt",
        cdpa_ckpt="Diffusion_Walnut_CBCT_256_ft20_cond.pt",
        ft_epochs=[10, 10], ft_lr=[5e-3, 5e-4], ft_batch_size=20,
    ),
    "dental": dict(
        n_volumes=20,
        unet_ckpt="Unet_Dental_CBCT_256.pt",
        dpa_ckpt="Diffusion_Dental_CBCT_256_ft20.pt",
        cdpa_ckpt="Diffusion_Dental_CBCT_256_ft20_cond.pt",
        ft_epochs=[10, 10], ft_lr=[5e-3, 5e-4], ft_batch_size=20,
    ),
    "spine": dict(
        n_volumes=20,
        unet_ckpt="Unet_Spine_CBCT_256.pt",
        dpa_ckpt="Diffusion_Spine_CBCT_256_ft20.pt",
        cdpa_ckpt="Diffusion_Spine_CBCT_256_ft20_cond.pt",
        ft_epochs=[5], ft_lr=[5e-4], ft_batch_size=20,
    ),
}

# Guidance settings per method, as used by the published launch scripts:
# conditional needs far fewer data-consistency passes than unconditional
# because the FDK prior already carries most of the measurement information.
GUIDANCE = {
    "cdpa": dict(max_epochs=[5], lr=[5e-4]),
    "dpa": dict(max_epochs=[20], lr=[2e-3]),
}

# Methods each group is responsible for producing.
GROUP_METHODS = {
    "classical": ["fdk", "gd_zero", "gd_fdk", "unet", "unet_ft"],
    "dpa": ["dpa", f"mu_dpa_n{N_SAMPLES}"],
    "cdpa": ["cdpa", f"mu_cdpa_n{N_SAMPLES}"],
}

# Per-sample wall-clock for the estimate printout.
#
# These are *uncontended* figures, from the historical W&B runs (71 s
# conditional / 103 s unconditional per sample on a dedicated A100), rounded up.
# Observed on the cluster with ~10 of these jobs co-resident: ~180 s/sample for
# cdpa, i.e. 2x. A --gpu-portion-request partitions GPU *memory*, not SMs, so
# co-scheduled jobs share compute and each one slows down proportionally.
#
# The consequence for planning: the printed GPU-hours is a lower bound on
# aggregate compute, and total wall-clock is governed by how much cluster GPU
# time these preemptible low-priority jobs actually get, not by how many are
# submitted at once. Submitting more jobs spreads the same throughput; it does
# not finish sooner.
SECONDS_PER_SAMPLE = {"cdpa": 90, "dpa": 130}
SECONDS_CLASSICAL_PER_VOLUME = 150


# ---------------------------------------------------------------------------
def metrics_dir() -> Path:
    return Path(OUTPUT_DIR) / "metrics"


def is_done(dataset: str, cbct_id: int, method: str) -> bool:
    """True if this (volume, method) already has a readable result JSON."""
    p = metrics_dir() / f"{dataset}_n{NVIEWS}_id{cbct_id}_{method}.json"
    if not p.is_file():
        return False
    try:
        json.loads(p.read_text())
        return True
    except (json.JSONDecodeError, OSError):
        return False


def pending_ids(dataset: str, group: str) -> list[int]:
    """Volume ids in *dataset* that still miss at least one of *group*'s methods."""
    n = DATASETS[dataset]["n_volumes"]
    return [
        i for i in range(n)
        if not all(is_done(dataset, i, m) for m in GROUP_METHODS[group])
    ]


def chunk(xs: list[int], size: int) -> list[list[int]]:
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def job_name(dataset: str, group: str, ids: list[int]) -> str:
    span = f"{ids[0]}" if len(ids) == 1 else f"{ids[0]}-{ids[-1]}"
    # RunAI names must be DNS-1123 labels: lowercase alphanumerics and '-'.
    return f"cbct256-{group}-{dataset}-{span}".replace("_", "-").lower()


def build_command(dataset: str, group: str, ids: list[int]) -> str:
    cfg = DATASETS[dataset]
    parts = [
        WORKER,
        f"--data_path {DATA_ROOT}/{dataset}",
        f"--cbct_id {' '.join(str(i) for i in ids)}",
        f"--nviews {NVIEWS}",
        f"--method_group {group}",
        f"--output_dir {OUTPUT_DIR}",
        f"--ssim_jobs {SSIM_JOBS}",
    ]
    if group == "classical":
        parts += [
            f"--unet_checkpoint {CKPT_DIR}/{cfg['unet_ckpt']}",
            f"--ft_epochs {' '.join(str(e) for e in cfg['ft_epochs'])}",
            f"--ft_lr {' '.join(str(l) for l in cfg['ft_lr'])}",
            f"--ft_batch_size {cfg['ft_batch_size']}",
        ]
    else:
        ckpt = cfg["cdpa_ckpt"] if group == "cdpa" else cfg["dpa_ckpt"]
        g = GUIDANCE[group]
        parts += [
            f"--diffusion_checkpoint {CKPT_DIR}/{ckpt}",
            f"--n_samples {N_SAMPLES}",
            f"--guidance_max_epochs {' '.join(str(e) for e in g['max_epochs'])}",
            f"--guidance_lr {' '.join(str(l) for l in g['lr'])}",
        ]
    return " ".join(parts)


def build_submit(dataset: str, group: str, ids: list[int]) -> tuple[str, str]:
    name = job_name(dataset, group, ids)
    # `training submit`, not `workspace submit`: Workspaces are interactive and
    # default to non-preemptible, so an over-quota Workspace sits in Pending
    # forever (observed: Preemptible=No even with the flag below). Trainings are
    # the preemptible workload type, which is what every earlier CBCT job used
    # and what over-quota scheduling requires -- the sdate-luisb GPU quota is
    # 0.20, and this sweep needs far more than that in parallel.
    cmd = (
        f"runai training submit {name} "
        f"-i {IMAGE} -p {PROJECT} {RESOURCES} "
        f"--preemptibility preemptible "
        f"--command -- {LAUNCHER} {build_command(dataset, group, ids)}"
    )
    return name, cmd


# ---------------------------------------------------------------------------
def print_status(datasets: list[str], groups: list[str]) -> None:
    print(f"Results: {OUTPUT_DIR}/metrics\n")
    header = f"{'dataset':9s} {'group':10s} {'done':>9s}  {'volumes still pending'}"
    print(header)
    print("-" * len(header))
    grand_done = grand_total = 0
    for ds in datasets:
        for g in groups:
            n = DATASETS[ds]["n_volumes"]
            pend = pending_ids(ds, g)
            done = n - len(pend)
            grand_done += done
            grand_total += n
            shown = str(pend) if len(pend) <= 12 else f"{pend[:12]}... (+{len(pend)-12})"
            print(f"{ds:9s} {g:10s} {done:4d}/{n:<4d}  {shown if pend else '-'}")
    print("-" * len(header))
    pct = 100.0 * grand_done / grand_total if grand_total else 0.0
    print(f"{'TOTAL':20s} {grand_done:4d}/{grand_total:<4d}  ({pct:.1f}% of volume-groups complete)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestrate the 256^3 CBCT suite on RunAI")
    ap.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    ap.add_argument("--groups", nargs="+", default=["classical", "cdpa", "dpa"],
                    choices=["classical", "cdpa", "dpa"])
    ap.add_argument("--volumes_per_job", type=int, default=None,
                    help="volumes per pod (default: all of a dataset for 'classical', "
                         "1 for the diffusion groups so a preemption costs little)")
    ap.add_argument("--submit", action="store_true", help="actually submit to RunAI")
    ap.add_argument("--dry_run", action="store_true", help="print the commands only")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--max_jobs", type=int, default=None, help="cap the number submitted")
    args = ap.parse_args()

    if args.status:
        print_status(args.datasets, args.groups)
        return

    jobs: list[tuple[str, str, list[int]]] = []
    for ds in args.datasets:
        for g in args.groups:
            pend = pending_ids(ds, g)
            if not pend:
                continue
            per = args.volumes_per_job or (len(pend) if g == "classical" else 1)
            jobs.extend((ds, g, ids) for ids in chunk(pend, per))

    if not jobs:
        print("Nothing pending — the suite is complete.")
        print_status(args.datasets, args.groups)
        return

    if args.max_jobs:
        jobs = jobs[:args.max_jobs]

    # Resource estimate
    gpu_seconds = 0.0
    for ds, g, ids in jobs:
        if g == "classical":
            gpu_seconds += SECONDS_CLASSICAL_PER_VOLUME * len(ids)
        else:
            gpu_seconds += SECONDS_PER_SAMPLE[g] * N_SAMPLES * len(ids)
    # Storage: count each (dataset, id) once even though it appears in up to
    # three groups. ~440 MB per volume-id for the full set of persisted
    # reconstructions (fdk, unet, unet_ft, dpa, mu_dpa, std_dpa, cdpa, mu_cdpa,
    # std_cdpa at ~45-62 MB each compressed).
    unique_volumes = {(ds, i) for ds, _, ids in jobs for i in ids}
    print(f"# {len(jobs)} job(s) to submit")
    print(f"# estimated GPU time: {gpu_seconds/3600:.1f} GPU-hours "
          f"({GPU_PORTION} GPU each, so ~{gpu_seconds/3600*GPU_PORTION:.1f} "
          f"whole-GPU-hours of quota-equivalent usage)")
    print(f"# volumes persisted to {OUTPUT_DIR}/volumes "
          f"(~440 MB per volume-id x {len(unique_volumes)} volumes "
          f"= ~{0.44*len(unique_volumes):.1f} GB)")
    print()

    if not (args.submit or args.dry_run):
        print("Neither --submit nor --dry_run given; showing commands only "
              "(same as --dry_run).\n")

    submitted, failed = [], []
    for ds, g, ids in jobs:
        name, cmd = build_submit(ds, g, ids)
        if args.submit:
            print(f"--> {name}", flush=True)
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                submitted.append(name)
            else:
                failed.append((name, (r.stderr or r.stdout).strip().splitlines()[-1:]))
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
        print(f"\nMonitor with:  runai workload list -p {PROJECT}")
        print(f"Progress with: python {__file__} --status")


if __name__ == "__main__":
    main()
