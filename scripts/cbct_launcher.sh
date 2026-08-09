#!/bin/bash --noprofile --norc
#
# cbct_launcher.sh -- RunAI entrypoint for the cbct-diffusion evaluation suite.
#
# Sets up the environment inside the container and then runs the command passed
# as arguments. Modelled on chip-project/scripts/chip_launcher.sh, but installs
# only what this package needs: astra-torch (the CUDA/ASTRA projector wrapper)
# and cbct-diffusion itself.
#
# cbct-diffusion is put on PYTHONPATH rather than pip-installed because its
# pyproject declares a build-backend that does not resolve
# ("setuptools.backends._legacy:_Backend"), so `pip install -e .` fails.
#
# Usage:
#   bash scripts/cbct_launcher.sh python -m cbct_diffusion.inference.evaluate_volume --...

set -o pipefail

if [[ $# -eq 0 ]]; then
    echo "Error: no command provided" >&2
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

export PATH="/opt/conda/bin:$PATH"
export XDG_CACHE_HOME=/myhome/.cache

echo "=== cbct_launcher: setting up ==="

# Both packages go on PYTHONPATH rather than being pip-installed:
#  - cbct-diffusion's pyproject declares a build-backend that does not resolve
#    ("setuptools.backends._legacy:_Backend"), so `pip install -e .` fails;
#  - astra-torch is pure Python (no ext_modules), so an editable install buys
#    nothing -- and it would write an egg-info directory into the *shared* NFS
#    checkout, which dozens of concurrent pods would race on.
export PYTHONPATH="/myhome/cbct-diffusion:/myhome/astra-torch${PYTHONPATH:+:$PYTHONPATH}"

# Install only what the image is actually missing. These land in the pod's own
# site-packages, so concurrent jobs do not interfere.
#
# pytest is not a test dependency here: astra_torch pulls in cupy, and torch's
# _register_fake introspection walks sys.modules and touches the lazily-loaded
# cupy.testing module, which imports pytest. Without it the run dies with a
# misleading "Failed to import diffusers.models.unets.unet_2d ... No module
# named 'pytest'". The dev server happens to have pytest installed, which is why
# this failure only reproduces inside the container image.
missing=""
for pkg in SimpleITK:SimpleITK joblib:joblib pytest:pytest lpips:lpips; do
    mod="${pkg%%:*}"; dist="${pkg##*:}"
    python -c "import ${mod}" 2>/dev/null || missing="${missing} ${dist}"
done
if [ -n "$missing" ]; then
    echo "  installing missing packages:${missing}"
    python -m pip install --quiet ${missing} || {
        echo "FATAL: dependency install failed" >&2; exit 1; }
else
    echo "  all dependencies already present in the image"
fi

# Fail fast and loudly if the container cannot see a GPU or cannot import the
# stack, rather than burning a scheduling slot on a job that dies 10 minutes in.
#
# This deliberately imports in the same order as the worker, including the
# diffusers-before-astra_torch ordering: an earlier version of this check
# imported only torch + astra_torch + metrics, passed, and then the worker died
# on `from diffusers import UNet2DModel`.
python - <<'PYCHECK' || exit 1
import sys, torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible")
print(f"  device: {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
import cbct_diffusion
import astra_torch
from cbct_diffusion.models import LatentUnet2D
from cbct_diffusion.schedulers import GuidedDDIMScheduler, DDIMPipeline
from cbct_diffusion.data import SliceCBCTDataset
from cbct_diffusion.utils.metrics import cbct_ssim_3d_gaal
import SimpleITK  # noqa: F401  (needed to read gt_volume.nii.gz / proj.nii.gz)
print("  imports OK")
PYCHECK

echo "=== cbct_launcher: running: $* ==="
"$@"
status=$?
echo "=== cbct_launcher: exit status $status ==="
exit $status
