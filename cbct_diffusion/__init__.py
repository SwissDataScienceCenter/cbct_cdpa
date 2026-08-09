"""
CBCT Diffusion: Diffusion-based CBCT Reconstruction Library.

This library provides UNet and diffusion-based methods for sparse-view
Cone-Beam Computed Tomography (CBCT) reconstruction. It includes:

- Slice-wise 2D UNet for CBCT artifact removal
- DDPM/DDIM diffusion models with sinogram-guided reconstruction
- FDK and gradient-descent baselines
- Training and inference pipelines
- Support for Walnut, Dental, and Spine CBCT datasets
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Import-order guard: diffusers must be imported before astra_torch.
#
# astra_torch's import triggers an early registration in torch's
# ``_c10d_functional`` namespace. If diffusers is imported afterwards, its
# ``torch._dynamo`` import re-runs ``torch.library.register_autograd`` for
# ``wait_tensor`` and torch raises:
#
#   RuntimeError: This is not allowed since there's already a kernel registered
#   from python overriding wait_tensor's behavior for Autograd dispatch key and
#   _c10d_functional namespace.
#
# Reproduced on torch 2.12.0+cu130 / diffusers 0.38.0. Importing diffusers first
# avoids it, so we force that here: any `import cbct_diffusion...` gets a safe
# ordering regardless of what the calling script imports first. Scripts that
# import astra_torch at the very top (before any cbct_diffusion module) must
# still import this package first -- see inference/evaluate_volume.py.
#
# Note this must pull the *concrete* submodule, not just the top-level package:
# diffusers lazy-loads, so a bare ``import diffusers`` defers
# ``diffusers.models.unets.unet_2d`` until first attribute access -- which then
# happens after astra_torch, defeating the guard. Importing UNet2DModel forces
# the real work to happen here.
#
# A second, related failure this ordering avoids: astra_torch pulls in cupy, and
# torch's ``_register_fake`` introspection walks sys.modules, touching the lazy
# ``cupy.testing`` module, which imports pytest. In an image without pytest that
# surfaces as a confusing "Failed to import diffusers.models.unets.unet_2d ...
# No module named 'pytest'". Importing diffusers first sidesteps it; installing
# pytest fixes it outright (see scripts/cbct_launcher.sh).
#
# It is wrapped so that environments without diffusers (e.g. reading result
# JSONs offline with only numpy) can still import the package.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    from diffusers import UNet2DModel as _UNet2DModel  # noqa: F401
except ImportError:  # pragma: no cover
    pass
