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
