"""ADMM + 1D-TV-along-z guidance, i.e. the DiffusionMBIR mechanism.

DiffusionMBIR (Chung et al., "Solving 3D Inverse Problems using Pre-trained 2D
Diffusion Models", CVPR 2023) augments a 2D diffusion prior with an ADMM data-
consistency step that couples adjacent axial slices via a 1D total-variation
penalty along the axial (z) axis, instead of our Resample-style plain
gradient-descent data-consistency step. Everything else -- the pretrained 2D
score network, the per-slice DDIM sampling loop -- is unchanged; only the
guidance callable passed to ``GuidedDDIMScheduler`` differs. This lets us run
this baseline against our EXISTING unconditional DPA checkpoint with no
retraining.

At each reverse diffusion step, given the current Tweedie estimate x_hat_0
(the full (D,H,W) volume, D = axial), this solves

    min_x  (1/2)||A x - y||^2 + lambda * ||D_z x||_1

via a small, fixed number of ADMM iterations, each solving the resulting
least-squares x-update with conjugate gradient (CG). D_z is the first-
difference operator along axis 0 (Neumann boundary: the last slice's
difference is 0, matching the interpretation of tv_ratio_axial in
``cbct_diffusion.utils.metrics``). x_hat_0 seeds the ADMM x-update as a warm
start -- there is no explicit "stay close to x_hat_0" penalty in the ADMM
objective itself, matching both the original paper and our own Resample-style
guidance (``astra_torch.gd_reconstruction_masked``), which likewise optimizes
pure data-consistency from an initialization rather than adding a proximity
term.

Usage
-----
    guidance = ADMMTVGuidance(
        vecs=d["vecs"], mask=d["mask"], vol_shape=d["vol_shape"],
        det_rows=d["projs"].shape[1], det_cols=d["projs"].shape[2],
        voxel_size_mm=d["vsm"], device=device,
        y=d["projs"][d["mask"]].to(device),
        rho=1.0, lam=0.01, n_admm_iters=3, n_cg_iters=5,
    )
    scheduler = GuidedDDIMScheduler(..., guidance_function=guidance)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from astra_torch.cbct import build_conebeam_projector


# ---------------------------------------------------------------------------
# 1D finite-difference operator along axis 0 (axial), Neumann boundary
# ---------------------------------------------------------------------------
def diff_z(x: torch.Tensor) -> torch.Tensor:
    """(D_z x)[i] = x[i+1] - x[i] for i < D-1, 0 at the last slice."""
    d = torch.zeros_like(x)
    d[:-1] = x[1:] - x[:-1]
    return d


def diff_z_adjoint(d: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`diff_z`: (D_z^T d)[0] = -d[0], [j] = d[j-1]-d[j], [-1] = d[-2]."""
    out = torch.zeros_like(d)
    out[:-1] -= d[:-1]
    out[1:] += d[:-1]
    return out


def soft_threshold(v: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.sign(v) * torch.clamp(v.abs() - thresh, min=0.0)


def cg_solve(hx_fn, b: torch.Tensor, x0: torch.Tensor, n_iters: int,
             tol: float = 1e-6) -> torch.Tensor:
    """Conjugate gradient for the SPD system hx_fn(x) = b, warm-started at x0."""
    x = x0.clone()
    r = b - hx_fn(x)
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(n_iters):
        if rs_old.sqrt() < tol:
            break
        hp = hx_fn(p)
        alpha = rs_old / ((p * hp).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * hp
        rs_new = (r * r).sum()
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


class ADMMTVGuidance:
    """Callable ``guidance_function(x, t) -> x_guided`` implementing DiffusionMBIR.

    Builds the cone-beam projector once (the selected sparse-view subset is
    fixed for the whole reconstruction of one volume) and reuses it across
    every diffusion timestep and every ADMM/CG iteration.
    """

    def __init__(
        self,
        vecs: np.ndarray | torch.Tensor,
        mask: Optional[Sequence] ,
        vol_shape: tuple,
        det_rows: int,
        det_cols: int,
        voxel_size_mm: float,
        y: torch.Tensor,
        device: torch.device,
        rho: float = 1.0,
        lam: float = 0.01,
        n_admm_iters: int = 3,
        n_cg_iters: int = 5,
        clamp_min: Optional[float] = 0.0,
        verbose: bool = False,
    ):
        vecs_np = vecs.detach().cpu().numpy() if torch.is_tensor(vecs) else np.asarray(vecs)
        if mask is not None:
            mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            sel_vecs = vecs_np[mask_np]
        else:
            sel_vecs = vecs_np

        self.vol_shape = tuple(vol_shape)
        self.device = device
        self.projector = build_conebeam_projector(
            self.vol_shape, det_rows, det_cols, sel_vecs, voxel_size_mm, device=device,
        )
        # y: (V, R, C) measured projections restricted to the selected views,
        # reshaped to the (1, V, R, C) the projector layer expects/produces.
        self.y = y.to(device).unsqueeze(0) if y.dim() == 3 else y.to(device)
        self.rho = rho
        self.lam = lam
        self.n_admm_iters = n_admm_iters
        self.n_cg_iters = n_cg_iters
        self.clamp_min = clamp_min
        self.verbose = verbose
        # A^T y does not depend on the ADMM/CG iterate, compute it once.
        self._Aty = self._At(self.y)

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x.unsqueeze(0).unsqueeze(0))

    def _At(self, y: torch.Tensor) -> torch.Tensor:
        # __call__ runs under torch.no_grad() (CG/ADMM are hand-rolled, not
        # autograd-optimized), but computing A^T via the projector's custom
        # backward still needs a graph built around this specific forward pass.
        with torch.enable_grad():
            x0 = torch.zeros((1, 1) + self.vol_shape, device=self.device, requires_grad=True)
            out = self.projector(x0)
            (grad_x0,) = torch.autograd.grad(out, x0, grad_outputs=y)
        return grad_x0[0, 0].detach()

    def _hx(self, v: torch.Tensor) -> torch.Tensor:
        return self._At(self._A(v)) + self.rho * diff_z_adjoint(diff_z(v))

    def __call__(self, x: torch.Tensor, t) -> torch.Tensor:
        with torch.no_grad():
            x = x.detach().to(self.device)
            z = diff_z(x)
            u = torch.zeros_like(z)
            for k in range(self.n_admm_iters):
                b = self._Aty + self.rho * diff_z_adjoint(z - u)
                x = cg_solve(self._hx, b, x, self.n_cg_iters)
                if self.clamp_min is not None:
                    x = x.clamp_min(self.clamp_min)
                dx = diff_z(x)
                z = soft_threshold(dx + u, self.lam / self.rho)
                u = u + dx - z
                if self.verbose:
                    data_term = 0.5 * ((self._A(x) - self.y) ** 2).sum().item()
                    tv_term = self.lam * dx.abs().sum().item()
                    print(f"    [ADMM-TV] iter {k+1}/{self.n_admm_iters} "
                          f"data={data_term:.4e} tv={tv_term:.4e}")
        return x
