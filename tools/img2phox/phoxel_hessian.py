"""F.12.2 — Phoxoidal extraction from voxel grid via local Hessian.

The novel CRYPSOID-distinctive piece. Standard voxel→splat extraction emits
isotropic, axis-aligned blobs. We do better:

For each occupied voxel cell, compute the 3x3 Hessian H of the density
field via central finite differences. Eigendecompose:

    H = R diag(λ1, λ2, λ3) R^T

with |λ1| ≤ |λ2| ≤ |λ3|. Then:

  - **Normal direction** = eigenvector for largest |λ| (curvature peak)
  - **Tangent plane** = eigenvectors for the two smaller |λ|
  - **Principal curvatures** (κ₁, κ₂) = the two smaller eigenvalues
    normalized by the gradient magnitude (Weingarten map proxy)
  - **Blob quaternion** = R rotated so that x-axis = normal
  - **Blob scales** anisotropic: along normal short, along tangents long

The result: blobs lie ON the implicit surface defined by the density level
set, with their major axes aligned to the surface, just like in the
catastrophe-optic Pearcey germs.

This is the bridge from "voxel grid that happens to fit images" to "splats
that natively encode surface geometry the way phoxoidal blobs are designed
to". Once these blobs hit the CRYPSOID renderer, the v32a/b/c lighting
stack should make them look right because their normals + curvatures are
finally meaningful instead of arbitrary.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


def compute_density_hessian(density: np.ndarray, cell_size: np.ndarray):
    """Compute per-cell 3x3 Hessian of density via central finite differences.

    Returns (Nx, Ny, Nz, 3, 3) Hessian and (Nx, Ny, Nz, 3) gradient.
    Boundary cells get zero (rolled-edge values).
    """
    Nx, Ny, Nz = density.shape
    h = cell_size.astype(np.float32)
    # Gradient (central diffs)
    grad = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    grad[1:-1, :, :, 0] = (density[2:, :, :] - density[:-2, :, :]) / (2 * h[0])
    grad[:, 1:-1, :, 1] = (density[:, 2:, :] - density[:, :-2, :]) / (2 * h[1])
    grad[:, :, 1:-1, 2] = (density[:, :, 2:] - density[:, :, :-2]) / (2 * h[2])
    # Hessian — symmetric, compute 6 unique components
    hess = np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.float32)
    # H_ii — second derivatives
    hess[1:-1, :, :, 0, 0] = (density[2:, :, :] - 2*density[1:-1, :, :] + density[:-2, :, :]) / (h[0]**2)
    hess[:, 1:-1, :, 1, 1] = (density[:, 2:, :] - 2*density[:, 1:-1, :] + density[:, :-2, :]) / (h[1]**2)
    hess[:, :, 1:-1, 2, 2] = (density[:, :, 2:] - 2*density[:, :, 1:-1] + density[:, :, :-2]) / (h[2]**2)
    # H_ij — mixed second derivatives via central-diff of central-diff
    # H_xy(i,j,k) = (D(i+1,j+1,k) - D(i+1,j-1,k) - D(i-1,j+1,k) + D(i-1,j-1,k)) / (4 hx hy)
    hess[1:-1, 1:-1, :, 0, 1] = (
        density[2:, 2:, :] - density[2:, :-2, :] - density[:-2, 2:, :] + density[:-2, :-2, :]
    ) / (4 * h[0] * h[1])
    hess[1:-1, :, 1:-1, 0, 2] = (
        density[2:, :, 2:] - density[2:, :, :-2] - density[:-2, :, 2:] + density[:-2, :, :-2]
    ) / (4 * h[0] * h[2])
    hess[:, 1:-1, 1:-1, 1, 2] = (
        density[:, 2:, 2:] - density[:, 2:, :-2] - density[:, :-2, 2:] + density[:, :-2, :-2]
    ) / (4 * h[1] * h[2])
    # Symmetrize
    hess[..., 1, 0] = hess[..., 0, 1]
    hess[..., 2, 0] = hess[..., 0, 2]
    hess[..., 2, 1] = hess[..., 1, 2]
    return hess, grad


def extract_phoxoidal_blobs(grid,
                              density_threshold: float = 0.5,
                              max_blobs: int = 300_000,
                              anisotropy: float = 0.4,
                              normal_align: bool = True,
                              ):
    """From a voxel grid (PhoxelGrid), extract phoxoidal blobs aligned to local
    surface geometry derived from the Hessian of the density field.

    Args:
      grid:        PhoxelGrid (from img2phox.phoxel)
      density_threshold: only cells with density > this are emitted
      max_blobs:   keep top-density blobs if more than this
      anisotropy:  ratio of normal-axis scale to tangent-axis scale.
                   0.4 means normal-axis is 40% the size of tangents (flat
                   pancake-like discs). 1.0 = isotropic. <0.5 recommended.
      normal_align: if True, use eigenvalue-derived rotation matrix.
                    if False, fall back to identity (axis-aligned) — useful
                    for A/B comparisons.

    Returns:
      BlobBundle with xyz, scales (anisotropic log-sigma), quats (per-blob R),
      opacity, sh_dc.
    """
    from .data_classes import BlobBundle

    cell = grid.cell_size
    cell_diag = float(np.linalg.norm(cell))

    # Find occupied cells
    mask = grid.density > density_threshold
    if not mask.any():
        return BlobBundle(
            xyz=np.zeros((0,3), dtype=np.float32),
            scales=np.zeros((0,3), dtype=np.float32),
            quats=np.tile([1, 0, 0, 0], (0,1)).astype(np.float32),
            opacity=np.zeros(0, dtype=np.float32),
            sh_dc=np.zeros((0,3), dtype=np.float32),
        )

    # Compute Hessian (only need where occupied — but allocate full for clarity)
    print(f'  computing density Hessian for {grid.density.shape}...')
    hess, grad = compute_density_hessian(grid.density, cell)

    idx = np.argwhere(mask)  # (N, 3)
    if len(idx) > max_blobs:
        # Keep highest density
        densities_at_mask = grid.density[idx[:, 0], idx[:, 1], idx[:, 2]]
        keep = np.argsort(densities_at_mask)[::-1][:max_blobs]
        idx = idx[keep]
    n = len(idx)
    print(f'  extracting {n:,} phoxoidal blobs...')

    # Center of each cell in world space
    xyz = grid.origin[None, :] + (idx.astype(np.float32) + 0.5) * cell[None, :]

    # Per-blob density and color
    sigma = grid.density[idx[:, 0], idx[:, 1], idx[:, 2]]
    color = grid.color[idx[:, 0], idx[:, 1], idx[:, 2]]
    opa = (1.0 - np.exp(-sigma * cell_diag)).astype(np.float32)

    # Per-blob Hessian + gradient
    H = hess[idx[:, 0], idx[:, 1], idx[:, 2]]    # (N, 3, 3)
    g = grad[idx[:, 0], idx[:, 1], idx[:, 2]]    # (N, 3)

    if normal_align:
        # Eigendecompose batched
        # eigvals: (N, 3) sorted ascending; eigvecs: (N, 3, 3) columns are vectors
        eigvals, eigvecs = np.linalg.eigh(H)   # ascending eigenvalues
        # Pick the eigenvector whose direction is most parallel to gradient
        # (the gradient points perpendicular to the level set, so the principal
        # curvature direction we want for "normal" is aligned with grad).
        g_norm = np.linalg.norm(g, axis=1, keepdims=True)
        g_unit = np.where(g_norm > 1e-8, g / np.maximum(g_norm, 1e-8), np.array([0,0,1.0], dtype=np.float32))
        # Cosine with each eigvec (columns)
        # eigvecs shape (N, 3, 3), columns indexed by axis 2
        cos = np.einsum('ni,nij->nj', g_unit, eigvecs)  # (N, 3)
        normal_axis = np.argmax(np.abs(cos), axis=1)  # which eigvec is "normal"

        # Build rotation matrix per blob: columns = (normal, tangent1, tangent2)
        normals = np.zeros((n, 3), dtype=np.float32)
        tang1 = np.zeros((n, 3), dtype=np.float32)
        tang2 = np.zeros((n, 3), dtype=np.float32)
        # Choose tangent ordering canonically (the two non-normal axes)
        for k in range(n):
            na = normal_axis[k]
            normals[k] = eigvecs[k, :, na]
            # Flip normal toward gradient
            if np.dot(normals[k], g_unit[k]) < 0:
                normals[k] = -normals[k]
            others = [a for a in (0, 1, 2) if a != na]
            tang1[k] = eigvecs[k, :, others[0]]
            tang2[k] = eigvecs[k, :, others[1]]

        # Quaternion from rotation matrix (R columns = normal, t1, t2)
        R = np.stack([normals, tang1, tang2], axis=2)  # (N, 3, 3)
        # Ensure right-handed (det = +1)
        det = np.linalg.det(R)
        flip = det < 0
        tang2[flip] = -tang2[flip]
        R[flip, :, 2] = -R[flip, :, 2]
        quats = _rmat_to_quat_batched(R)
    else:
        quats = np.zeros((n, 4), dtype=np.float32); quats[:, 0] = 1.0

    # Anisotropic scales:
    #   Normal axis: anisotropy * 0.5 * mean(cell_size)   (squashed)
    #   Tangent axes: 0.5 * mean(cell_size)               (full size)
    base = 0.5 * float(cell.mean())
    normal_scale  = anisotropy * base
    tangent_scale = base
    scales = np.tile(np.log([normal_scale, tangent_scale, tangent_scale]),
                       (n, 1)).astype(np.float32)

    return BlobBundle(
        xyz=xyz.astype(np.float32),
        scales=scales,
        quats=quats.astype(np.float32),
        opacity=opa,
        sh_dc=color.astype(np.float32),
        sh_rest=None,
        tier=None,
    )


def _rmat_to_quat_batched(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) quaternions in (w, x, y, z) order.
    Uses Shoemake's method, branch-by-trace for stability.
    """
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    pos_tr = tr > 0
    # Branch 0: trace > 0
    if pos_tr.any():
        Rp = R[pos_tr]
        s = np.sqrt(tr[pos_tr] + 1.0) * 2  # s = 4qw
        q[pos_tr, 0] = 0.25 * s
        q[pos_tr, 1] = (Rp[:, 2, 1] - Rp[:, 1, 2]) / s
        q[pos_tr, 2] = (Rp[:, 0, 2] - Rp[:, 2, 0]) / s
        q[pos_tr, 3] = (Rp[:, 1, 0] - Rp[:, 0, 1]) / s

    rest = ~pos_tr
    if rest.any():
        Rr = R[rest]
        # Pick largest diagonal element
        d0 = Rr[:, 0, 0]; d1 = Rr[:, 1, 1]; d2 = Rr[:, 2, 2]
        m0 = (d0 >= d1) & (d0 >= d2)
        m1 = (~m0) & (d1 >= d2)
        m2 = (~m0) & (~m1)
        rest_idx = np.where(rest)[0]
        if m0.any():
            ri = rest_idx[m0]; Rs = Rr[m0]
            s = np.sqrt(1.0 + Rs[:, 0, 0] - Rs[:, 1, 1] - Rs[:, 2, 2]) * 2
            q[ri, 0] = (Rs[:, 2, 1] - Rs[:, 1, 2]) / s
            q[ri, 1] = 0.25 * s
            q[ri, 2] = (Rs[:, 0, 1] + Rs[:, 1, 0]) / s
            q[ri, 3] = (Rs[:, 0, 2] + Rs[:, 2, 0]) / s
        if m1.any():
            ri = rest_idx[m1]; Rs = Rr[m1]
            s = np.sqrt(1.0 + Rs[:, 1, 1] - Rs[:, 0, 0] - Rs[:, 2, 2]) * 2
            q[ri, 0] = (Rs[:, 0, 2] - Rs[:, 2, 0]) / s
            q[ri, 1] = (Rs[:, 0, 1] + Rs[:, 1, 0]) / s
            q[ri, 2] = 0.25 * s
            q[ri, 3] = (Rs[:, 1, 2] + Rs[:, 2, 1]) / s
        if m2.any():
            ri = rest_idx[m2]; Rs = Rr[m2]
            s = np.sqrt(1.0 + Rs[:, 2, 2] - Rs[:, 0, 0] - Rs[:, 1, 1]) * 2
            q[ri, 0] = (Rs[:, 1, 0] - Rs[:, 0, 1]) / s
            q[ri, 1] = (Rs[:, 0, 2] + Rs[:, 2, 0]) / s
            q[ri, 2] = (Rs[:, 1, 2] + Rs[:, 2, 1]) / s
            q[ri, 3] = 0.25 * s

    # Normalize
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q.astype(np.float32)
