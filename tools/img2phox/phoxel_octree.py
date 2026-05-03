"""F.12.3 — Two-level sparse octree for phoxel reconstruction.

Why this exists:
  A uniform 96^3 grid wastes 80%+ of its 884k cells on empty air. Memory and
  per-iter compute scale with cell count, but only the surface-adjacent cells
  carry useful gradient. A two-level octree spends fine-grained cells only
  where they're needed.

Layout:
  - Coarse grid:   (Cx, Cy, Cz) cells, each holds (density, RGB)
  - Subdiv map:    (Cx, Cy, Cz) int32, -1 if coarse leaf, else fine_idx
  - Fine cells:    (M, F, F, F)  — M is number of subdivided coarse cells,
                                   F is fine-cells-per-axis-per-coarse-cell
  - Effective res: (Cx*F, Cy*F, Cz*F) at the boundary, Cx,Cy,Cz elsewhere.

For a 32^3 root with F=4 subdivision: effective 128^3 detail at the surface,
but only the surface-adjacent ~10-20% of root cells get fine arrays — so total
memory is roughly (32^3 + 0.15 * 32^3 * 4^3) ~ 130k cells vs uniform 128^3 = 2M.
A ~16× memory savings.

Both forward and backward use the same descent logic in Numba:
  - Compute world->coarse coords
  - If subdiv[ci,cj,ck] == -1, sample (trilinear) from coarse arrays
  - Else descend: compute world->fine coords inside the coarse cell, sample
    (trilinear) from fine[m, ...]
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from numba import njit, prange


# =================================================================
#                         Octree representation
# =================================================================

@dataclass
class PhoxelOctree:
    """Two-level sparse octree.

    Coarse arrays: (Cx, Cy, Cz, ...) — density / color at root resolution.
    Subdiv map:    (Cx, Cy, Cz) int32 — -1 means coarse leaf, else fine index.
    Fine arrays:   (M, F, F, F, ...) — only allocated for subdivided cells.
    """
    origin:      np.ndarray   # (3,)  world-space corner of cell (0,0,0)
    size:        np.ndarray   # (3,)  total grid extent in world units
    res:         np.ndarray   # (3,) int — coarse cells per axis
    fine_factor: int          # subdivision per coarse cell (typically 4)

    coarse_density: np.ndarray   # (Cx, Cy, Cz) float32
    coarse_color:   np.ndarray   # (Cx, Cy, Cz, 3) float32
    subdiv:         np.ndarray   # (Cx, Cy, Cz) int32 — -1 or fine index

    fine_density: np.ndarray     # (M, F, F, F) float32
    fine_color:   np.ndarray     # (M, F, F, F, 3) float32

    @classmethod
    def from_bounds(cls, lo, hi, coarse_res: int = 32, fine_factor: int = 4,
                     init_density: float = 0.05, init_color: float = 0.5):
        lo = np.asarray(lo, dtype=np.float32)
        hi = np.asarray(hi, dtype=np.float32)
        size = hi - lo
        res = np.array([coarse_res]*3, dtype=np.int32)
        cd = np.full(tuple(res), init_density, dtype=np.float32)
        cc = np.full((res[0], res[1], res[2], 3), init_color, dtype=np.float32)
        sd = np.full(tuple(res), -1, dtype=np.int32)
        # M starts at 0
        fd = np.zeros((0, fine_factor, fine_factor, fine_factor), dtype=np.float32)
        fc = np.zeros((0, fine_factor, fine_factor, fine_factor, 3), dtype=np.float32)
        return cls(origin=lo, size=size, res=res, fine_factor=fine_factor,
                    coarse_density=cd, coarse_color=cc, subdiv=sd,
                    fine_density=fd, fine_color=fc)

    @property
    def coarse_cell_size(self) -> np.ndarray:
        return self.size / self.res.astype(np.float32)

    @property
    def fine_cell_size(self) -> np.ndarray:
        return self.coarse_cell_size / self.fine_factor

    @property
    def n_subdivided(self) -> int:
        return self.fine_density.shape[0]

    @property
    def total_cells(self) -> int:
        nc = int(np.prod(self.res))
        nf = self.n_subdivided * (self.fine_factor ** 3)
        # Subdivided coarse cells are "replaced" by fine, so:
        return (nc - self.n_subdivided) + nf

    # ---- Subdivision ----
    def subdivide(self, mask: np.ndarray):
        """Subdivide all coarse cells where mask[i,j,k] is True AND not already subdivided.

        Initializes fine cells from parent's coarse value (constant block),
        so the operation is value-preserving. The optimizer can then resolve
        sub-cell detail.
        """
        Cx, Cy, Cz = self.res
        new_idx = np.argwhere(mask & (self.subdiv == -1))
        if len(new_idx) == 0:
            return 0
        F = self.fine_factor
        # Allocate new fine blocks
        old_M = self.fine_density.shape[0]
        new_M = old_M + len(new_idx)
        new_fd = np.zeros((new_M, F, F, F), dtype=np.float32)
        new_fc = np.zeros((new_M, F, F, F, 3), dtype=np.float32)
        new_fd[:old_M] = self.fine_density
        new_fc[:old_M] = self.fine_color
        # Init from parent
        for k, (i, j, kk) in enumerate(new_idx):
            m = old_M + k
            new_fd[m, :, :, :]    = self.coarse_density[i, j, kk]
            new_fc[m, :, :, :, :] = self.coarse_color[i, j, kk]
            self.subdiv[i, j, kk] = m
        self.fine_density = new_fd
        self.fine_color   = new_fc
        return len(new_idx)


# =================================================================
#                Numba-JIT'd forward and backward (octree)
# =================================================================

@njit(cache=True, fastmath=True, inline='always')
def _sample_octree(coarse_density, coarse_color,
                     subdiv, fine_density, fine_color,
                     gx, gy, gz,    # world-space coords in *coarse* cell units
                     Cx, Cy, Cz, F):
    """Sample density (scalar) and color (3) at world-space coord g, given as
    coords in coarse-cell units (i.e. g ∈ [0, Cx) etc.).

    Returns (sigma, cr, cg, cb, hit) where hit=0 means out of bounds.
    """
    # Convert to coarse-cell index
    ci = int(np.floor(gx))
    cj = int(np.floor(gy))
    ck = int(np.floor(gz))
    if ci < 0 or ci >= Cx or cj < 0 or cj >= Cy or ck < 0 or ck >= Cz:
        return 0.0, 0.0, 0.0, 0.0, 0
    # Position within coarse cell, [0, 1)
    fx = gx - ci
    fy = gy - cj
    fz = gz - ck
    m = subdiv[ci, cj, ck]
    if m < 0:
        # Coarse leaf — NN (we don't trilinear-blend coarse leaves with neighbors
        # because mixing into a fine-cell neighbor would require sub-cell descent
        # on the neighbor side. Cheap and correct: sample the leaf cell value).
        sigma = coarse_density[ci, cj, ck]
        cr = coarse_color[ci, cj, ck, 0]
        cg = coarse_color[ci, cj, ck, 1]
        cb = coarse_color[ci, cj, ck, 2]
        return sigma, cr, cg, cb, 1
    # Subdivided — trilinear within fine block
    fgx = fx * F - 0.5
    fgy = fy * F - 0.5
    fgz = fz * F - 0.5
    fi0 = int(np.floor(fgx)); fi1 = fi0 + 1
    fj0 = int(np.floor(fgy)); fj1 = fj0 + 1
    fk0 = int(np.floor(fgz)); fk1 = fk0 + 1
    # Clamp to fine block bounds
    if fi0 < 0: fi0 = 0
    if fi1 >= F: fi1 = F - 1
    if fj0 < 0: fj0 = 0
    if fj1 >= F: fj1 = F - 1
    if fk0 < 0: fk0 = 0
    if fk1 >= F: fk1 = F - 1
    tx = fgx - fi0; ty = fgy - fj0; tz = fgz - fk0
    if tx < 0: tx = 0.0
    if tx > 1: tx = 1.0
    if ty < 0: ty = 0.0
    if ty > 1: ty = 1.0
    if tz < 0: tz = 0.0
    if tz > 1: tz = 1.0
    w000 = (1-tx)*(1-ty)*(1-tz); w001 = (1-tx)*(1-ty)*tz
    w010 = (1-tx)*ty*(1-tz);     w011 = (1-tx)*ty*tz
    w100 = tx*(1-ty)*(1-tz);     w101 = tx*(1-ty)*tz
    w110 = tx*ty*(1-tz);         w111 = tx*ty*tz
    sigma = (w000*fine_density[m,fi0,fj0,fk0] + w001*fine_density[m,fi0,fj0,fk1] +
             w010*fine_density[m,fi0,fj1,fk0] + w011*fine_density[m,fi0,fj1,fk1] +
             w100*fine_density[m,fi1,fj0,fk0] + w101*fine_density[m,fi1,fj0,fk1] +
             w110*fine_density[m,fi1,fj1,fk0] + w111*fine_density[m,fi1,fj1,fk1])
    cr = (w000*fine_color[m,fi0,fj0,fk0,0] + w001*fine_color[m,fi0,fj0,fk1,0] +
          w010*fine_color[m,fi0,fj1,fk0,0] + w011*fine_color[m,fi0,fj1,fk1,0] +
          w100*fine_color[m,fi1,fj0,fk0,0] + w101*fine_color[m,fi1,fj0,fk1,0] +
          w110*fine_color[m,fi1,fj1,fk0,0] + w111*fine_color[m,fi1,fj1,fk1,0])
    cg = (w000*fine_color[m,fi0,fj0,fk0,1] + w001*fine_color[m,fi0,fj0,fk1,1] +
          w010*fine_color[m,fi0,fj1,fk0,1] + w011*fine_color[m,fi0,fj1,fk1,1] +
          w100*fine_color[m,fi1,fj0,fk0,1] + w101*fine_color[m,fi1,fj0,fk1,1] +
          w110*fine_color[m,fi1,fj1,fk0,1] + w111*fine_color[m,fi1,fj1,fk1,1])
    cb = (w000*fine_color[m,fi0,fj0,fk0,2] + w001*fine_color[m,fi0,fj0,fk1,2] +
          w010*fine_color[m,fi0,fj1,fk0,2] + w011*fine_color[m,fi0,fj1,fk1,2] +
          w100*fine_color[m,fi1,fj0,fk0,2] + w101*fine_color[m,fi1,fj0,fk1,2] +
          w110*fine_color[m,fi1,fj1,fk0,2] + w111*fine_color[m,fi1,fj1,fk1,2])
    return sigma, cr, cg, cb, 1


@njit(cache=True, fastmath=True, parallel=True)
def _ray_march_forward_oct(
    coarse_density, coarse_color,
    subdiv, fine_density, fine_color,
    origin, coarse_cell_size, res, fine_factor,
    ray_origins, ray_dirs, t_near, t_far,
    n_samples,
    out_pixels, bg_color,
):
    R = ray_origins.shape[0]
    Cx, Cy, Cz = res[0], res[1], res[2]
    F = fine_factor
    for ri in prange(R):
        if t_far[ri] <= t_near[ri]:
            out_pixels[ri, 0] = bg_color[0]
            out_pixels[ri, 1] = bg_color[1]
            out_pixels[ri, 2] = bg_color[2]
            continue
        ox = ray_origins[ri, 0]; oy = ray_origins[ri, 1]; oz = ray_origins[ri, 2]
        dx = ray_dirs[ri, 0];    dy = ray_dirs[ri, 1];    dz = ray_dirs[ri, 2]
        dt = (t_far[ri] - t_near[ri]) / n_samples
        t = t_near[ri] + 0.5 * dt
        T = 1.0
        cR = 0.0; cG = 0.0; cB = 0.0
        for k in range(n_samples):
            wx = ox + t * dx; wy = oy + t * dy; wz = oz + t * dz
            gx = (wx - origin[0]) / coarse_cell_size[0]
            gy = (wy - origin[1]) / coarse_cell_size[1]
            gz = (wz - origin[2]) / coarse_cell_size[2]
            sigma, cr, cg, cb, hit = _sample_octree(
                coarse_density, coarse_color, subdiv, fine_density, fine_color,
                gx, gy, gz, Cx, Cy, Cz, F)
            if hit and sigma > 1e-7:
                alpha = 1.0 - np.exp(-sigma * dt)
                w = T * alpha
                cR += w * cr
                cG += w * cg
                cB += w * cb
                T *= (1.0 - alpha)
                if T < 1e-4:
                    break
            t += dt
        cR += T * bg_color[0]
        cG += T * bg_color[1]
        cB += T * bg_color[2]
        out_pixels[ri, 0] = cR
        out_pixels[ri, 1] = cG
        out_pixels[ri, 2] = cB


@njit(cache=True, fastmath=True, parallel=True)
def _ray_march_backward_oct(
    coarse_density, coarse_color,
    subdiv, fine_density, fine_color,
    origin, coarse_cell_size, res, fine_factor,
    ray_origins, ray_dirs, t_near, t_far,
    n_samples, residuals,
    grad_coarse_density, grad_coarse_color,
    grad_fine_density,   grad_fine_color,
):
    R = ray_origins.shape[0]
    Cx, Cy, Cz = res[0], res[1], res[2]
    F = fine_factor
    for ri in prange(R):
        if t_far[ri] <= t_near[ri]:
            continue
        ox = ray_origins[ri, 0]; oy = ray_origins[ri, 1]; oz = ray_origins[ri, 2]
        dx = ray_dirs[ri, 0];    dy = ray_dirs[ri, 1];    dz = ray_dirs[ri, 2]
        dt = (t_far[ri] - t_near[ri]) / n_samples

        # Per-ray scratch
        alphas = np.zeros(n_samples, dtype=np.float32)
        Ts     = np.zeros(n_samples, dtype=np.float32)
        cs     = np.zeros((n_samples, 3), dtype=np.float32)
        valid  = np.zeros(n_samples, dtype=np.int32)
        # For each sample we need to remember which cells received what trilinear
        # weight, so backward can splat. For coarse leaves it's 1 cell at weight 1.
        # For fine: 8 cells at trilinear weights.
        # Encoding: kind[k] = 0 (coarse) or 1 (fine)
        # If coarse: ci/cj/ck stored, weight 1 implicit.
        # If fine: m, fi0..fk1, and 8 weights.
        kind   = np.zeros(n_samples, dtype=np.int32)
        cells_i  = np.zeros((n_samples, 3), dtype=np.int32)   # coarse ijk
        fine_m   = np.zeros(n_samples, dtype=np.int32)
        fine_idx = np.zeros((n_samples, 8, 3), dtype=np.int32)  # fi,fj,fk for each of 8
        fine_w   = np.zeros((n_samples, 8), dtype=np.float32)

        # Forward pass + record
        t = t_near[ri] + 0.5 * dt
        T = 1.0
        for k in range(n_samples):
            Ts[k] = T
            wx = ox + t * dx; wy = oy + t * dy; wz = oz + t * dz
            gx = (wx - origin[0]) / coarse_cell_size[0]
            gy = (wy - origin[1]) / coarse_cell_size[1]
            gz = (wz - origin[2]) / coarse_cell_size[2]
            ci = int(np.floor(gx)); cj = int(np.floor(gy)); ck = int(np.floor(gz))
            if ci < 0 or ci >= Cx or cj < 0 or cj >= Cy or ck < 0 or ck >= Cz:
                t += dt
                continue
            fx = gx - ci; fy = gy - cj; fz = gz - ck
            m = subdiv[ci, cj, ck]
            if m < 0:
                sigma = coarse_density[ci, cj, ck]
                cr = coarse_color[ci, cj, ck, 0]
                cg = coarse_color[ci, cj, ck, 1]
                cb = coarse_color[ci, cj, ck, 2]
                kind[k] = 0
                cells_i[k, 0] = ci; cells_i[k, 1] = cj; cells_i[k, 2] = ck
            else:
                fgx = fx * F - 0.5; fgy = fy * F - 0.5; fgz = fz * F - 0.5
                fi0 = int(np.floor(fgx)); fi1 = fi0 + 1
                fj0 = int(np.floor(fgy)); fj1 = fj0 + 1
                fk0 = int(np.floor(fgz)); fk1 = fk0 + 1
                if fi0 < 0: fi0 = 0
                if fi1 >= F: fi1 = F - 1
                if fj0 < 0: fj0 = 0
                if fj1 >= F: fj1 = F - 1
                if fk0 < 0: fk0 = 0
                if fk1 >= F: fk1 = F - 1
                tx = fgx - fi0; ty = fgy - fj0; tz = fgz - fk0
                if tx < 0: tx = 0.0
                if tx > 1: tx = 1.0
                if ty < 0: ty = 0.0
                if ty > 1: ty = 1.0
                if tz < 0: tz = 0.0
                if tz > 1: tz = 1.0
                w000 = (1-tx)*(1-ty)*(1-tz); w001 = (1-tx)*(1-ty)*tz
                w010 = (1-tx)*ty*(1-tz);     w011 = (1-tx)*ty*tz
                w100 = tx*(1-ty)*(1-tz);     w101 = tx*(1-ty)*tz
                w110 = tx*ty*(1-tz);         w111 = tx*ty*tz
                sigma = (w000*fine_density[m,fi0,fj0,fk0] + w001*fine_density[m,fi0,fj0,fk1] +
                         w010*fine_density[m,fi0,fj1,fk0] + w011*fine_density[m,fi0,fj1,fk1] +
                         w100*fine_density[m,fi1,fj0,fk0] + w101*fine_density[m,fi1,fj0,fk1] +
                         w110*fine_density[m,fi1,fj1,fk0] + w111*fine_density[m,fi1,fj1,fk1])
                cr = (w000*fine_color[m,fi0,fj0,fk0,0] + w001*fine_color[m,fi0,fj0,fk1,0] +
                      w010*fine_color[m,fi0,fj1,fk0,0] + w011*fine_color[m,fi0,fj1,fk1,0] +
                      w100*fine_color[m,fi1,fj0,fk0,0] + w101*fine_color[m,fi1,fj0,fk1,0] +
                      w110*fine_color[m,fi1,fj1,fk0,0] + w111*fine_color[m,fi1,fj1,fk1,0])
                cg = (w000*fine_color[m,fi0,fj0,fk0,1] + w001*fine_color[m,fi0,fj0,fk1,1] +
                      w010*fine_color[m,fi0,fj1,fk0,1] + w011*fine_color[m,fi0,fj1,fk1,1] +
                      w100*fine_color[m,fi1,fj0,fk0,1] + w101*fine_color[m,fi1,fj0,fk1,1] +
                      w110*fine_color[m,fi1,fj1,fk0,1] + w111*fine_color[m,fi1,fj1,fk1,1])
                cb = (w000*fine_color[m,fi0,fj0,fk0,2] + w001*fine_color[m,fi0,fj0,fk1,2] +
                      w010*fine_color[m,fi0,fj1,fk0,2] + w011*fine_color[m,fi0,fj1,fk1,2] +
                      w100*fine_color[m,fi1,fj0,fk0,2] + w101*fine_color[m,fi1,fj0,fk1,2] +
                      w110*fine_color[m,fi1,fj1,fk0,2] + w111*fine_color[m,fi1,fj1,fk1,2])
                kind[k] = 1
                fine_m[k] = m
                fine_idx[k, 0, 0]=fi0; fine_idx[k, 0, 1]=fj0; fine_idx[k, 0, 2]=fk0; fine_w[k, 0]=w000
                fine_idx[k, 1, 0]=fi0; fine_idx[k, 1, 1]=fj0; fine_idx[k, 1, 2]=fk1; fine_w[k, 1]=w001
                fine_idx[k, 2, 0]=fi0; fine_idx[k, 2, 1]=fj1; fine_idx[k, 2, 2]=fk0; fine_w[k, 2]=w010
                fine_idx[k, 3, 0]=fi0; fine_idx[k, 3, 1]=fj1; fine_idx[k, 3, 2]=fk1; fine_w[k, 3]=w011
                fine_idx[k, 4, 0]=fi1; fine_idx[k, 4, 1]=fj0; fine_idx[k, 4, 2]=fk0; fine_w[k, 4]=w100
                fine_idx[k, 5, 0]=fi1; fine_idx[k, 5, 1]=fj0; fine_idx[k, 5, 2]=fk1; fine_w[k, 5]=w101
                fine_idx[k, 6, 0]=fi1; fine_idx[k, 6, 1]=fj1; fine_idx[k, 6, 2]=fk0; fine_w[k, 6]=w110
                fine_idx[k, 7, 0]=fi1; fine_idx[k, 7, 1]=fj1; fine_idx[k, 7, 2]=fk1; fine_w[k, 7]=w111
            alpha = 1.0 - np.exp(-sigma * dt)
            alphas[k] = alpha
            cs[k, 0] = cr; cs[k, 1] = cg; cs[k, 2] = cb
            valid[k] = 1
            T *= (1.0 - alpha)
            if T < 1e-4:
                break
            t += dt

        # Backward sweep
        rest_r = 0.0; rest_g = 0.0; rest_b = 0.0
        for k in range(n_samples - 1, -1, -1):
            if valid[k] == 0:
                continue
            T_k = Ts[k]; alpha_k = alphas[k]
            w_k = T_k * alpha_k
            cr_k = cs[k, 0]; cg_k = cs[k, 1]; cb_k = cs[k, 2]
            dc_r = w_k * residuals[ri, 0]
            dc_g = w_k * residuals[ri, 1]
            dc_b = w_k * residuals[ri, 2]
            dL_dalpha = ((T_k * cr_k - rest_r) * residuals[ri, 0]
                       + (T_k * cg_k - rest_g) * residuals[ri, 1]
                       + (T_k * cb_k - rest_b) * residuals[ri, 2])
            dL_dsigma = dL_dalpha * dt * (1.0 - alpha_k)
            if kind[k] == 0:
                # Coarse
                ii = cells_i[k, 0]; jj = cells_i[k, 1]; kk_ = cells_i[k, 2]
                grad_coarse_density[ii, jj, kk_] += dL_dsigma
                grad_coarse_color[ii, jj, kk_, 0] += dc_r
                grad_coarse_color[ii, jj, kk_, 1] += dc_g
                grad_coarse_color[ii, jj, kk_, 2] += dc_b
            else:
                m = fine_m[k]
                for v in range(8):
                    fi = fine_idx[k, v, 0]; fj = fine_idx[k, v, 1]; fk = fine_idx[k, v, 2]
                    w = fine_w[k, v]
                    grad_fine_density[m, fi, fj, fk]    += dL_dsigma * w
                    grad_fine_color[m, fi, fj, fk, 0]   += dc_r * w
                    grad_fine_color[m, fi, fj, fk, 1]   += dc_g * w
                    grad_fine_color[m, fi, fj, fk, 2]   += dc_b * w
            rest_r += w_k * cr_k
            rest_g += w_k * cg_k
            rest_b += w_k * cb_k


# =================================================================
#                Python wrappers + optimizer
# =================================================================

def render_image_oct(oct_grid: PhoxelOctree, cam_intr, cam_extr,
                       H: int = None, W: int = None,
                       n_samples: int = 64, bg_color=(0.07, 0.07, 0.07)):
    from .phoxel import build_camera_rays, _ray_aabb_intersect
    ro, rd, (Hr, Wr) = build_camera_rays(cam_intr, cam_extr, H=H, W=W)
    aabb_lo = oct_grid.origin
    aabb_hi = oct_grid.origin + oct_grid.size
    tn, tf = _ray_aabb_intersect(ro, rd, aabb_lo, aabb_hi)
    out = np.zeros((ro.shape[0], 3), dtype=np.float32)
    bg = np.array(bg_color, dtype=np.float32)
    _ray_march_forward_oct(
        oct_grid.coarse_density, oct_grid.coarse_color,
        oct_grid.subdiv, oct_grid.fine_density, oct_grid.fine_color,
        oct_grid.origin.astype(np.float32),
        oct_grid.coarse_cell_size.astype(np.float32),
        oct_grid.res.astype(np.int32),
        np.int32(oct_grid.fine_factor),
        ro, rd, tn, tf, n_samples, out, bg,
    )
    return out.reshape(Hr, Wr, 3)


def accumulate_grad_oct(oct_grid: PhoxelOctree, cam_intr, cam_extr,
                          gt_image, rendered, n_samples: int = 64,
                          grad_cd=None, grad_cc=None,
                          grad_fd=None, grad_fc=None):
    from .phoxel import build_camera_rays, _ray_aabb_intersect
    H, W = gt_image.shape[:2]
    ro, rd, _ = build_camera_rays(cam_intr, cam_extr, H=H, W=W)
    aabb_lo = oct_grid.origin
    aabb_hi = oct_grid.origin + oct_grid.size
    tn, tf = _ray_aabb_intersect(ro, rd, aabb_lo, aabb_hi)
    residuals = (rendered - gt_image).reshape(-1, 3).astype(np.float32)
    if grad_cd is None: grad_cd = np.zeros_like(oct_grid.coarse_density)
    if grad_cc is None: grad_cc = np.zeros_like(oct_grid.coarse_color)
    if grad_fd is None: grad_fd = np.zeros_like(oct_grid.fine_density)
    if grad_fc is None: grad_fc = np.zeros_like(oct_grid.fine_color)
    _ray_march_backward_oct(
        oct_grid.coarse_density, oct_grid.coarse_color,
        oct_grid.subdiv, oct_grid.fine_density, oct_grid.fine_color,
        oct_grid.origin.astype(np.float32),
        oct_grid.coarse_cell_size.astype(np.float32),
        oct_grid.res.astype(np.int32),
        np.int32(oct_grid.fine_factor),
        ro, rd, tn, tf, n_samples, residuals,
        grad_cd, grad_cc, grad_fd, grad_fc,
    )
    return grad_cd, grad_cc, grad_fd, grad_fc


@dataclass
class OctreeOptimizer:
    """RMSProp on coarse + fine arrays separately."""
    lr_density: float = 2.0
    lr_color:   float = 0.3
    rms_decay:  float = 0.95
    eps:        float = 1e-6

    rms_cd: Optional[np.ndarray] = None
    rms_cc: Optional[np.ndarray] = None
    rms_fd: Optional[np.ndarray] = None
    rms_fc: Optional[np.ndarray] = None

    def step(self, oct_grid: PhoxelOctree,
              grad_cd, grad_cc, grad_fd, grad_fc,
              n_rays_seen: int = 1):
        gcd = grad_cd / max(n_rays_seen, 1)
        gcc = grad_cc / max(n_rays_seen, 1)
        gfd = grad_fd / max(n_rays_seen, 1)
        gfc = grad_fc / max(n_rays_seen, 1)
        # RMS state — re-init if shapes changed (subdivision happened)
        if self.rms_cd is None or self.rms_cd.shape != gcd.shape:
            self.rms_cd = np.zeros_like(gcd); self.rms_cc = np.zeros_like(gcc)
        if self.rms_fd is None or self.rms_fd.shape != gfd.shape:
            self.rms_fd = np.zeros_like(gfd); self.rms_fc = np.zeros_like(gfc)
        self.rms_cd = self.rms_decay * self.rms_cd + (1 - self.rms_decay) * gcd * gcd
        self.rms_cc = self.rms_decay * self.rms_cc + (1 - self.rms_decay) * gcc * gcc
        if gfd.size > 0:
            self.rms_fd = self.rms_decay * self.rms_fd + (1 - self.rms_decay) * gfd * gfd
            self.rms_fc = self.rms_decay * self.rms_fc + (1 - self.rms_decay) * gfc * gfc
        oct_grid.coarse_density -= self.lr_density * gcd / (np.sqrt(self.rms_cd) + self.eps)
        oct_grid.coarse_color   -= self.lr_color   * gcc / (np.sqrt(self.rms_cc) + self.eps)
        if gfd.size > 0:
            oct_grid.fine_density   -= self.lr_density * gfd / (np.sqrt(self.rms_fd) + self.eps)
            oct_grid.fine_color     -= self.lr_color   * gfc / (np.sqrt(self.rms_fc) + self.eps)
        np.maximum(oct_grid.coarse_density, 0.0, out=oct_grid.coarse_density)
        np.clip(oct_grid.coarse_color, 0.0, 1.0, out=oct_grid.coarse_color)
        if gfd.size > 0:
            np.maximum(oct_grid.fine_density, 0.0, out=oct_grid.fine_density)
            np.clip(oct_grid.fine_color, 0.0, 1.0, out=oct_grid.fine_color)


# =================================================================
#                Octree → BlobBundle extraction
# =================================================================

def extract_blobs_from_octree(oct_grid: PhoxelOctree,
                                density_threshold: float = 0.5,
                                max_blobs: int = 500_000):
    """Extract blobs from both coarse leaves and fine cells.

    Coarse blob has scale = 0.5 * coarse_cell_size.
    Fine blob has scale = 0.5 * fine_cell_size  (1/F as big).
    """
    from .data_classes import BlobBundle
    Cx, Cy, Cz = oct_grid.res
    F = oct_grid.fine_factor
    cell = oct_grid.coarse_cell_size
    fcell = oct_grid.fine_cell_size

    xyz_list = []; sigma_list = []; col_list = []; scale_list = []

    # Coarse leaves
    leaf_mask = (oct_grid.subdiv == -1) & (oct_grid.coarse_density > density_threshold)
    if leaf_mask.any():
        idx = np.argwhere(leaf_mask)
        xyz = oct_grid.origin[None, :] + (idx.astype(np.float32) + 0.5) * cell[None, :]
        sigmas = oct_grid.coarse_density[idx[:,0], idx[:,1], idx[:,2]]
        cols = oct_grid.coarse_color[idx[:,0], idx[:,1], idx[:,2]]
        scales = np.tile(np.log(0.5 * cell), (len(idx), 1)).astype(np.float32)
        xyz_list.append(xyz); sigma_list.append(sigmas); col_list.append(cols); scale_list.append(scales)

    # Fine cells in subdivided coarse cells
    for ci in range(Cx):
        for cj in range(Cy):
            for ck in range(Cz):
                m = oct_grid.subdiv[ci, cj, ck]
                if m < 0: continue
                fd = oct_grid.fine_density[m]
                if not (fd > density_threshold).any(): continue
                fidx = np.argwhere(fd > density_threshold)
                # coarse origin = origin + (ci,cj,ck) * cell
                base = oct_grid.origin + np.array([ci, cj, ck]) * cell
                xyz = base[None, :] + (fidx.astype(np.float32) + 0.5) * fcell[None, :]
                sigmas = fd[fidx[:,0], fidx[:,1], fidx[:,2]]
                cols = oct_grid.fine_color[m][fidx[:,0], fidx[:,1], fidx[:,2]]
                scales = np.tile(np.log(0.5 * fcell), (len(fidx), 1)).astype(np.float32)
                xyz_list.append(xyz); sigma_list.append(sigmas); col_list.append(cols); scale_list.append(scales)

    if not xyz_list:
        return BlobBundle(
            xyz=np.zeros((0,3), dtype=np.float32),
            scales=np.zeros((0,3), dtype=np.float32),
            quats=np.zeros((0,4), dtype=np.float32),
            opacity=np.zeros(0, dtype=np.float32),
            sh_dc=np.zeros((0,3), dtype=np.float32),
        )
    xyz = np.concatenate(xyz_list, axis=0).astype(np.float32)
    sig = np.concatenate(sigma_list, axis=0).astype(np.float32)
    col = np.concatenate(col_list, axis=0).astype(np.float32)
    scl = np.concatenate(scale_list, axis=0).astype(np.float32)
    n = len(xyz)
    if n > max_blobs:
        order = np.argsort(sig)[::-1][:max_blobs]
        xyz = xyz[order]; sig = sig[order]; col = col[order]; scl = scl[order]
        n = len(xyz)
    quats = np.zeros((n, 4), dtype=np.float32); quats[:, 0] = 1.0
    cell_diag = float(np.linalg.norm(cell))
    opa = (1.0 - np.exp(-sig * cell_diag)).astype(np.float32)
    return BlobBundle(
        xyz=xyz, scales=scl, quats=quats, opacity=opa, sh_dc=col,
        sh_rest=None, tier=None,
    )
