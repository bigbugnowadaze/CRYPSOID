"""F.23 — Contracted phoxel grid for unbounded outdoor scenes.

Mip-NeRF 360 contraction (Barron et al. 2022): squashes all of unbounded R³
into a finite ball of radius 2, so a fixed-size dense voxel grid can
represent foreground at full detail AND distant background at sky-resolution
in the same data structure.

Pipeline difference vs bounded PhoxelGrid:

    bounded:     world coord ──── trilinear lookup ──> grid[i,j,k]
    contracted:  world coord ─ normalise ─ contract ─ trilinear ─> grid

Normalise puts cameras at ||x|| ≈ 1; contract maps r > 1 into [1, 2).

Per docs/scene_contraction_spec.md.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from numba import njit, prange

# =================================================================
#                   Coordinate transforms
# =================================================================
# All scalar / inlinable so Numba can use them inside the hot loops.


@njit(cache=True, fastmath=True, inline='always')
def _contract_xyz(x, y, z):
    """Mip-NeRF 360 contraction of a single point (scalar form)."""
    r = np.sqrt(x * x + y * y + z * z)
    if r <= 1.0:
        return x, y, z
    s = (2.0 - 1.0 / r) / r
    return x * s, y * s, z * s


def contract_array(xyz: np.ndarray) -> np.ndarray:
    """Vectorised contraction for offline / numpy use."""
    r = np.linalg.norm(xyz, axis=-1, keepdims=True)
    s = np.where(r > 1.0, (2.0 - 1.0 / np.maximum(r, 1e-9)) / np.maximum(r, 1e-9), 1.0)
    return xyz * s


def uncontract_array(c: np.ndarray) -> np.ndarray:
    """Inverse of contract_array; for diagnostics only."""
    rc = np.linalg.norm(c, axis=-1, keepdims=True)
    # If rc <= 1 we're in the inner ball: identity. Else solve r from rc:
    #   rc = (2 - 1/r), so r = 1/(2 - rc).
    r = np.where(rc > 1.0, 1.0 / np.maximum(2.0 - rc, 1e-6), rc)
    s = np.where(rc > 1.0, r / np.maximum(rc, 1e-9), 1.0)
    return c * s


# =================================================================
#                    Contracted grid container
# =================================================================


@dataclass
class ContractedPhoxelGrid:
    """Dense voxel grid living in CONTRACTED space.

    The grid covers the cube (-2, -2, -2) to (+2, +2, +2). The Mip-NeRF 360
    contracted ball (radius 2) sits inside that cube; cells outside the ball
    are unreachable but cost almost nothing (just memory).

    `world_center` and `world_scale` define the world↔normalised transform:
        normalised = (world - world_center) / world_scale
    `world_scale` should be the median camera-to-center distance, so that the
    typical camera sits at ||x|| ≈ 1 in normalised space.
    """

    # World-space normalisation (set at construction)
    world_center: np.ndarray   # (3,)
    world_scale:  float

    # Grid lives in contracted space: origin = -2, size = 4, fixed.
    res:     np.ndarray        # (3,) int
    density: np.ndarray        # (Nx, Ny, Nz) float32
    color:   np.ndarray        # (Nx, Ny, Nz, 3) float32

    @property
    def origin(self) -> np.ndarray:
        return np.array([-2.0, -2.0, -2.0], dtype=np.float32)

    @property
    def size(self) -> np.ndarray:
        return np.array([4.0, 4.0, 4.0], dtype=np.float32)

    @property
    def cell_size(self) -> np.ndarray:
        return self.size / self.res.astype(np.float32)

    @classmethod
    def from_cameras(cls, camera_positions: np.ndarray, res: int = 96,
                     init_density: float = 0.05, init_color: float = 0.5):
        """Build a fresh contracted grid sized from a set of camera positions.

        camera_positions: (N, 3). We use mean as center and median ||cam-center||
        as scale.
        """
        cam = np.asarray(camera_positions, dtype=np.float32)
        if cam.ndim != 2 or cam.shape[1] != 3 or cam.shape[0] < 1:
            raise ValueError(f'camera_positions must be (N,3) with N>=1, got {cam.shape}')
        center = cam.mean(axis=0)
        d = np.linalg.norm(cam - center, axis=1)
        scale = float(np.median(d)) if cam.shape[0] > 1 else 1.0
        if scale < 1e-6:
            scale = 1.0   # all cams coincident → no information; still build
        res_arr = np.array([res, res, res], dtype=np.int32)
        density = np.full(tuple(res_arr), init_density, dtype=np.float32)
        color = np.full((res_arr[0], res_arr[1], res_arr[2], 3), init_color, dtype=np.float32)
        return cls(world_center=center, world_scale=scale, res=res_arr,
                   density=density, color=color)


# =================================================================
#               Per-ray sample schedule (disparity)
# =================================================================


def disparity_t_samples(t_near: np.ndarray, t_far: np.ndarray, n: int,
                         split_inner: float = 0.6, t_inner: float = 2.5):
    """Per-ray sample positions: dense in foreground, sparse in background.

    We split the budget: first `n_inner = round(n * split_inner)` samples are
    LINEAR in t between t_near and t_inner (covers the unit-ball foreground at
    full resolution). The remaining `n - n_inner` samples are LINEAR in
    INVERSE-t (disparity) between t_inner and t_far (covers the contracted
    background shell with constant screen-space spacing).

    This keeps foreground sharp AND gives enough tail samples for the bg sphere
    at distances out to t_far.

    Returns:
        t_samples: (R, n)   sample positions in normalised-world units
        dt:        (R, n)   per-sample interval (for alpha-from-sigma)
    """
    R = t_near.shape[0]
    n_inner = max(2, int(round(n * split_inner)))
    n_outer = n - n_inner
    if n_outer < 2:
        n_outer = 2
        n_inner = n - n_outer

    # Inner: linear in t from t_near to t_inner
    s_in = (np.arange(n_inner, dtype=np.float32) + 0.5) / n_inner       # (n_inner,)
    t_in = t_near[:, None] + (t_inner - t_near[:, None]) * s_in[None, :]   # (R, n_inner)

    # Outer: linear in 1/t from t_inner to t_far
    inv_in = 1.0 / max(t_inner, 1e-3)
    inv_far = 1.0 / np.maximum(t_far, 1e-3)[:, None]                    # (R, 1)
    s_out = (np.arange(n_outer, dtype=np.float32) + 0.5) / n_outer
    inv_t_out = inv_in + (inv_far - inv_in) * s_out[None, :]
    t_out = 1.0 / np.maximum(inv_t_out, 1e-6)

    t = np.concatenate([t_in, t_out], axis=1).astype(np.float32)        # (R, n)
    # dt for alpha-compositing: forward neighbour difference
    dt = np.empty_like(t)
    dt[:, :-1] = t[:, 1:] - t[:, :-1]
    dt[:, -1] = dt[:, -2]
    return t.astype(np.float32), dt.astype(np.float32)


# =================================================================
#                   JIT'd forward + backward
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _ray_march_forward_contracted(
    density, color,
    res,
    ray_origins_norm, ray_dirs_norm,    # in NORMALISED world space
    t_samples, dt_samples,              # (R, n) each
    out_pixels,                          # (R, 3)
    bg_color,
):
    R = ray_origins_norm.shape[0]
    Nx, Ny, Nz = res[0], res[1], res[2]
    # cell size = size / res; size = 4
    inv_cell_x = float(Nx) / 4.0
    inv_cell_y = float(Ny) / 4.0
    inv_cell_z = float(Nz) / 4.0
    n = t_samples.shape[1]
    for ri in prange(R):
        ox = ray_origins_norm[ri, 0]; oy = ray_origins_norm[ri, 1]; oz = ray_origins_norm[ri, 2]
        dx = ray_dirs_norm[ri, 0];    dy = ray_dirs_norm[ri, 1];    dz = ray_dirs_norm[ri, 2]
        T = 1.0
        cR = 0.0; cG = 0.0; cB = 0.0
        for k in range(n):
            t = t_samples[ri, k]
            wx = ox + t * dx; wy = oy + t * dy; wz = oz + t * dz
            # Inline contract: maps world to ball of radius 2
            r2 = wx * wx + wy * wy + wz * wz
            if r2 > 1.0:
                rmag = np.sqrt(r2)
                s = (2.0 - 1.0 / rmag) / rmag
                cx = wx * s; cy = wy * s; cz = wz * s
            else:
                cx = wx; cy = wy; cz = wz
            # Grid coords: origin is (-2,-2,-2); grid coord = (c - (-2)) / cell
            #   = (c + 2) * (N / 4)
            gx = (cx + 2.0) * inv_cell_x - 0.5
            gy = (cy + 2.0) * inv_cell_y - 0.5
            gz = (cz + 2.0) * inv_cell_z - 0.5
            i0 = int(np.floor(gx)); i1 = i0 + 1
            j0 = int(np.floor(gy)); j1 = j0 + 1
            kk0 = int(np.floor(gz)); kk1 = kk0 + 1
            if (i0 < 0 or i1 >= Nx or j0 < 0 or j1 >= Ny or kk0 < 0 or kk1 >= Nz):
                continue
            fx = gx - i0; fy = gy - j0; fz = gz - kk0
            w000 = (1-fx)*(1-fy)*(1-fz); w001 = (1-fx)*(1-fy)*fz
            w010 = (1-fx)*fy*(1-fz);     w011 = (1-fx)*fy*fz
            w100 = fx*(1-fy)*(1-fz);     w101 = fx*(1-fy)*fz
            w110 = fx*fy*(1-fz);         w111 = fx*fy*fz
            sigma = (w000*density[i0,j0,kk0] + w001*density[i0,j0,kk1] +
                     w010*density[i0,j1,kk0] + w011*density[i0,j1,kk1] +
                     w100*density[i1,j0,kk0] + w101*density[i1,j0,kk1] +
                     w110*density[i1,j1,kk0] + w111*density[i1,j1,kk1])
            if sigma > 1e-7:
                cr = (w000*color[i0,j0,kk0,0] + w001*color[i0,j0,kk1,0] +
                      w010*color[i0,j1,kk0,0] + w011*color[i0,j1,kk1,0] +
                      w100*color[i1,j0,kk0,0] + w101*color[i1,j0,kk1,0] +
                      w110*color[i1,j1,kk0,0] + w111*color[i1,j1,kk1,0])
                cg = (w000*color[i0,j0,kk0,1] + w001*color[i0,j0,kk1,1] +
                      w010*color[i0,j1,kk0,1] + w011*color[i0,j1,kk1,1] +
                      w100*color[i1,j0,kk0,1] + w101*color[i1,j0,kk1,1] +
                      w110*color[i1,j1,kk0,1] + w111*color[i1,j1,kk1,1])
                cb = (w000*color[i0,j0,kk0,2] + w001*color[i0,j0,kk1,2] +
                      w010*color[i0,j1,kk0,2] + w011*color[i0,j1,kk1,2] +
                      w100*color[i1,j0,kk0,2] + w101*color[i1,j0,kk1,2] +
                      w110*color[i1,j1,kk0,2] + w111*color[i1,j1,kk1,2])
                dt_k = dt_samples[ri, k]
                alpha = 1.0 - np.exp(-sigma * dt_k)
                weight = T * alpha
                cR += weight * cr
                cG += weight * cg
                cB += weight * cb
                T *= (1.0 - alpha)
                if T < 1e-4:
                    break
        cR += T * bg_color[0]
        cG += T * bg_color[1]
        cB += T * bg_color[2]
        out_pixels[ri, 0] = cR
        out_pixels[ri, 1] = cG
        out_pixels[ri, 2] = cB


@njit(cache=True, fastmath=True, parallel=True)
def _ray_march_backward_contracted(
    density, color,
    res,
    ray_origins_norm, ray_dirs_norm,
    t_samples, dt_samples,
    residuals,
    grad_density, grad_color,
):
    R = ray_origins_norm.shape[0]
    Nx, Ny, Nz = res[0], res[1], res[2]
    inv_cell_x = float(Nx) / 4.0
    inv_cell_y = float(Ny) / 4.0
    inv_cell_z = float(Nz) / 4.0
    n = t_samples.shape[1]
    for ri in prange(R):
        ox = ray_origins_norm[ri, 0]; oy = ray_origins_norm[ri, 1]; oz = ray_origins_norm[ri, 2]
        dx = ray_dirs_norm[ri, 0];    dy = ray_dirs_norm[ri, 1];    dz = ray_dirs_norm[ri, 2]
        # Per-ray scratch
        alphas  = np.zeros(n, dtype=np.float32)
        Ts      = np.zeros(n, dtype=np.float32)
        cs      = np.zeros((n, 3), dtype=np.float32)
        valid   = np.zeros(n, dtype=np.int32)
        idxs    = np.zeros((n, 8, 3), dtype=np.int32)
        ws      = np.zeros((n, 8), dtype=np.float32)
        dts_kept = np.zeros(n, dtype=np.float32)
        # Forward pass to fill scratch
        T = 1.0
        for k in range(n):
            Ts[k] = T
            t = t_samples[ri, k]
            wx = ox + t * dx; wy = oy + t * dy; wz = oz + t * dz
            r2 = wx * wx + wy * wy + wz * wz
            if r2 > 1.0:
                rmag = np.sqrt(r2)
                s = (2.0 - 1.0 / rmag) / rmag
                cx = wx * s; cy = wy * s; cz = wz * s
            else:
                cx = wx; cy = wy; cz = wz
            gx = (cx + 2.0) * inv_cell_x - 0.5
            gy = (cy + 2.0) * inv_cell_y - 0.5
            gz = (cz + 2.0) * inv_cell_z - 0.5
            i0 = int(np.floor(gx)); i1 = i0 + 1
            j0 = int(np.floor(gy)); j1 = j0 + 1
            kk0 = int(np.floor(gz)); kk1 = kk0 + 1
            if (i0 < 0 or i1 >= Nx or j0 < 0 or j1 >= Ny or kk0 < 0 or kk1 >= Nz):
                continue
            fx = gx - i0; fy = gy - j0; fz = gz - kk0
            w000 = (1-fx)*(1-fy)*(1-fz); w001 = (1-fx)*(1-fy)*fz
            w010 = (1-fx)*fy*(1-fz);     w011 = (1-fx)*fy*fz
            w100 = fx*(1-fy)*(1-fz);     w101 = fx*(1-fy)*fz
            w110 = fx*fy*(1-fz);         w111 = fx*fy*fz
            sigma = (w000*density[i0,j0,kk0] + w001*density[i0,j0,kk1] +
                     w010*density[i0,j1,kk0] + w011*density[i0,j1,kk1] +
                     w100*density[i1,j0,kk0] + w101*density[i1,j0,kk1] +
                     w110*density[i1,j1,kk0] + w111*density[i1,j1,kk1])
            cr = (w000*color[i0,j0,kk0,0] + w001*color[i0,j0,kk1,0] +
                  w010*color[i0,j1,kk0,0] + w011*color[i0,j1,kk1,0] +
                  w100*color[i1,j0,kk0,0] + w101*color[i1,j0,kk1,0] +
                  w110*color[i1,j1,kk0,0] + w111*color[i1,j1,kk1,0])
            cg = (w000*color[i0,j0,kk0,1] + w001*color[i0,j0,kk1,1] +
                  w010*color[i0,j1,kk0,1] + w011*color[i0,j1,kk1,1] +
                  w100*color[i1,j0,kk0,1] + w101*color[i1,j0,kk1,1] +
                  w110*color[i1,j1,kk0,1] + w111*color[i1,j1,kk1,1])
            cb = (w000*color[i0,j0,kk0,2] + w001*color[i0,j0,kk1,2] +
                  w010*color[i0,j1,kk0,2] + w011*color[i0,j1,kk1,2] +
                  w100*color[i1,j0,kk0,2] + w101*color[i1,j0,kk1,2] +
                  w110*color[i1,j1,kk0,2] + w111*color[i1,j1,kk1,2])
            dt_k = dt_samples[ri, k]
            alpha = 1.0 - np.exp(-sigma * dt_k)
            alphas[k] = alpha
            cs[k, 0] = cr; cs[k, 1] = cg; cs[k, 2] = cb
            idxs[k, 0, 0]=i0; idxs[k, 0, 1]=j0; idxs[k, 0, 2]=kk0; ws[k, 0]=w000
            idxs[k, 1, 0]=i0; idxs[k, 1, 1]=j0; idxs[k, 1, 2]=kk1; ws[k, 1]=w001
            idxs[k, 2, 0]=i0; idxs[k, 2, 1]=j1; idxs[k, 2, 2]=kk0; ws[k, 2]=w010
            idxs[k, 3, 0]=i0; idxs[k, 3, 1]=j1; idxs[k, 3, 2]=kk1; ws[k, 3]=w011
            idxs[k, 4, 0]=i1; idxs[k, 4, 1]=j0; idxs[k, 4, 2]=kk0; ws[k, 4]=w100
            idxs[k, 5, 0]=i1; idxs[k, 5, 1]=j0; idxs[k, 5, 2]=kk1; ws[k, 5]=w101
            idxs[k, 6, 0]=i1; idxs[k, 6, 1]=j1; idxs[k, 6, 2]=kk0; ws[k, 6]=w110
            idxs[k, 7, 0]=i1; idxs[k, 7, 1]=j1; idxs[k, 7, 2]=kk1; ws[k, 7]=w111
            valid[k] = 1
            dts_kept[k] = dt_k
            T *= (1.0 - alpha)
            if T < 1e-4:
                break
        rest_r = 0.0; rest_g = 0.0; rest_b = 0.0
        for k in range(n - 1, -1, -1):
            if valid[k] == 0:
                continue
            T_k = Ts[k]
            alpha_k = alphas[k]
            w_k = T_k * alpha_k
            cr_k = cs[k, 0]; cg_k = cs[k, 1]; cb_k = cs[k, 2]
            dc_r = w_k * residuals[ri, 0]
            dc_g = w_k * residuals[ri, 1]
            dc_b = w_k * residuals[ri, 2]
            for v in range(8):
                ii = idxs[k, v, 0]; jj = idxs[k, v, 1]; kk = idxs[k, v, 2]
                weight = ws[k, v]
                grad_color[ii, jj, kk, 0] += dc_r * weight
                grad_color[ii, jj, kk, 1] += dc_g * weight
                grad_color[ii, jj, kk, 2] += dc_b * weight
            dL_dalpha = ((T_k * cr_k - rest_r) * residuals[ri, 0]
                       + (T_k * cg_k - rest_g) * residuals[ri, 1]
                       + (T_k * cb_k - rest_b) * residuals[ri, 2])
            dL_dsigma = dL_dalpha * dts_kept[k] * (1.0 - alpha_k)
            for v in range(8):
                ii = idxs[k, v, 0]; jj = idxs[k, v, 1]; kk = idxs[k, v, 2]
                grad_density[ii, jj, kk] += dL_dsigma * ws[k, v]
            rest_r += w_k * cr_k
            rest_g += w_k * cg_k
            rest_b += w_k * cb_k


# =================================================================
#                  Python wrappers (mirror phoxel.py shape)
# =================================================================


def _world_to_normalised(grid: ContractedPhoxelGrid, ro_world, rd_world):
    """Apply the world↔normalised transform to a batch of rays.

    Direction stays unit (we normalise its magnitude to 1; the ray parameter
    `t` is then in normalised-world units).
    """
    ro_n = (ro_world - grid.world_center[None, :]) / grid.world_scale
    rd_n = rd_world / grid.world_scale
    nrm = np.linalg.norm(rd_n, axis=1, keepdims=True)
    rd_n = rd_n / np.maximum(nrm, 1e-9)
    return ro_n.astype(np.float32), rd_n.astype(np.float32)


def render_image_contracted(grid: ContractedPhoxelGrid, cam_intr, cam_extr,
                              H: int = None, W: int = None,
                              n_samples: int = 96, t_far_norm: float = 30.0,
                              bg_color=(0.07, 0.07, 0.07)):
    """Render a single camera through the contracted grid.

    t_far_norm is in NORMALISED-WORLD units, not world. 30 covers most of the
    contracted ball for typical setups (cameras at ||x||≈1 → far = 30 means
    the ray passes through 30 normalised units, which contracted maps near
    the boundary).
    """
    from .phoxel import build_camera_rays
    ro_w, rd_w, (Hr, Wr) = build_camera_rays(cam_intr, cam_extr, H=H, W=W)
    ro_n, rd_n = _world_to_normalised(grid, ro_w, rd_w)

    R = ro_n.shape[0]
    t_near = np.full(R, 0.05, dtype=np.float32)
    t_far  = np.full(R, t_far_norm, dtype=np.float32)
    t_samples, dt_samples = disparity_t_samples(t_near, t_far, n_samples)

    out = np.zeros((R, 3), dtype=np.float32)
    bg = np.array(bg_color, dtype=np.float32)
    _ray_march_forward_contracted(
        grid.density, grid.color, grid.res.astype(np.int32),
        ro_n, rd_n, t_samples, dt_samples, out, bg,
    )
    return out.reshape(Hr, Wr, 3)


def accumulate_grad_contracted(grid: ContractedPhoxelGrid, cam_intr, cam_extr,
                                gt_image: np.ndarray, rendered: np.ndarray,
                                n_samples: int = 96, t_far_norm: float = 30.0,
                                grad_density: np.ndarray = None,
                                grad_color: np.ndarray = None):
    from .phoxel import build_camera_rays
    H, W = gt_image.shape[:2]
    ro_w, rd_w, _ = build_camera_rays(cam_intr, cam_extr, H=H, W=W)
    ro_n, rd_n = _world_to_normalised(grid, ro_w, rd_w)

    R = ro_n.shape[0]
    t_near = np.full(R, 0.05, dtype=np.float32)
    t_far  = np.full(R, t_far_norm, dtype=np.float32)
    t_samples, dt_samples = disparity_t_samples(t_near, t_far, n_samples)

    residuals = (rendered - gt_image).reshape(-1, 3).astype(np.float32)
    if grad_density is None:
        grad_density = np.zeros_like(grid.density)
    if grad_color is None:
        grad_color = np.zeros_like(grid.color)
    _ray_march_backward_contracted(
        grid.density, grid.color, grid.res.astype(np.int32),
        ro_n, rd_n, t_samples, dt_samples, residuals,
        grad_density, grad_color,
    )
    return grad_density, grad_color
