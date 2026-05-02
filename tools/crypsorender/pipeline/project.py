"""Project splats to pixel space + 2D covariances (EWA)."""

from __future__ import annotations
import numpy as np

from ..io.splat_buffer import SplatBuffer
from ..math.ewa import build_3d_cov, ewa_project, get_radius_2d, invert_2x2
from .camera import Camera


def project_splats(scene: SplatBuffer, camera: Camera):
    """Returns: centers_2d (m,2 in pixels), cov_2d_inv (m,2,2),
    radii (m,), depths (m, +z forward), keep mask (n,), splat indices (m,)."""
    n = scene.n
    centers_3d_cam = camera.world_to_cam(scene.xyz)
    depths = -centers_3d_cam[:, 2]  # positive = in front (camera looks down -z)
    keep = depths > camera.params.near
    if not keep.any():
        z = np.zeros
        return (z((0,2),np.float32), z((0,2,2),np.float32), z((0,),np.float32),
                z((0,),np.float32), keep, np.array([], dtype=np.int32))

    centers_3d_cam_v = centers_3d_cam[keep]
    scales_v = scene.scales[keep]
    quats_v = scene.quats[keep]
    depths_v = depths[keep]

    cov_3d = build_3d_cov(scales_v, quats_v)
    jac = camera.projection_jacobian(centers_3d_cam_v)
    # world -> camera rotation: this is camera.view_rot (3,3) — the same matrix
    # used by world_to_cam.  Broadcast as (1,3,3) and ewa_project will tile it.
    centers_2d, cov_2d = ewa_project(
        jac, camera.view_rot, centers_3d_cam_v, cov_3d,
        focal=camera.focal, size=camera.size,
    )
    cov_2d_inv = invert_2x2(cov_2d)
    radii = get_radius_2d(cov_2d, sigma=3.0)

    # Cull screen-space giants: 3-sigma > 1/16 of image side. These are usually
    # degenerate flat splats that touch hundreds of tiles each but contribute
    # almost nothing visually. Standard 3DGS rasterizers skip them too.
    MAX_RADIUS_FRAC = 1.0 / 16.0
    max_radius = MAX_RADIUS_FRAC * camera.size
    radius_ok = radii <= max_radius
    if not radius_ok.all():
        n_culled = int((~radius_ok).sum())
        print(f"  culled {n_culled} oversized splats (>{max_radius:.0f}px)", flush=True)
        # Filter all per-visible arrays
        centers_2d = centers_2d[radius_ok]
        cov_2d_inv = cov_2d_inv[radius_ok]
        radii = radii[radius_ok]
        depths_v = depths_v[radius_ok]
        # Update keep mask: only the splats that survive both checks
        keep_idx = np.where(keep)[0][radius_ok]
        keep = np.zeros_like(keep)
        keep[keep_idx] = True

    splat_indices = np.where(keep)[0].astype(np.int32)
    return centers_2d, cov_2d_inv, radii, depths_v, keep, splat_indices
