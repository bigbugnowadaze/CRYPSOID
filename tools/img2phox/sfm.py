"""Structure-from-Motion stage (synthetic mode for F.2)."""
from __future__ import annotations
import numpy as np
from typing import List
from scipy.optimize import least_squares

from .data_classes import (
    PhotoSet, CameraIntrinsics, CameraExtrinsics, CameraBundle, PointCloud,
)


def project_points(xyz_world, intr, extr):
    cam = extr.world_to_cam(xyz_world)
    z = np.maximum(cam[:, 2], 1e-6)
    px = (cam[:, 0] / z) * intr.focal_x + intr.cx
    py = (cam[:, 1] / z) * intr.focal_y + intr.cy
    # Image-y-down convention: no flip needed (R encodes the down-flip in -cam_up row).
    return np.stack([px, py], axis=1).astype(np.float32)


def make_correspondences_synthetic(true_cloud, true_cameras, noise_px=0.5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for extr in true_cameras.extrinsics:
        proj = project_points(true_cloud.xyz, true_cameras.intrinsics, extr)
        proj = proj + rng.normal(0, noise_px, proj.shape).astype(np.float32)
        out.append(proj)
    return out


def triangulate_two_view(p0, p1, intr, R0, t0, R1, t1):
    K = intr.K
    P0 = K @ np.hstack([R0, t0[:, None]])
    P1 = K @ np.hstack([R1, t1[:, None]])
    M = p0.shape[0]
    out = np.zeros((M, 3), dtype=np.float32)
    for i in range(M):
        u0, v0 = p0[i, 0], p0[i, 1]
        u1, v1 = p1[i, 0], p1[i, 1]
        A = np.stack([
            u0 * P0[2] - P0[0],
            v0 * P0[2] - P0[1],
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
        ], axis=0)
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        out[i] = X[:3] / X[3]
    return out


def _rotvec_to_mat(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)).astype(np.float32)


def _mat_to_rotvec(R):
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float32)
    s = 2 * np.sin(theta)
    k = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / s
    return (theta * k).astype(np.float32)


def bundle_adjust(intr, extrinsics, points, observations, max_iter=30):
    N = len(extrinsics); M = points.shape[0]
    params0 = np.zeros(6*N + 3*M, dtype=np.float64)
    for i, e in enumerate(extrinsics):
        params0[6*i:6*i+3] = _mat_to_rotvec(e.R)
        params0[6*i+3:6*i+6] = e.t
    params0[6*N:] = points.flatten()
    obs_stack = np.stack(observations, axis=0)

    def residuals(params):
        cams = params[:6*N].reshape(N, 6)
        pts  = params[6*N:].reshape(M, 3)
        res = np.zeros(2 * N * M, dtype=np.float64)
        for i in range(N):
            R = _rotvec_to_mat(cams[i, :3])
            t = cams[i, 3:6]
            cam_pts = pts @ R.T + t[None, :]
            z = np.maximum(cam_pts[:, 2], 1e-6)
            px = (cam_pts[:, 0] / z) * intr.focal_x + intr.cx
            py = (cam_pts[:, 1] / z) * intr.focal_y + intr.cy
            obs = obs_stack[i]
            d = np.stack([px - obs[:, 0], py - obs[:, 1]], axis=1)
            res[2*i*M:2*(i+1)*M] = d.flatten()
        return res

    out = least_squares(residuals, params0, method='trf',
                        loss='huber', f_scale=2.0,
                        max_nfev=max_iter*50, verbose=0)
    cams_out = out.x[:6*N].reshape(N, 6)
    pts_out  = out.x[6*N:].reshape(M, 3)
    new_ext = []
    for i in range(N):
        new_ext.append(CameraExtrinsics(
            R=_rotvec_to_mat(cams_out[i, :3]),
            t=cams_out[i, 3:6].astype(np.float32),
        ))
    return new_ext, pts_out.astype(np.float32), float(out.cost)


def run_sfm_synthetic(true_cloud, true_cameras,
                       noise_px=0.5, seed=0,
                       initial_pose_perturbation=0.0,
                       initial_point_perturbation=0.05,
                       run_ba=True):
    intr = true_cameras.intrinsics
    obs = make_correspondences_synthetic(true_cloud, true_cameras, noise_px=noise_px, seed=seed)
    e0, e1 = true_cameras.extrinsics[0], true_cameras.extrinsics[1]
    triangulated = triangulate_two_view(obs[0], obs[1], intr, e0.R, e0.t, e1.R, e1.t)
    z0 = (triangulated @ e0.R.T + e0.t)[:, 2]
    z1 = (triangulated @ e1.R.T + e1.t)[:, 2]
    cheir = (z0 > 0.05) & (z1 > 0.05) & (np.linalg.norm(triangulated, axis=1) < 50.0)
    rng = np.random.default_rng(seed)
    if not cheir.all():
        triangulated[~cheir] = true_cloud.xyz[~cheir] + rng.normal(
            0, initial_point_perturbation, (int((~cheir).sum()), 3))
    initial_pts = triangulated.astype(np.float32)

    if initial_pose_perturbation > 0:
        init_extr = []
        for e in true_cameras.extrinsics:
            R_perturb = _rotvec_to_mat(rng.normal(0, initial_pose_perturbation, 3))
            init_extr.append(CameraExtrinsics(R=(R_perturb @ e.R).astype(np.float32),
                                                t=(e.t + rng.normal(0, initial_pose_perturbation, 3)).astype(np.float32)))
    else:
        init_extr = [CameraExtrinsics(R=e.R.copy(), t=e.t.copy()) for e in true_cameras.extrinsics]

    init_pts = initial_pts + rng.normal(0, initial_point_perturbation, initial_pts.shape).astype(np.float32)

    if run_ba:
        ref_extr, ref_pts, cost = bundle_adjust(intr, init_extr, init_pts, obs)
    else:
        ref_extr, ref_pts, cost = init_extr, init_pts, float('nan')

    out_cloud = PointCloud(xyz=ref_pts, colors=true_cloud.colors,
                            visibility=[set(range(len(true_cameras))) for _ in range(len(ref_pts))])
    out_cams = CameraBundle(intrinsics=intr, extrinsics=ref_extr)
    return out_cams, out_cloud, cost


def pose_error(R_pred, t_pred, R_true, t_true):
    R_diff = R_pred @ R_true.T
    cos = (np.trace(R_diff) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos)))
    t_norm = float(np.linalg.norm(t_pred - t_true) / (np.linalg.norm(t_true) + 1e-9))
    return rot_deg, t_norm


# ---------------- Sparse Bundle Adjustment (F.5+ scale) ----------------
# Replaces the dense bundle_adjust above with a scipy.sparse-aware version
# that scales to thousands of cameras and tens of thousands of points.

from scipy.sparse import lil_matrix


def _build_ba_sparsity(n_cameras, n_points, observations_per_cam):
    """Build the (2*total_obs, 6*N + 3*M) sparsity pattern.

    observations_per_cam[i] is a list of (point_id, u, v) for camera i.
    Each observation row has 6 nonzeros (its camera's params) + 3 nonzeros
    (its point's coords) = 9 nonzeros per residual pair.
    """
    total_obs = sum(len(o) for o in observations_per_cam)
    n_residuals = 2 * total_obs
    n_params = 6 * n_cameras + 3 * n_points
    S = lil_matrix((n_residuals, n_params), dtype=np.uint8)
    row = 0
    for cam_id, obs in enumerate(observations_per_cam):
        cam_off = 6 * cam_id
        for (pt_id, _u, _v) in obs:
            pt_off = 6 * n_cameras + 3 * pt_id
            for k in range(6):
                S[row, cam_off + k] = 1
                S[row + 1, cam_off + k] = 1
            for k in range(3):
                S[row, pt_off + k] = 1
                S[row + 1, pt_off + k] = 1
            row += 2
    return S.tocsr()


def bundle_adjust_sparse(intr, extrinsics, points,
                         observations_per_cam,
                         max_nfev=80, verbose=False):
    """Sparse-Jacobian bundle adjustment using observation lists.

    Args:
        intr: shared CameraIntrinsics (held fixed)
        extrinsics: list of N CameraExtrinsics initial guesses
        points: (M, 3) initial 3D point positions
        observations_per_cam: list of N lists of (point_id, u_pixel, v_pixel) tuples
        max_nfev: max function evaluations for least_squares
        verbose: print scipy progress

    Returns: (refined_extrinsics, refined_points, residual_cost, n_iter)
    """
    N = len(extrinsics); M = points.shape[0]
    H_img = intr.height
    fx, fy, cx, cy = intr.focal_x, intr.focal_y, intr.cx, intr.cy

    # Pack: per-cam (rotvec, t) then per-point xyz
    params0 = np.zeros(6*N + 3*M, dtype=np.float64)
    for i, e in enumerate(extrinsics):
        params0[6*i:6*i+3] = _mat_to_rotvec(e.R)
        params0[6*i+3:6*i+6] = e.t
    params0[6*N:] = points.flatten()

    # Flatten observation list: parallel arrays
    cam_ids = []; pt_ids = []; uvs = []
    for ci, obs in enumerate(observations_per_cam):
        for (pt_id, u, v) in obs:
            cam_ids.append(ci); pt_ids.append(pt_id); uvs.append((u, v))
    cam_ids = np.asarray(cam_ids, dtype=np.int32)
    pt_ids  = np.asarray(pt_ids,  dtype=np.int32)
    uvs     = np.asarray(uvs,     dtype=np.float64)
    n_obs = len(cam_ids)

    def residuals(params):
        cams = params[:6*N].reshape(N, 6)
        pts  = params[6*N:].reshape(M, 3)
        # Vectorized: get this observation's camera + point
        rotvecs = cams[cam_ids, :3]    # (n_obs, 3)
        ts      = cams[cam_ids, 3:6]   # (n_obs, 3)
        xs      = pts[pt_ids]          # (n_obs, 3)
        # Convert all rotvecs to matrices (slow per-obs; one-shot vectorized below)
        # Use vectorized Rodrigues via numpy ops for speed
        thetas = np.linalg.norm(rotvecs, axis=1, keepdims=True) + 1e-12
        ks = rotvecs / thetas
        sin_t = np.sin(thetas); cos_t = np.cos(thetas); one_minus_cos = 1.0 - cos_t
        # cam_pts = R @ x + t, computed per-obs
        # Using Rodrigues form: v_rot = v*cos + (k x v)*sin + k*(k.v)*(1-cos)
        kdotx = np.einsum('ij,ij->i', ks, xs)[:, None]      # (n_obs, 1)
        kcrossx = np.cross(ks, xs)
        rotated = xs * cos_t + kcrossx * sin_t + ks * kdotx * one_minus_cos
        cam_pts = rotated + ts                              # (n_obs, 3)
        z = np.maximum(cam_pts[:, 2], 1e-6)
        px = (cam_pts[:, 0] / z) * fx + cx
        py = (cam_pts[:, 1] / z) * fy + cy
        # Image-y-down: cameras built with -cam_up row mean cam_y already maps
        # to image-y-down. No H_img flip.
        res = np.empty(2 * n_obs)
        res[0::2] = px - uvs[:, 0]
        res[1::2] = py - uvs[:, 1]
        return res

    # Build sparse Jacobian pattern
    sparsity = _build_ba_sparsity(N, M, observations_per_cam)
    if verbose:
        nnz = sparsity.nnz
        dense_size = sparsity.shape[0] * sparsity.shape[1]
        print(f"  [BA] residuals={2*n_obs}, params={6*N+3*M}, "
              f"sparsity nnz={nnz:,}/{dense_size:,} ({100*nnz/dense_size:.4f}%)")

    out = least_squares(
        residuals, params0,
        jac_sparsity=sparsity,
        method='trf', loss='huber', f_scale=2.0,
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )

    cams_out = out.x[:6*N].reshape(N, 6)
    pts_out  = out.x[6*N:].reshape(M, 3)
    new_ext = []
    for i in range(N):
        new_ext.append(CameraExtrinsics(
            R=_rotvec_to_mat(cams_out[i, :3]),
            t=cams_out[i, 3:6].astype(np.float32),
        ))
    return new_ext, pts_out.astype(np.float32), float(out.cost), int(out.nfev)
