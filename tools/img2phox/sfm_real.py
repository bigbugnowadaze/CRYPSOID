"""F.5 + F.5+ Real-photo SfM via OpenCV ORB with continual triangulation + sparse BA."""
from __future__ import annotations
import time
import numpy as np
import cv2
from typing import List, Optional

from .data_classes import (
    Photo, PhotoSet, CameraIntrinsics, CameraExtrinsics, CameraBundle, PointCloud,
)


class FeatureSet:
    def __init__(self, kpts, descs):
        self.kpts = kpts
        self.descs = descs
    def __len__(self): return len(self.kpts)


def detect_features(photos, n_features=5000, verbose=False):
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=15, fastThreshold=15)
    out = []
    for i, ph in enumerate(photos.photos):
        gray = (ph.image * 255).clip(0, 255).astype(np.uint8)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is None:
            kp, desc = [], np.zeros((0, 32), dtype=np.uint8)
        out.append(FeatureSet(kp, desc))
        if verbose:
            print(f"    photo {i}: {len(kp)} ORB features")
    return out


def match_pairs(features, ratio=0.78, min_matches=30, verbose=False):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    out = {}
    N = len(features)
    for i in range(N):
        for j in range(i + 1, N):
            if len(features[i]) < 8 or len(features[j]) < 8: continue
            knn = matcher.knnMatch(features[i].descs, features[j].descs, k=2)
            good = []
            for m_pair in knn:
                if len(m_pair) < 2: continue
                m, n = m_pair
                if m.distance < ratio * n.distance:
                    good.append([m.queryIdx, m.trainIdx])
            if len(good) >= min_matches:
                out[(i, j)] = np.asarray(good, dtype=np.int32)
                if verbose:
                    print(f"    pair {i:2d}-{j:2d}: {len(good)} good matches")
    return out


def keypoints_to_xy(features):
    return [np.array([k.pt for k in f.kpts], dtype=np.float32) for f in features]


def verify_pair_essential(kp_i, kp_j, matches, K, ransac_thresh=1.5):
    pts_i = kp_i[matches[:, 0]]
    pts_j = kp_j[matches[:, 1]]
    E, mask = cv2.findEssentialMat(pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
    if E is None:
        return None, None, None, 0
    _, R, t, mask2 = cv2.recoverPose(E, pts_i, pts_j, K, mask=mask)
    inliers = mask2.ravel().astype(bool) if mask2 is not None else mask.ravel().astype(bool)
    return R.astype(np.float32), t.ravel().astype(np.float32), inliers, int(inliers.sum())


def triangulate(K, R0, t0, R1, t1, pts0, pts1):
    P0 = K @ np.hstack([R0, t0[:, None]])
    P1 = K @ np.hstack([R1, t1[:, None]])
    pts4d = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    pts3d = (pts4d[:3] / pts4d[3:4]).T
    return pts3d.astype(np.float32)


def _grow_tracks_from_pairs(new_cam, added_order, extrinsics, kp_xy, photos, pair_matches,
                              K, tracks, track_lookup, pts3d, point_colors, verbose=False):
    """Continual triangulation: add new tracks from new_cam vs every existing cam."""
    n_new = 0
    for added in added_order:
        if added == new_cam:
            continue
        if (added, new_cam) in pair_matches:
            m = pair_matches[(added, new_cam)]; ka = m[:, 0]; kn = m[:, 1]
        elif (new_cam, added) in pair_matches:
            m = pair_matches[(new_cam, added)]; kn = m[:, 0]; ka = m[:, 1]
        else:
            continue
        pa, pn, pairs = [], [], []
        for k in range(len(m)):
            ai = int(ka[k]); ni = int(kn[k])
            if ai in track_lookup[added]: continue
            if ni in track_lookup[new_cam]: continue
            pa.append(kp_xy[added][ai]); pn.append(kp_xy[new_cam][ni])
            pairs.append((ai, ni))
        if len(pa) < 4:
            continue
        new_pts = triangulate(K, extrinsics[added].R, extrinsics[added].t,
                              extrinsics[new_cam].R, extrinsics[new_cam].t,
                              np.asarray(pa, dtype=np.float32),
                              np.asarray(pn, dtype=np.float32))
        za = (new_pts @ extrinsics[added].R.T + extrinsics[added].t)[:, 2]
        zn = (new_pts @ extrinsics[new_cam].R.T + extrinsics[new_cam].t)[:, 2]
        keep = (za > 0.05) & (zn > 0.05) & (np.linalg.norm(new_pts, axis=1) < 50.0)
        for j in range(len(new_pts)):
            if not keep[j]:
                continue
            ai, ni = pairs[j]
            ti = pts3d.shape[0]
            pts3d = np.vstack([pts3d, new_pts[j:j+1]])
            x, y = kp_xy[added][ai]
            xi = int(np.clip(x, 0, photos[added].width - 1))
            yi = int(np.clip(y, 0, photos[added].height - 1))
            point_colors = np.vstack([point_colors, photos[added].image[yi:yi+1, xi]])
            tracks.append({added: ai, new_cam: ni})
            track_lookup[added][ai] = ti
            track_lookup[new_cam][ni] = ti
            n_new += 1
    if verbose and n_new > 0:
        print(f"  [F.5]   cam {new_cam}: continual triangulation added {n_new} tracks (db now {len(pts3d)})")
    return pts3d, point_colors, n_new


def run_sfm_real(photos, intr=None, fov_deg_prior=50.0,
                  n_features=5000, ratio=0.78, min_matches=30,
                  pnp_min_inliers=4,
                  run_global_ba=True, ba_max_nfev=200, verbose=True):
    t0 = time.perf_counter()
    if intr is None:
        H, W = photos.photos[0].height, photos.photos[0].width
        intr = CameraIntrinsics.from_fov(fov_deg_prior, W, H)
    K = intr.K
    N = len(photos)

    if verbose:
        print(f"  [F.5] {N} photos, intrinsics K=\n{K}")
        print(f"  [F.5] detecting ORB features (target {n_features}/photo) ...")
    feats = detect_features(photos, n_features=n_features, verbose=verbose)
    kp_xy = keypoints_to_xy(feats)

    if verbose:
        print(f"  [F.5] matching pairs (Lowe ratio={ratio}, min_matches={min_matches}) ...")
    pair_matches = match_pairs(feats, ratio=ratio, min_matches=min_matches, verbose=verbose)
    if not pair_matches:
        raise RuntimeError("No pair survived feature matching.")

    if verbose:
        print(f"  [F.5] bootstrapping from best pair ...")
    pair_scores = []
    ess_min_inliers = max(8, min_matches // 4)
    for (i, j), m in pair_matches.items():
        R, t, mask, n_inl = verify_pair_essential(kp_xy[i], kp_xy[j], m, K)
        if R is None or n_inl < ess_min_inliers:
            continue
        pair_scores.append(((i, j), R, t, mask, n_inl))
    if not pair_scores:
        raise RuntimeError("No pair survived essential-matrix verification.")
    pair_scores.sort(key=lambda x: -x[4])
    (boot_i, boot_j), R_rel, t_rel, mask_rel, n_inl = pair_scores[0]
    if verbose:
        print(f"  [F.5]   bootstrap: pair {boot_i}-{boot_j} with {n_inl} inliers")

    extrinsics = [None] * N
    extrinsics[boot_i] = CameraExtrinsics(R=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32))
    extrinsics[boot_j] = CameraExtrinsics(R=R_rel, t=t_rel)

    matches_boot = pair_matches[(boot_i, boot_j)]
    inlier_matches = matches_boot[mask_rel]
    pts3d = triangulate(K, extrinsics[boot_i].R, extrinsics[boot_i].t,
                          extrinsics[boot_j].R, extrinsics[boot_j].t,
                          kp_xy[boot_i][inlier_matches[:, 0]],
                          kp_xy[boot_j][inlier_matches[:, 1]])
    z0 = (pts3d @ extrinsics[boot_i].R.T + extrinsics[boot_i].t)[:, 2]
    z1 = (pts3d @ extrinsics[boot_j].R.T + extrinsics[boot_j].t)[:, 2]
    keep = (z0 > 0.05) & (z1 > 0.05) & (np.linalg.norm(pts3d, axis=1) < 50.0)
    pts3d = pts3d[keep]
    inlier_matches = inlier_matches[keep]
    if verbose:
        print(f"  [F.5]   triangulated {len(pts3d)} bootstrap points")

    tracks = []
    track_lookup = {boot_i: {}, boot_j: {}}
    point_colors = []
    for ti in range(len(pts3d)):
        ai = int(inlier_matches[ti, 0]); ni = int(inlier_matches[ti, 1])
        tracks.append({boot_i: ai, boot_j: ni})
        track_lookup[boot_i][ai] = ti
        track_lookup[boot_j][ni] = ti
        x, y = kp_xy[boot_i][ai]
        xi = int(np.clip(x, 0, photos[boot_i].width - 1))
        yi = int(np.clip(y, 0, photos[boot_i].height - 1))
        point_colors.append(photos[boot_i].image[yi, xi])
    point_colors = np.asarray(point_colors, dtype=np.float32)

    remaining = [k for k in range(N) if k != boot_i and k != boot_j]
    added_order = [boot_i, boot_j]
    for new_cam in remaining:
        twoD, threeD, used = [], [], []
        for added in added_order:
            if (added, new_cam) in pair_matches:
                m = pair_matches[(added, new_cam)]; aks = m[:, 0]; nks = m[:, 1]
            elif (new_cam, added) in pair_matches:
                m = pair_matches[(new_cam, added)]; nks = m[:, 0]; aks = m[:, 1]
            else:
                continue
            for k in range(len(m)):
                ti = track_lookup.get(added, {}).get(int(aks[k]))
                if ti is not None and ti not in used:
                    twoD.append(kp_xy[new_cam][int(nks[k])])
                    threeD.append(pts3d[ti])
                    used.append(ti)
        if len(twoD) < pnp_min_inliers:
            if verbose:
                print(f"  [F.5]   cam {new_cam}: only {len(twoD)} 3D-2D, skip")
            continue
        twoD_np = np.asarray(twoD, dtype=np.float32)
        threeD_np = np.asarray(threeD, dtype=np.float32)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            threeD_np.reshape(-1, 1, 3), twoD_np.reshape(-1, 1, 2),
            K, None, reprojectionError=4.0, confidence=0.999, iterationsCount=100,
        )
        if not ok or inliers is None or len(inliers) < pnp_min_inliers:
            if verbose:
                print(f"  [F.5]   cam {new_cam}: PnP failed")
            continue
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.ravel()
        extrinsics[new_cam] = CameraExtrinsics(R=R_new.astype(np.float32), t=t_new.astype(np.float32))
        added_order.append(new_cam)
        track_lookup[new_cam] = {}
        if verbose:
            print(f"  [F.5]   cam {new_cam}: PnP OK, {len(inliers)}/{len(twoD)} inliers")
        # Continual triangulation
        pts3d, point_colors, _ = _grow_tracks_from_pairs(
            new_cam, added_order, extrinsics, kp_xy, photos, pair_matches,
            K, tracks, track_lookup, pts3d, point_colors, verbose=verbose,
        )

    valid_cam_idx = [k for k, e in enumerate(extrinsics) if e is not None]
    if verbose:
        print(f"  [F.5] registered {len(valid_cam_idx)}/{N} cams, {len(pts3d)} 3D pts")
    final_ext = [extrinsics[k] for k in valid_cam_idx]

    # Global sparse BA
    if run_global_ba and len(final_ext) >= 2 and len(pts3d) >= 8:
        from .sfm import bundle_adjust_sparse
        cam_idx_map = {orig: compact for compact, orig in enumerate(valid_cam_idx)}
        obs_per_cam = [[] for _ in range(len(final_ext))]
        for ti, track in enumerate(tracks):
            for orig_cam, kpt_idx in track.items():
                if orig_cam not in cam_idx_map: continue
                u, v = kp_xy[orig_cam][kpt_idx]
                obs_per_cam[cam_idx_map[orig_cam]].append((ti, float(u), float(v)))
        n_obs = sum(len(o) for o in obs_per_cam)
        if verbose:
            print(f"  [F.5+] global sparse BA: {len(final_ext)} cams, {len(pts3d)} pts, {n_obs} obs ...")
        t_ba = time.perf_counter()
        try:
            final_ext, pts3d, ba_cost, ba_nfev = bundle_adjust_sparse(
                intr, final_ext, pts3d, obs_per_cam, max_nfev=ba_max_nfev, verbose=False,
            )
            if verbose:
                print(f"  [F.5+] BA done in {time.perf_counter()-t_ba:.2f}s, cost={ba_cost:.2f}, nfev={ba_nfev}")
        except Exception as e:
            if verbose:
                print(f"  [F.5+] BA failed: {e}")

    out_cams = CameraBundle(intrinsics=intr, extrinsics=final_ext)
    out_cloud = PointCloud(xyz=pts3d, colors=point_colors,
                            visibility=[set(range(len(valid_cam_idx))) for _ in range(len(pts3d))])
    stats = {
        'n_photos': N,
        'n_pairs_matched': len(pair_matches),
        'n_cameras_registered': len(valid_cam_idx),
        'n_3d_points': len(pts3d),
        'wall_seconds': time.perf_counter() - t0,
        'valid_cam_idx': valid_cam_idx,
    }
    if verbose:
        print(f"  [F.5] done in {stats['wall_seconds']:.2f}s ({len(valid_cam_idx)} cams, {len(pts3d)} pts)")
    return out_cams, out_cloud, stats
