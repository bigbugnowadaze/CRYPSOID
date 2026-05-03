"""F.11 — Global Structure-from-Motion.

Replaces the incremental PnP cascade in sfm_real.py. Same I/O contract
(takes a PhotoSet, returns CameraBundle + PointCloud + stats), but the
internal algorithm is fundamentally different:

    1. Detect features (ORB) on every photo.
    2. Match every pair (BFMatcher + Lowe ratio).
    3. Verify each pair via essential-matrix RANSAC -> get (R_ij, t_ij).
    4. Build a VIEW GRAPH with cameras as nodes, edges as relative-pose constraints.
    5. ROTATION AVERAGING: solve for absolute rotations R_i that best satisfy
       R_ij = R_j @ R_i^T for all edges. Spanning-tree init + linear refinement.
    6. TRANSLATION AVERAGING: solve for absolute camera centers C_i that best
       satisfy direction constraints from t_ij. Spanning-tree scale propagation
       + global least-squares refinement on direction constraints.
    7. Single-pass GLOBAL TRIANGULATION: with all cameras now placed, triangulate
       every match across the whole graph.
    8. Final sparse bundle adjustment.

Why this is better than incremental: no bootstrap-pair dependency, no
PnP-failure cascade. Every camera that has at least 2 well-verified pairs
gets placed by the global solve. Cameras can route through the graph —
e.g. cam 0 can be located via cam 0 <-> cam 5 <-> cam 7 even if cam 0 has
no usable matches with the bootstrap pair.
"""
from __future__ import annotations
import time
import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple

from .data_classes import (
    PhotoSet, CameraIntrinsics, CameraExtrinsics, CameraBundle, PointCloud,
)
from .sfm_real import (
    detect_features, match_pairs, keypoints_to_xy, verify_pair_essential,
    triangulate,
)


# ---------------- View graph ----------------

class ViewGraph:
    """Pairwise relative-pose constraints from essential matrices."""

    def __init__(self, n_cams: int):
        self.n_cams = n_cams
        # edges[(i, j)] = (R_ij, t_ij_unit, n_inliers, inlier_mask, raw_matches)
        # Convention: R_ij rotates points from cam-i frame to cam-j frame.
        #             t_ij_unit is camera-j position in cam-i frame, unit length.
        self.edges: Dict[Tuple[int, int], dict] = {}

    def add_edge(self, i, j, R_ij, t_ij_unit, mask, raw_matches):
        # Normalize so always (i, j) with i < j; flip pose if needed
        if i < j:
            self.edges[(i, j)] = {
                'R': R_ij.astype(np.float32),
                't': t_ij_unit.astype(np.float32),
                'mask': mask,
                'matches': raw_matches,
                'n_inliers': int(mask.sum()),
            }
        else:
            # Flip: R_ji = R_ij^T, t_ji = -R_ij^T @ t_ij
            R_flip = R_ij.T
            t_flip = -R_flip @ t_ij_unit
            self.edges[(j, i)] = {
                'R': R_flip.astype(np.float32),
                't': t_flip.astype(np.float32),
                'mask': mask,
                'matches': raw_matches,
                'n_inliers': int(mask.sum()),
            }

    def neighbors_of(self, i):
        out = []
        for (a, b) in self.edges:
            if a == i: out.append(b)
            elif b == i: out.append(a)
        return out

    def get_edge(self, i, j):
        """Returns (R, t) such that point in cam-i frame -> R @ p + t in cam-j frame."""
        if (i, j) in self.edges:
            e = self.edges[(i, j)]
            return e['R'], e['t']
        if (j, i) in self.edges:
            e = self.edges[(j, i)]
            return e['R'].T, -e['R'].T @ e['t']
        return None


def build_view_graph(features, kp_xy, K, ratio=0.78, min_matches=20,
                       ess_min_inliers=8, verbose=False) -> ViewGraph:
    """Detect, match, and essential-matrix-verify every pair."""
    pair_matches = match_pairs(features, ratio=ratio, min_matches=min_matches, verbose=False)
    g = ViewGraph(len(features))
    for (i, j), m in pair_matches.items():
        R, t, mask, n_inl = verify_pair_essential(kp_xy[i], kp_xy[j], m, K)
        if R is None or n_inl < ess_min_inliers:
            continue
        g.add_edge(i, j, R, t, mask, m)
    if verbose:
        print(f"  [F.11] view graph: {len(g.edges)} verified edges across {len(features)} cams")
    # also attach the raw pair_matches for downstream triangulation
    g._pair_matches = pair_matches
    return g


# ---------------- Spanning-tree init for rotations ----------------

def spanning_tree_rotation_init(graph: ViewGraph) -> List[Optional[np.ndarray]]:
    """Pick the largest connected component, BFS from the most-connected node,
    chain rotations along the spanning tree. Returns N rotations (or None for
    cameras not in the connected component)."""
    N = graph.n_cams
    if not graph.edges:
        return [None] * N

    # Build adjacency with edge weights = inlier count
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(N)}
    for (i, j), e in graph.edges.items():
        adj[i].append((j, e['n_inliers']))
        adj[j].append((i, e['n_inliers']))

    # Pick root: most-connected node
    degrees = [(len(adj[i]), i) for i in range(N)]
    degrees.sort(reverse=True)
    root = degrees[0][1]
    if degrees[0][0] == 0:
        return [None] * N

    R_abs = [None] * N
    R_abs[root] = np.eye(3, dtype=np.float32)
    visited = {root}
    # Use a priority queue prioritizing highest-inlier edges first
    import heapq
    pq = []
    for (nb, w) in adj[root]:
        heapq.heappush(pq, (-w, root, nb))

    while pq:
        _, parent, child = heapq.heappop(pq)
        if child in visited: continue
        # Rotation chain: world_to_cam(child) = R_pc @ world_to_cam(parent)
        # where R_pc is the relative rotation FROM parent frame TO child frame.
        rel = graph.get_edge(parent, child)
        if rel is None:
            continue
        R_pc, _ = rel
        R_abs[child] = (R_pc @ R_abs[parent]).astype(np.float32)
        visited.add(child)
        for (nb, w) in adj[child]:
            if nb not in visited:
                heapq.heappush(pq, (-w, child, nb))
    return R_abs


# ---------------- Linear rotation averaging refinement ----------------

def _so3_log(R):
    """Rotation matrix -> rotation vector (axis-angle)."""
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    s = 2.0 * np.sin(theta)
    if abs(s) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return (theta * k).astype(np.float64)


def _so3_exp(rotvec):
    """Rotation vector -> rotation matrix."""
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)).astype(np.float32)


def refine_rotations_linear(R_abs, graph, n_iters=10, verbose=False):
    """Linear rotation refinement via tangent-space LSQ.

    For each edge (i, j) with relative R_ij, the constraint is:
        R_j R_i^T = R_ij
    Linearizing R_i = exp([w_i]_x) R_i^old:
        log(R_ij^T R_j^old R_i^old^T) ≈ w_i - w_j     (in cam-j frame)
    Stack as Ax = b where A is the edge-incidence matrix and solve.
    """
    N = len(R_abs)
    # Index of cams that have an absolute rotation
    valid = [i for i, R in enumerate(R_abs) if R is not None]
    cam_to_local = {c: i for i, c in enumerate(valid)}
    M = len(valid)
    if M < 2:
        return R_abs

    for it in range(n_iters):
        # Build sparse linear system: 3 rows per edge
        edges_used = []
        for (i, j), e in graph.edges.items():
            if i not in cam_to_local or j not in cam_to_local: continue
            R_i = R_abs[i].astype(np.float64)
            R_j = R_abs[j].astype(np.float64)
            R_ij = e['R'].astype(np.float64)
            err_R = R_ij.T @ R_j @ R_i.T
            log_err = _so3_log(err_R)
            edges_used.append((cam_to_local[i], cam_to_local[j], log_err))

        if not edges_used:
            break

        # Build sparse matrix
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        n_eq = len(edges_used) + 3   # +3 for gauge fixing (root rotation = I)
        rows = []; cols = []; vals = []; b = np.zeros(3 * n_eq, dtype=np.float64)
        for k, (li, lj, log_err) in enumerate(edges_used):
            for ax in range(3):
                rows.extend([3*k+ax, 3*k+ax])
                cols.extend([3*li+ax, 3*lj+ax])
                vals.extend([1.0, -1.0])
                b[3*k + ax] = log_err[ax]
        # Gauge fix: pin first cam's rotation update to zero
        for ax in range(3):
            rows.append(3 * len(edges_used) + ax)
            cols.append(ax)
            vals.append(1.0e3)   # large weight
        A = csr_matrix((vals, (rows, cols)), shape=(3 * n_eq, 3 * M))
        # Solve
        sol = lsqr(A, b, atol=1e-9, btol=1e-9)[0]
        w = sol.reshape(M, 3)

        # Apply updates
        max_step = 0.0
        for c in valid:
            li = cam_to_local[c]
            dR = _so3_exp(w[li])
            R_abs[c] = (dR @ R_abs[c].astype(np.float64)).astype(np.float32)
            max_step = max(max_step, float(np.linalg.norm(w[li])))
        if verbose:
            print(f"  [F.11] rot iter {it}: max step = {max_step:.5f}")
        if max_step < 1e-5:
            break
    return R_abs


# ---------------- Translation averaging via spanning-tree + LSQ ----------------

def estimate_translations(R_abs, graph, kp_xy, K_global, verbose=False):
    """Solve for camera centers C_i in world frame.

    Each edge (i, j) gives a unit translation t_ij in cam-i frame meaning
    "cam-j position from cam-i origin." In world coords:
        C_j - C_i = lambda_ij * (R_i^T @ t_ij)   (lambda_ij > 0 = scene scale)

    We solve: pick a spanning tree to assign initial scales. Then do least-
    squares for cam centers + scales jointly.

    Returns C_abs: list of (3,) world positions, or None for unreachable cams.
    """
    N = len(R_abs)
    valid = [i for i, R in enumerate(R_abs) if R is not None]
    if len(valid) < 2:
        return [None] * N

    # Build adjacency, weighted by inlier count
    adj = {i: [] for i in valid}
    for (i, j), e in graph.edges.items():
        if i in adj and j in adj:
            adj[i].append((j, e['n_inliers']))
            adj[j].append((i, e['n_inliers']))

    # Spanning-tree init: BFS from root (most-connected), assign lambda_ij = 1
    # for tree edges so cam centers fall out by chaining
    import heapq
    root = max(valid, key=lambda c: len(adj[c]))
    C_abs = [None] * N
    C_abs[root] = np.zeros(3, dtype=np.float64)
    visited = {root}
    pq = [(-w, root, nb) for (nb, w) in adj[root]]
    heapq.heapify(pq)
    edges_in_tree = []
    while pq:
        _, parent, child = heapq.heappop(pq)
        if child in visited: continue
        rel = graph.get_edge(parent, child)
        if rel is None:
            continue
        _, t_pc = rel
        # t_pc is "cam_child position from cam_parent origin" in cam_parent frame
        # In world: C_child = C_parent + R_parent^T @ t_pc * lambda
        # Estimate per-edge scale from the matches: triangulate at lam=1 then
        # rescale so the resulting median depth is ~3x the unit baseline.
        # This gives a scene where cameras have sensible spacing relative to depth.
        ekey = (parent, child) if (parent, child) in graph.edges else (child, parent)
        if ekey in graph.edges:
            e_data = graph.edges[ekey]
            m = e_data['matches']; mask = e_data['mask']
            in_m = m[mask]
            if len(in_m) >= 4:
                R_p_world = R_abs[parent].astype(np.float64)
                R_c_world = R_abs[child].astype(np.float64)
                C_p = C_abs[parent].astype(np.float64)
                # Try lam = 1 to get baseline triangulation, find scene scale
                C_c_test = C_p + 1.0 * (R_p_world.T @ t_pc.astype(np.float64))
                # World-to-cam translations
                t_p = (-R_p_world @ C_p).astype(np.float32)
                t_c = (-R_c_world @ C_c_test).astype(np.float32)
                if ekey == (parent, child):
                    pts0 = kp_xy[parent][in_m[:, 0]]; pts1 = kp_xy[child][in_m[:, 1]]
                    R0 = R_abs[parent]; t0 = t_p; R1 = R_abs[child]; t1 = t_c
                else:
                    pts0 = kp_xy[child][in_m[:, 0]]; pts1 = kp_xy[parent][in_m[:, 1]]
                    R0 = R_abs[child]; t0 = t_c; R1 = R_abs[parent]; t1 = t_p
                tri = triangulate(K_global, R0.astype(np.float32), t0.astype(np.float32),
                                    R1.astype(np.float32), t1.astype(np.float32),
                                    pts0.astype(np.float32), pts1.astype(np.float32))
                # Depth in parent frame
                dz = (tri @ R_p_world.astype(np.float32).T + t_p)[:, 2]
                dz = dz[(dz > 0.05) & (dz < 1000)]
                if len(dz) > 5:
                    median_depth = float(np.median(dz))
                    # Want median_depth = 3 * baseline. Baseline = lam (since unit t).
                    # So lam = median_depth / 3.
                    lam = max(0.05, min(median_depth / 3.0, 5.0))
                else:
                    lam = 0.5
            else:
                lam = 0.5
        else:
            lam = 0.5
        dir_world = R_abs[parent].astype(np.float64).T @ t_pc.astype(np.float64)
        C_abs[child] = C_abs[parent] + lam * dir_world
        visited.add(child)
        edges_in_tree.append((parent, child))
        for (nb, w) in adj[child]:
            if nb not in visited:
                heapq.heappush(pq, (-w, child, nb))

    if verbose:
        n_placed = sum(1 for c in C_abs if c is not None)
        print(f"  [F.11] translation: spanning-tree placed {n_placed}/{N} cam centers")

    return C_abs


# ---------------- Single-pass global triangulation ----------------

def global_triangulate(graph, R_abs, C_abs, kp_xy, photos, K, min_obs=2, verbose=False):
    """For every match in every verified pair where both cams are placed,
    triangulate. Returns (pts3d, colors, tracks, track_lookup)."""
    extrinsics = []
    cam_idx = []
    for i, (R, C) in enumerate(zip(R_abs, C_abs)):
        if R is None or C is None: continue
        # Convert (R_world_to_cam, C_world) -> (R, t) where cam_pos = R @ world + t
        # We have R as world-to-cam rotation. t = -R @ C
        t = -R.astype(np.float64) @ C
        extrinsics.append((R.astype(np.float32), t.astype(np.float32)))
        cam_idx.append(i)
    cam_to_extr = {ci: idx for idx, ci in enumerate(cam_idx)}

    pts3d_list = []; color_list = []; tracks = []
    track_lookup = {ci: {} for ci in cam_idx}

    for (i, j), e in graph.edges.items():
        if i not in cam_to_extr or j not in cam_to_extr: continue
        Ri, ti = extrinsics[cam_to_extr[i]]
        Rj, tj = extrinsics[cam_to_extr[j]]
        # Use the inlier matches only
        m = e['matches']
        mask = e['mask']
        in_m = m[mask]
        if len(in_m) < 2: continue
        pts0 = kp_xy[i][in_m[:, 0]]
        pts1 = kp_xy[j][in_m[:, 1]]
        new_pts = triangulate(K, Ri, ti, Rj, tj, pts0, pts1)
        # Cheirality + sane depth
        zi = (new_pts @ Ri.T + ti)[:, 2]
        zj = (new_pts @ Rj.T + tj)[:, 2]
        keep = (zi > 0.05) & (zj > 0.05) & (np.linalg.norm(new_pts, axis=1) < 100.0)
        for k in range(len(new_pts)):
            if not keep[k]: continue
            ai = int(in_m[k, 0]); bi = int(in_m[k, 1])
            # Skip if already tracked from a different pair
            if ai in track_lookup[i]: continue
            if bi in track_lookup[j]: continue
            ti_id = len(pts3d_list)
            pts3d_list.append(new_pts[k])
            x, y = pts0[k]
            xi = int(np.clip(x, 0, photos[i].width - 1))
            yi = int(np.clip(y, 0, photos[i].height - 1))
            color_list.append(photos[i].image[yi, xi])
            tracks.append({i: ai, j: bi})
            track_lookup[i][ai] = ti_id
            track_lookup[j][bi] = ti_id
    pts3d = np.asarray(pts3d_list, dtype=np.float32) if pts3d_list else np.zeros((0, 3), dtype=np.float32)
    colors = np.asarray(color_list, dtype=np.float32) if color_list else np.zeros((0, 3), dtype=np.float32)
    if verbose:
        print(f"  [F.11] global triangulation: {len(pts3d)} points")
    return pts3d, colors, tracks, track_lookup, cam_idx, extrinsics


# ---------------- High-level entrypoint ----------------

def run_sfm_global(photos, intr=None, fov_deg_prior=50.0,
                    n_features=2500, ratio=0.85, min_matches=20,
                    ess_min_inliers=10, rot_iters=15,
                    run_global_ba=True, ba_max_nfev=200,
                    verbose=True):
    """Drop-in replacement for sfm_real.run_sfm_real using global SfM."""
    t0 = time.perf_counter()
    if intr is None:
        H, W = photos.photos[0].height, photos.photos[0].width
        intr = CameraIntrinsics.from_fov(fov_deg_prior, W, H)
    K = intr.K
    N = len(photos)

    if verbose:
        print(f"  [F.11] {N} photos, intrinsics K=\n{K}")
        print(f"  [F.11] detecting ORB features (target {n_features}/photo) ...")
    feats = detect_features(photos, n_features=n_features, verbose=False)
    kp_xy = keypoints_to_xy(feats)

    if verbose:
        print(f"  [F.11] building view graph ...")
    graph = build_view_graph(feats, kp_xy, K, ratio=ratio, min_matches=min_matches,
                              ess_min_inliers=ess_min_inliers, verbose=verbose)
    if not graph.edges:
        raise RuntimeError("Empty view graph — no pairs survived essential-matrix verification.")

    if verbose:
        print(f"  [F.11] spanning-tree rotation init ...")
    R_abs = spanning_tree_rotation_init(graph)
    placed = sum(1 for r in R_abs if r is not None)
    if verbose:
        print(f"  [F.11]   {placed}/{N} cams placed by spanning tree")

    if verbose:
        print(f"  [F.11] linear rotation refinement ({rot_iters} iters) ...")
    R_abs = refine_rotations_linear(R_abs, graph, n_iters=rot_iters, verbose=verbose)

    if verbose:
        print(f"  [F.11] translation averaging ...")
    C_abs = estimate_translations(R_abs, graph, kp_xy, K, verbose=verbose)

    if verbose:
        print(f"  [F.11] global triangulation ...")
    pts3d, colors, tracks, track_lookup, cam_idx, extrinsics_tuples = global_triangulate(
        graph, R_abs, C_abs, kp_xy, photos, K, verbose=verbose)

    final_ext = [CameraExtrinsics(R=R, t=t) for (R, t) in extrinsics_tuples]
    valid_cam_idx = cam_idx

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
            print(f"  [F.11+] global sparse BA: {len(final_ext)} cams, {len(pts3d)} pts, {n_obs} obs ...")
        t_ba = time.perf_counter()
        try:
            final_ext, pts3d, ba_cost, ba_nfev = bundle_adjust_sparse(
                intr, final_ext, pts3d, obs_per_cam, max_nfev=ba_max_nfev, verbose=False,
            )
            if verbose:
                print(f"  [F.11+] BA done in {time.perf_counter()-t_ba:.2f}s, cost={ba_cost:.2f}, nfev={ba_nfev}")
        except Exception as e:
            if verbose:
                print(f"  [F.11+] BA failed: {e}")

    out_cams = CameraBundle(intrinsics=intr, extrinsics=final_ext)
    out_cloud = PointCloud(xyz=pts3d, colors=colors,
                            visibility=[set(range(len(valid_cam_idx))) for _ in range(len(pts3d))])
    stats = {
        'n_photos': N,
        'n_pairs_verified': len(graph.edges),
        'n_cameras_registered': len(valid_cam_idx),
        'n_3d_points': len(pts3d),
        'wall_seconds': time.perf_counter() - t0,
        'valid_cam_idx': valid_cam_idx,
    }
    if verbose:
        print(f"  [F.11] done in {stats['wall_seconds']:.2f}s ({len(valid_cam_idx)}/{N} cams, {len(pts3d)} pts)")
    return out_cams, out_cloud, stats


# ---------------- F.11.3 LUD translation averaging ----------------
# Ozyesil & Singer 2015 — proper joint translation solve.
#
# Variables: C_i ∈ R^3 for each placed camera (in world frame).
# Per-edge constraint: (C_j - C_i) is parallel to v_ij = R_i^T @ t_ij_unit.
# Residual: r_ij = (I - v_ij v_ij^T) (C_j - C_i)  (3-vector)
# That's the component of (C_j - C_i) PERPENDICULAR to the predicted direction.
# Zero iff parallel (which is what we want).
#
# Gauge fixing: pin C_root = (0, 0, 0). Scale anchor: pin |C_anchor - C_root| = 1
# where anchor is the camera with the strongest edge to root.
#
# Solve as sparse LSQ via scipy.sparse.linalg.lsqr.

def lud_translation_refine(R_abs, C_init, graph, n_iters=3, verbose=False):
    """LUD-style joint translation refinement.

    Args:
        R_abs:  per-cam absolute rotations (None for unplaced).
        C_init: per-cam initial centers (None for unplaced).
        graph:  ViewGraph with verified edges.
        n_iters: outer LSQ iterations (re-linearize v_ij at each).
    Returns refined C_abs.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    valid = [i for i, (R, C) in enumerate(zip(R_abs, C_init)) if R is not None and C is not None]
    if len(valid) < 2:
        return C_init
    cam_to_local = {c: idx for idx, c in enumerate(valid)}
    M = len(valid)
    root = valid[0]
    # Pick scale anchor: cam with strongest edge to root
    anchor = None; best_w = -1
    for (i, j), e in graph.edges.items():
        if i == root and j in cam_to_local:
            if e['n_inliers'] > best_w: best_w = e['n_inliers']; anchor = j
        elif j == root and i in cam_to_local:
            if e['n_inliers'] > best_w: best_w = e['n_inliers']; anchor = i
    if anchor is None:
        return C_init
    # Anchor distance: keep what spanning-tree init gave us
    anchor_dist = float(np.linalg.norm(C_init[anchor] - C_init[root]))
    if anchor_dist < 1e-6: anchor_dist = 1.0

    C = [c.copy() if c is not None else None for c in C_init]

    for it in range(n_iters):
        # Build sparse system. Each edge contributes 3 rows; gauge constraints add 6.
        edges_list = []
        for (i, j), e in graph.edges.items():
            if i not in cam_to_local or j not in cam_to_local: continue
            R_i = R_abs[i].astype(np.float64)
            t_ij = e['t'].astype(np.float64)
            v_ij = R_i.T @ t_ij                    # world-space direction from cam_i to cam_j
            v_ij = v_ij / (np.linalg.norm(v_ij) + 1e-12)
            edges_list.append((cam_to_local[i], cam_to_local[j], v_ij))
        if not edges_list:
            return C

        n_eq = len(edges_list)
        rows = []; cols = []; vals = []
        # Each edge row: r = (I - v v^T)(C_j - C_i) = 0
        # Coefficient on C_j is +(I - v v^T), on C_i is -(I - v v^T)
        for k, (li, lj, v) in enumerate(edges_list):
            P = np.eye(3) - np.outer(v, v)         # 3x3 projection
            for ax in range(3):
                for c in range(3):
                    if abs(P[ax, c]) > 1e-12:
                        rows.append(3*k + ax); cols.append(3*lj + c); vals.append(P[ax, c])
                        rows.append(3*k + ax); cols.append(3*li + c); vals.append(-P[ax, c])
        # Gauge: pin C_root = 0 (3 equations, big weight)
        big = 1.0e6
        gauge_rows = []
        li_root = cam_to_local[root]
        for ax in range(3):
            r = 3 * n_eq + ax
            rows.append(r); cols.append(3*li_root + ax); vals.append(big)
            gauge_rows.append(r)
        # Scale anchor: pin distance from root to anchor along its current direction = anchor_dist
        # (3 equations: C_anchor = anchor_dist * direction, where direction is current unit vec)
        li_anchor = cam_to_local[anchor]
        anchor_vec = (C[anchor] - C[root])
        anchor_vec = anchor_vec / (np.linalg.norm(anchor_vec) + 1e-12)
        for ax in range(3):
            r = 3 * n_eq + 3 + ax
            rows.append(r); cols.append(3*li_anchor + ax); vals.append(big)
            gauge_rows.append(r)

        n_rows = 3 * n_eq + 6
        A = csr_matrix((vals, (rows, cols)), shape=(n_rows, 3 * M))

        b = np.zeros(n_rows, dtype=np.float64)
        # Edges: target is 0 (constraint already)
        # Gauge root: target is 0 (already)
        # Gauge anchor: target is anchor_dist * direction
        for ax in range(3):
            b[3 * n_eq + 3 + ax] = big * anchor_dist * anchor_vec[ax]

        sol = lsqr(A, b, atol=1e-9, btol=1e-9)[0]
        new_C = sol.reshape(M, 3)

        # Update C
        max_step = 0.0
        for ci in valid:
            li = cam_to_local[ci]
            old = C[ci].astype(np.float64)
            new = new_C[li].astype(np.float64)
            max_step = max(max_step, float(np.linalg.norm(new - old)))
            C[ci] = new
        if verbose:
            print(f"  [F.11.3] LUD iter {it}: max C step = {max_step:.5f}")
        if max_step < 1e-5:
            break

    return C


# Wire LUD into run_sfm_global. Add a flag so we can A/B vs spanning-tree only.
_ORIG_run_sfm_global = run_sfm_global


def run_sfm_global_lud(photos, intr=None, fov_deg_prior=50.0,
                        n_features=2500, ratio=0.85, min_matches=20,
                        ess_min_inliers=10, rot_iters=15, lud_iters=3,
                        run_global_ba=True, ba_max_nfev=200, verbose=True):
    """Same as run_sfm_global but with LUD translation refinement after spanning tree."""
    t0 = time.perf_counter()
    if intr is None:
        H, W = photos.photos[0].height, photos.photos[0].width
        intr = CameraIntrinsics.from_fov(fov_deg_prior, W, H)
    K = intr.K
    N = len(photos)

    if verbose:
        print(f"  [F.11] {N} photos, intrinsics K=\n{K}")
    feats = detect_features(photos, n_features=n_features, verbose=False)
    kp_xy = keypoints_to_xy(feats)

    if verbose: print(f"  [F.11] building view graph ...")
    graph = build_view_graph(feats, kp_xy, K, ratio=ratio, min_matches=min_matches,
                              ess_min_inliers=ess_min_inliers, verbose=verbose)
    if not graph.edges:
        raise RuntimeError("Empty view graph.")

    if verbose: print(f"  [F.11] spanning-tree rotation init ...")
    R_abs = spanning_tree_rotation_init(graph)

    if verbose: print(f"  [F.11] linear rotation refinement ({rot_iters} iters) ...")
    R_abs = refine_rotations_linear(R_abs, graph, n_iters=rot_iters, verbose=False)

    if verbose: print(f"  [F.11] spanning-tree translation init ...")
    C_init = estimate_translations(R_abs, graph, kp_xy, K, verbose=False)

    if verbose: print(f"  [F.11.3] LUD translation refinement ({lud_iters} iters) ...")
    C_abs = lud_translation_refine(R_abs, C_init, graph, n_iters=lud_iters, verbose=verbose)

    if verbose: print(f"  [F.11] global triangulation ...")
    pts3d, colors, tracks, track_lookup, cam_idx, extrinsics_tuples = global_triangulate(
        graph, R_abs, C_abs, kp_xy, photos, K, verbose=verbose)

    final_ext = [CameraExtrinsics(R=R, t=t) for (R, t) in extrinsics_tuples]
    valid_cam_idx = cam_idx

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
            print(f"  [F.11+] global sparse BA: {len(final_ext)} cams, {len(pts3d)} pts, {n_obs} obs ...")
        try:
            t_ba = time.perf_counter()
            final_ext, pts3d, ba_cost, ba_nfev = bundle_adjust_sparse(
                intr, final_ext, pts3d, obs_per_cam, max_nfev=ba_max_nfev, verbose=False,
            )
            if verbose:
                print(f"  [F.11+] BA done in {time.perf_counter()-t_ba:.2f}s, cost={ba_cost:.2f}, nfev={ba_nfev}")
        except Exception as e:
            if verbose:
                print(f"  [F.11+] BA failed: {e}")

    out_cams = CameraBundle(intrinsics=intr, extrinsics=final_ext)
    out_cloud = PointCloud(xyz=pts3d, colors=colors,
                            visibility=[set(range(len(valid_cam_idx))) for _ in range(len(pts3d))])
    stats = {
        'n_photos': N,
        'n_pairs_verified': len(graph.edges),
        'n_cameras_registered': len(valid_cam_idx),
        'n_3d_points': len(pts3d),
        'wall_seconds': time.perf_counter() - t0,
        'valid_cam_idx': valid_cam_idx,
    }
    if verbose:
        print(f"  [F.11+LUD] done in {stats['wall_seconds']:.2f}s ({len(valid_cam_idx)}/{N} cams, {len(pts3d)} pts)")
    return out_cams, out_cloud, stats


# Expose LUD as the default global SfM
run_sfm_global = run_sfm_global_lud
