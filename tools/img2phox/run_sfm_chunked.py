"""Resumable chunked SfM with proper bundle adjustment wired in.

Stages:
  features    -- detect ORB features
  match       -- match pairs (resumable)
  verify      -- essential-matrix verify (resumable)
  pose        -- solve rotations + translations + triangulate, build observations_per_cam
  ba          -- global sparse bundle adjustment using observations
  all         -- run them all in order
"""
from __future__ import annotations
import sys, os, pickle, time, argparse
sys.path.insert(0, 'tools')

import numpy as np
import cv2
from pathlib import Path

from img2phox.load_photos import load_photoset
from img2phox.preprocess import preprocess_photoset
from img2phox.data_classes import PhotoSet, CameraExtrinsics, CameraBundle, PointCloud
from img2phox.sfm_real import (detect_features, keypoints_to_xy, verify_pair_essential,
                                  triangulate)
from img2phox.sfm_global import (ViewGraph, spanning_tree_rotation_init,
                                    refine_rotations_linear, lud_translation_refine,
                                    estimate_translations)


def atomic_pkl_save(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    with open(tmp, 'rb') as f_in, open(path, 'wb') as f_out:
        f_out.write(f_in.read())


def stage_features(args):
    N = args.n_views
    feat_path = f'outputs/_sfm_N{N}_features.pkl'
    if os.path.exists(feat_path) and os.path.getsize(feat_path) > 0:
        print(f'features already exist at {feat_path}', flush=True); return
    t0 = time.perf_counter()
    photoset = load_photoset(Path(args.photos), max_dim=args.max_dim)
    idxs = np.linspace(0, len(photoset)-1, N).astype(int)
    photoset = PhotoSet(photos=[photoset.photos[i] for i in idxs])
    photoset, intr = preprocess_photoset(photoset, fov_deg_fallback=50.0,
                                            exposure_method=None, verbose=False)
    print(f'  load+preproc: {time.perf_counter()-t0:.1f}s', flush=True)
    feats = detect_features(photoset, n_features=args.n_features, verbose=False)
    kp_xy = keypoints_to_xy(feats)
    descs = [f.descs for f in feats]
    print(f'  features: {time.perf_counter()-t0:.1f}s', flush=True)
    atomic_pkl_save(feat_path, {'photoset': photoset, 'intr': intr,
                                  'kp_xy': kp_xy, 'descs': descs, 'N': N})
    print(f'  -> {feat_path}', flush=True)


def stage_match(args):
    N = args.n_views
    feat_path = f'outputs/_sfm_N{N}_features.pkl'
    match_path = f'outputs/_sfm_N{N}_matches.pkl'
    with open(feat_path, 'rb') as f: fd = pickle.load(f)
    descs = fd['descs']

    if os.path.exists(match_path) and os.path.getsize(match_path) > 0:
        with open(match_path, 'rb') as f: existing = pickle.load(f)
        matches = existing['matches']
        done_pairs = set(tuple(p) for p in existing.get('done_pairs', list(matches.keys())))
    else:
        matches, done_pairs = {}, set()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, min(N, i+1+args.window))]
    todo = [p for p in all_pairs if p not in done_pairs]
    print(f'pairs to match: {len(todo)} / total {len(all_pairs)}', flush=True)

    t0 = time.perf_counter()
    for k, (i, j) in enumerate(todo):
        if time.perf_counter() - t0 > args.time_budget:
            print(f'  budget hit at {k}/{len(todo)}', flush=True); break
        done_pairs.add((i, j))
        if len(descs[i]) < 8 or len(descs[j]) < 8: continue
        knn = matcher.knnMatch(descs[i], descs[j], k=2)
        good = [[mp[0].queryIdx, mp[0].trainIdx] for mp in knn
                  if len(mp) >= 2 and mp[0].distance < args.ratio * mp[1].distance]
        if len(good) >= args.min_matches:
            matches[(i, j)] = np.asarray(good, dtype=np.int32)
        if (k+1) % 50 == 0 or k == len(todo)-1:
            atomic_pkl_save(match_path, {'matches': matches, 'done_pairs': list(done_pairs)})
    atomic_pkl_save(match_path, {'matches': matches, 'done_pairs': list(done_pairs)})
    print(f'  -> {match_path} ({len(matches)} pairs)', flush=True)


def stage_verify(args):
    N = args.n_views
    feat_path = f'outputs/_sfm_N{N}_features.pkl'
    match_path = f'outputs/_sfm_N{N}_matches.pkl'
    graph_path = f'outputs/_sfm_N{N}_graph.pkl'
    with open(feat_path, 'rb') as f: fd = pickle.load(f)
    with open(match_path, 'rb') as f: md = pickle.load(f)
    kp_xy = fd['kp_xy']; intr = fd['intr']; matches = md['matches']
    K = intr.K

    if os.path.exists(graph_path) and os.path.getsize(graph_path) > 0:
        with open(graph_path, 'rb') as f: gd = pickle.load(f)
        g = gd['graph']
        verified_pairs = set(tuple(p) for p in gd.get('verified_pairs', list(g.edges.keys())))
    else:
        g = ViewGraph(N); verified_pairs = set()

    todo = [(i, j) for (i, j) in matches.keys() if (i, j) not in verified_pairs]
    print(f'pairs to verify: {len(todo)}', flush=True)

    t0 = time.perf_counter()
    for k, (i, j) in enumerate(todo):
        if time.perf_counter() - t0 > args.time_budget:
            print(f'  budget hit at {k}/{len(todo)}', flush=True); break
        verified_pairs.add((i, j))
        m = matches[(i, j)]
        R, t, mask, n_inl = verify_pair_essential(kp_xy[i], kp_xy[j], m, K)
        if R is None or n_inl < 8: continue
        g.add_edge(i, j, R, t, mask, m)
        if (k+1) % 25 == 0 or k == len(todo)-1:
            g._pair_matches = matches
            atomic_pkl_save(graph_path, {'graph': g, 'verified_pairs': list(verified_pairs)})
    g._pair_matches = matches
    atomic_pkl_save(graph_path, {'graph': g, 'verified_pairs': list(verified_pairs)})
    print(f'  -> {graph_path} ({len(g.edges)} edges)', flush=True)


def _do_global_triangulation(g, R_abs, C_abs, kp_xy, photoset, K, cheir_thresh=0.01):
    extr_list = []
    for ci, (R, C) in enumerate(zip(R_abs, C_abs)):
        if R is None or C is None: extr_list.append(None); continue
        t = -R.astype(np.float64) @ C
        extr_list.append((R.astype(np.float32), t.astype(np.float32)))

    pts3d_list = []; color_list = []; tracks = []
    track_lookup = {ci: {} for ci in range(len(R_abs)) if R_abs[ci] is not None}

    for (i, j), e in g.edges.items():
        if extr_list[i] is None or extr_list[j] is None: continue
        Ri, ti = extr_list[i]; Rj, tj = extr_list[j]
        m = e['matches']; mask = e['mask']
        in_m = m[mask]
        if len(in_m) < 2: continue
        pts0 = kp_xy[i][in_m[:, 0]]
        pts1 = kp_xy[j][in_m[:, 1]]
        new_pts = triangulate(K, Ri, ti, Rj, tj, pts0, pts1)
        zi = (new_pts @ Ri.T + ti)[:, 2]
        zj = (new_pts @ Rj.T + tj)[:, 2]
        keep = (zi > cheir_thresh) & (zj > cheir_thresh) & (np.linalg.norm(new_pts, axis=1) < 100.0)
        for k in range(len(new_pts)):
            if not keep[k]: continue
            ai = int(in_m[k, 0]); bi = int(in_m[k, 1])
            if ai in track_lookup[i] or bi in track_lookup[j]: continue
            pid = len(pts3d_list)
            pts3d_list.append(new_pts[k])
            x, y = pts0[k]
            xi = int(np.clip(x, 0, photoset.photos[i].width - 1))
            yi = int(np.clip(y, 0, photoset.photos[i].height - 1))
            color_list.append(photoset.photos[i].image[yi, xi])
            tracks.append({i: ai, j: bi})
            track_lookup[i][ai] = pid
            track_lookup[j][bi] = pid
    pts = np.asarray(pts3d_list, dtype=np.float32) if pts3d_list else np.zeros((0,3), dtype=np.float32)
    colors = np.asarray(color_list, dtype=np.float32) if color_list else np.zeros((0,3), dtype=np.float32)
    obs = [[] for _ in range(len(R_abs))]
    for pid, t in enumerate(tracks):
        for ci, kpi in t.items():
            u, v = kp_xy[ci][kpi]
            obs[ci].append((pid, float(u), float(v)))
    return pts, colors, tracks, obs


def stage_pose(args):
    N = args.n_views
    feat_path = f'outputs/_sfm_N{N}_features.pkl'
    graph_path = f'outputs/_sfm_N{N}_graph.pkl'
    final_path = f'outputs/_sfm_cache_N{N}.pkl'
    with open(feat_path, 'rb') as f: fd = pickle.load(f)
    with open(graph_path, 'rb') as f: gd = pickle.load(f)
    photoset = fd['photoset']; intr = fd['intr']; kp_xy = fd['kp_xy']
    g = gd['graph']
    K = intr.K
    print(f'graph: {len(g.edges)} edges, {N} cams', flush=True)

    t1 = time.perf_counter()
    R_init = spanning_tree_rotation_init(g)
    R_abs = refine_rotations_linear(R_init, g, n_iters=10, verbose=False)
    print(f'  rotation: {time.perf_counter()-t1:.1f}s', flush=True)

    t1 = time.perf_counter()
    C_init = estimate_translations(R_abs, g, kp_xy, K, verbose=False)
    C_abs = lud_translation_refine(R_abs, C_init, g, n_iters=3, verbose=False)
    print(f'  translation: {time.perf_counter()-t1:.1f}s', flush=True)

    t1 = time.perf_counter()
    pts, colors, tracks, observations = _do_global_triangulation(g, R_abs, C_abs, kp_xy, photoset, K)
    n_obs = sum(len(o) for o in observations)
    print(f'  triangulate: {len(pts)} points, {n_obs} observations, {time.perf_counter()-t1:.1f}s',
          flush=True)

    extrinsics = []
    for i in range(N):
        if R_abs[i] is None or C_abs[i] is None:
            extrinsics.append(None)
        else:
            R = R_abs[i]; t = -R @ C_abs[i]
            extrinsics.append(CameraExtrinsics(R=R.astype(np.float32), t=t.astype(np.float32)))
    valid = [(i, e) for i, e in enumerate(extrinsics) if e is not None]
    rec_cams = CameraBundle(intrinsics=intr, extrinsics=[e for _, e in valid])
    valid_idx = [i for i, _ in valid]
    valid_photoset = PhotoSet(photos=[photoset.photos[i] for i in valid_idx])
    print(f'  registered: {len(valid)}/{N}', flush=True)

    obs_remapped = [observations[old] for old in valid_idx]
    sparse = PointCloud(xyz=pts, colors=colors)
    atomic_pkl_save(final_path, {
        'photoset': valid_photoset, 'rec_cams': rec_cams,
        'sparse': sparse, 'observations': obs_remapped,
        'tracks': tracks, 'valid_idx': valid_idx, 'n_views': N,
    })
    print(f'  -> {final_path}', flush=True)


def stage_ba(args):
    from img2phox.sfm import bundle_adjust_sparse
    N = args.n_views
    final_path = f'outputs/_sfm_cache_N{N}.pkl'
    with open(final_path, 'rb') as f: c = pickle.load(f)
    rec_cams = c['rec_cams']; sparse = c['sparse']; obs = c.get('observations')
    if obs is None:
        print('no observations — re-run pose first', flush=True); return
    print(f'BA on {len(rec_cams)} cams, {len(sparse)} pts, {sum(len(o) for o in obs)} obs',
          flush=True)
    if len(sparse) < 8:
        print(f'too few points, skipping', flush=True); return
    t0 = time.perf_counter()
    result = bundle_adjust_sparse(rec_cams.intrinsics, list(rec_cams.extrinsics),
                                     sparse.xyz, obs,
                                     max_nfev=args.ba_nfev, verbose=False)
    print(f'  BA done in {time.perf_counter()-t0:.1f}s, return type={type(result)}', flush=True)
    new_extr = result[0]; new_pts = result[1]
    rec_cams2 = CameraBundle(intrinsics=rec_cams.intrinsics, extrinsics=list(new_extr))
    sparse2 = PointCloud(xyz=np.asarray(new_pts, dtype=np.float32), colors=sparse.colors)
    c['rec_cams'] = rec_cams2; c['sparse'] = sparse2
    atomic_pkl_save(final_path, c)
    print(f'  -> {final_path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-views', type=int, default=30)
    p.add_argument('--max-dim', type=int, default=320)
    p.add_argument('--n-features', type=int, default=2000)
    p.add_argument('--ratio', type=float, default=0.85)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--min-matches', type=int, default=15)
    p.add_argument('--photos', type=str, default='inputs/Family')
    p.add_argument('--time-budget', type=float, default=30.0)
    p.add_argument('--ba-nfev', type=int, default=100)
    p.add_argument('--mode', choices=['features','match','verify','pose','ba','all'], default='all')
    args = p.parse_args()
    if args.mode in ('features','all'): stage_features(args)
    if args.mode in ('match','all'):    stage_match(args)
    if args.mode in ('verify','all'):   stage_verify(args)
    if args.mode in ('pose','all'):     stage_pose(args)
    if args.mode in ('ba','all'):       stage_ba(args)


if __name__ == '__main__':
    main()
