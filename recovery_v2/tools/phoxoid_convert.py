#!/usr/bin/env python3
"""
phoxoid_convert.py

Experimental Crypsara/Crypsoid converter:
- Reads an ASCII PLY point cloud / mesh / Gaussian-splat-like PLY with x,y,z vertex properties.
- Fits a baseline ellipsoidal Gaussian-style blob model.
- Fits a phoxoidal chart-germ model that replaces fixed quadratic ellipsoidal shape
  with a soft exponential caustic/surface-germ action.
- Exports .phox.json and benchmark JSON.

This is v0 research tooling, not a renderer. Its purpose is to make the proposed
phoxoidal primitive measurable against an ellipsoidal/flat local baseline.

Usage:
    python phoxoid_convert.py bunny.ply.txt --out bunny.phox.json --blobs 512 --benchmark-out bunny.phox_benchmark.json

Gaussian-splat-like PLYs:
    If a PLY contains properties such as opacity, scale_*, rot_*, f_dc_*, or colors,
    the converter preserves aggregate summaries in each phoxoidal blob's evidence block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# -----------------------------
# PLY parsing
# -----------------------------

@dataclass
class PlyData:
    path: str
    vertex_properties: List[str]
    vertices: np.ndarray
    faces: Optional[List[List[int]]]
    comments: List[str]
    header: List[str]


def read_ascii_ply(path: str | Path) -> PlyData:
    """Read a simple ASCII PLY file with vertex and face elements.

    The parser is intentionally small and dependency-free. It supports:
    - format ascii 1.0
    - vertex scalar properties
    - face list property lines beginning with a count
    """
    path = Path(path)
    header: List[str] = []
    comments: List[str] = []
    vertex_count = None
    face_count = 0
    vertex_props: List[str] = []
    in_vertex = False
    in_face = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if line == "":
                raise ValueError("Unexpected EOF before end_header")
            line = line.strip()
            header.append(line)
            if line.startswith("comment "):
                comments.append(line[len("comment "):])
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                if len(parts) < 2 or parts[1] != "ascii":
                    raise ValueError(f"Only ASCII PLY is supported by this v0 parser. Got: {line}")
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                in_face = parts[1] == "face"
                if in_vertex:
                    vertex_count = int(parts[2])
                elif in_face:
                    face_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                # property float32 x
                # property uchar red
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_props.append(parts[-1])

        if vertex_count is None:
            raise ValueError("PLY missing element vertex")

        verts = np.empty((vertex_count, len(vertex_props)), dtype=np.float64)
        for i in range(vertex_count):
            line = f.readline()
            if line == "":
                raise ValueError(f"Unexpected EOF reading vertex {i}/{vertex_count}")
            vals = line.strip().split()
            if len(vals) < len(vertex_props):
                raise ValueError(f"Vertex line {i} has {len(vals)} values, expected {len(vertex_props)}")
            verts[i] = [float(v) for v in vals[:len(vertex_props)]]

        faces: List[List[int]] = []
        for _ in range(face_count):
            line = f.readline()
            if line == "":
                break
            vals = line.strip().split()
            if not vals:
                continue
            n = int(vals[0])
            faces.append([int(v) for v in vals[1:1+n]])

    return PlyData(
        path=str(path),
        vertex_properties=vertex_props,
        vertices=verts,
        faces=faces if faces else None,
        comments=comments,
        header=header,
    )


# -----------------------------
# Utility math
# -----------------------------

def _as_list(x: np.ndarray | float, ndigits: int = 9) -> Any:
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), ndigits).tolist()
    if isinstance(x, (float, np.floating)):
        return round(float(x), ndigits)
    return x


def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w)) + 1e-12
    return np.sum(X * w[:, None], axis=0) / s


def weighted_cov(X: np.ndarray, w: np.ndarray, center: np.ndarray) -> np.ndarray:
    Y = X - center
    s = float(np.sum(w)) + 1e-12
    C = (Y * w[:, None]).T @ Y / s
    return C + np.eye(X.shape[1]) * 1e-12


def weighted_rms(x: np.ndarray, w: np.ndarray) -> float:
    return float(math.sqrt(np.sum(w * x * x) / (np.sum(w) + 1e-12)))


def robust_std(x: np.ndarray, w: np.ndarray) -> float:
    mu = float(np.sum(w * x) / (np.sum(w) + 1e-12))
    return float(math.sqrt(np.sum(w * (x - mu) ** 2) / (np.sum(w) + 1e-12)) + 1e-12)


def ridge_lstsq(A: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    sw = np.sqrt(np.maximum(w, 1e-12))
    Aw = A * sw[:, None]
    yw = y * sw
    ATA = Aw.T @ Aw
    ATy = Aw.T @ yw
    scale = float(np.trace(ATA) / max(1, ATA.shape[0]))
    return np.linalg.solve(ATA + np.eye(ATA.shape[0]) * ridge * (scale + 1e-12), ATy)


def phoxoidal_basis(s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Basis for the v0 phoxoidal germ H(s,t).

    The basis deliberately includes:
    - affine leakage terms for robustness,
    - quadratic curvature,
    - harmonic cubic caustic/cusp modes,
    - quartic support/lensing mode.

    H(s,t) = dot(coeffs, basis(s,t))
    """
    basis = [
        np.ones_like(s),
        s,
        t,
        s * s,
        s * t,
        t * t,
        s**3 - 3.0 * s * t**2,       # real cubic harmonic / cusp-twist mode
        3.0 * s**2 * t - t**3,       # imaginary cubic harmonic / swirl mode
        s**4 + t**4,                 # quartic lens/support mode
    ]
    names = [
        "bias",
        "tilt_s",
        "tilt_t",
        "curvature_ss",
        "curvature_st",
        "curvature_tt",
        "cubic_cusp_real",
        "cubic_cusp_imag",
        "quartic_lens",
    ]
    return np.stack(basis, axis=1), names


# -----------------------------
# Clustering
# -----------------------------

def kmeans_numpy(
    X: np.ndarray,
    weights: np.ndarray,
    k: int,
    iterations: int = 12,
    seed: int = 17,
    chunk: int = 4096,
) -> np.ndarray:
    """Small dependency-free weighted k-means."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    k = int(max(1, min(k, n)))

    # Weighted random init, then small jitter avoidance.
    p = np.maximum(weights, 1e-12)
    p = p / np.sum(p)
    init_idx = rng.choice(n, size=k, replace=False, p=p if np.all(np.isfinite(p)) else None)
    centers = X[init_idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        # assignment
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            D = np.sum((X[start:end, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels[start:end] = np.argmin(D, axis=1)

        # update
        new_centers = np.zeros_like(centers)
        sums = np.zeros(k, dtype=np.float64)
        for dim in range(X.shape[1]):
            np.add.at(new_centers[:, dim], labels, X[:, dim] * weights)
        np.add.at(sums, labels, weights)
        empty = sums <= 1e-12
        nonempty = ~empty
        new_centers[nonempty] /= sums[nonempty, None]
        if np.any(empty):
            # Reinitialize empty clusters to high-error points.
            # Approximation: random weighted points.
            repl = rng.choice(n, size=int(np.sum(empty)), replace=False, p=p if np.all(np.isfinite(p)) else None)
            new_centers[empty] = X[repl]
        centers = new_centers

    return labels


# -----------------------------
# Fitting
# -----------------------------

def property_summary(vertex_props: List[str], V: np.ndarray, idx: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for j, name in enumerate(vertex_props):
        if name in ("x", "y", "z"):
            continue
        vals = V[idx, j]
        mean = float(np.sum(w * vals) / (np.sum(w) + 1e-12))
        out[name] = {
            "mean": round(mean, 9),
            "std": round(robust_std(vals, w), 9),
            "min": round(float(np.min(vals)), 9),
            "max": round(float(np.max(vals)), 9),
        }
    return out


def fit_blob(
    blob_id: int,
    X: np.ndarray,
    V_all: np.ndarray,
    vertex_props: List[str],
    idx: np.ndarray,
    weights: np.ndarray,
    min_points: int = 12,
) -> Dict[str, Any]:
    P = X[idx]
    w = weights[idx].astype(np.float64)
    if len(P) < min_points:
        raise ValueError("cluster too small")

    center = weighted_mean(P, w)
    cov = weighted_cov(P, w, center)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Frame columns: tangent_s, tangent_t, normal.
    frame = eigvecs.copy()
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0

    U = (P - center) @ frame
    s, t, n = U[:, 0], U[:, 1], U[:, 2]

    gaussian_flat_rms = weighted_rms(n, w)
    gaussian_cov = cov

    A, basis_names = phoxoidal_basis(s, t)
    coeffs = ridge_lstsq(A, n, w, ridge=1e-7)
    pred = A @ coeffs
    residual = n - pred
    phox_rms = weighted_rms(residual, w)

    sigma_s = max(robust_std(s, w), 1e-9)
    sigma_t = max(robust_std(t, w), 1e-9)
    sigma_n = max(robust_std(residual, w), 1e-9)

    r_norm = np.sqrt((s / sigma_s) ** 2 + (t / sigma_t) ** 2)
    support_q95 = float(np.quantile(r_norm, 0.95))
    support_q99 = float(np.quantile(r_norm, 0.99))

    improvement = 0.0
    if gaussian_flat_rms > 1e-12:
        improvement = 1.0 - (phox_rms / gaussian_flat_rms)

    # A compact action formula for renderer/prototype consumers.
    action = {
        "kind": "phoxponential_caustic_chart_v0",
        "local_coordinates": "u = (x-center)^T frame; s=u0, t=u1, n=u2",
        "height_germ": "H(s,t)=dot(coefficients,basis)",
        "basis": basis_names,
        "density": "rho(x)=opacity*exp(-A(x)); A=((s/sigma_s)^2+(t/sigma_t)^2)+((n-H(s,t))/sigma_n)^2",
    }

    return {
        "id": int(blob_id),
        "point_count": int(len(P)),
        "center": _as_list(center),
        "frame_columns": _as_list(frame),
        "gaussian_baseline": {
            "covariance": _as_list(gaussian_cov),
            "eigenvalues_desc": _as_list(eigvals),
            "radii_sqrt_eigenvalues": _as_list(np.sqrt(np.maximum(eigvals, 0.0))),
            "flat_tangent_rms": round(float(gaussian_flat_rms), 12),
        },
        "phoxoidal_germ": {
            "kind": "caustic_surface_germ_v0",
            "basis": basis_names,
            "coefficients": _as_list(coeffs),
            "softness": {
                "sigma_s": round(float(sigma_s), 12),
                "sigma_t": round(float(sigma_t), 12),
                "sigma_n_residual": round(float(sigma_n), 12),
                "temperature_tau": 1.0,
            },
            "support": {
                "radial_q95": round(float(support_q95), 9),
                "radial_q99": round(float(support_q99), 9),
            },
            "action": action,
        },
        "evidence_summary": property_summary(vertex_props, V_all, idx, w),
        "metrics": {
            "phoxoidal_rms": round(float(phox_rms), 12),
            "gaussian_flat_rms": round(float(gaussian_flat_rms), 12),
            "relative_rms_improvement": round(float(improvement), 9),
            "mean_weight": round(float(np.mean(w)), 9),
        },
    }


def attach_neighbors(blobs: List[Dict[str, Any]], k: int = 6) -> None:
    if not blobs:
        return
    C = np.array([b["center"] for b in blobs], dtype=np.float64)
    # Local support scale proxy.
    R = np.array([
        max(
            b["phoxoidal_germ"]["softness"]["sigma_s"],
            b["phoxoidal_germ"]["softness"]["sigma_t"],
            b["gaussian_baseline"]["radii_sqrt_eigenvalues"][0],
            1e-9,
        )
        for b in blobs
    ], dtype=np.float64)

    n = len(blobs)
    for i, b in enumerate(blobs):
        d2 = np.sum((C - C[i]) ** 2, axis=1)
        order = np.argsort(d2)
        neigh = []
        for j in order[1:k+1]:
            denom = (R[i] + R[j]) ** 2 + 1e-12
            overlap = math.exp(-float(d2[j]) / denom)
            neigh.append({
                "id": int(blobs[j]["id"]),
                "center_distance": round(float(math.sqrt(d2[j])), 9),
                "soft_overlap_proxy": round(float(overlap), 9),
            })
        b["neighbors"] = neigh


def aggregate_benchmark(
    source_path: str,
    ply: PlyData,
    blobs: List[Dict[str, Any]],
    start_time: float,
) -> Dict[str, Any]:
    pc = np.array([b["point_count"] for b in blobs], dtype=np.float64)
    g = np.array([b["metrics"]["gaussian_flat_rms"] for b in blobs], dtype=np.float64)
    p = np.array([b["metrics"]["phoxoidal_rms"] for b in blobs], dtype=np.float64)
    weights = np.maximum(pc, 1.0)

    g_global = float(np.sum(weights * g) / np.sum(weights))
    p_global = float(np.sum(weights * p) / np.sum(weights))
    imp = 1.0 - p_global / (g_global + 1e-12)

    return {
        "source": str(source_path),
        "source_bytes": int(os.path.getsize(source_path)),
        "vertices": int(ply.vertices.shape[0]),
        "faces": int(len(ply.faces) if ply.faces is not None else 0),
        "vertex_properties": ply.vertex_properties,
        "blob_count": int(len(blobs)),
        "point_count_per_blob": {
            "min": int(np.min(pc)) if len(pc) else 0,
            "max": int(np.max(pc)) if len(pc) else 0,
            "mean": round(float(np.mean(pc)), 3) if len(pc) else 0,
        },
        "baseline_definition": "Gaussian-like local tangent baseline uses PCA frame and residual thickness n around n=0.",
        "phoxoidal_definition": "Phoxoidal v0 uses the same PCA frame but replaces n=0 with fitted caustic/surface germ H(s,t).",
        "weighted_global_gaussian_flat_rms": round(g_global, 12),
        "weighted_global_phoxoidal_rms": round(p_global, 12),
        "weighted_relative_rms_improvement": round(float(imp), 9),
        "runtime_seconds": round(float(time.time() - start_time), 3),
    }


def convert(
    input_path: str | Path,
    out_path: str | Path,
    benchmark_out: str | Path | None = None,
    blobs_requested: int = 512,
    kmeans_iterations: int = 12,
    seed: int = 17,
    min_points: int = 12,
) -> Dict[str, Any]:
    start = time.time()
    ply = read_ascii_ply(input_path)
    props = ply.vertex_properties
    for required in ("x", "y", "z"):
        if required not in props:
            raise ValueError(f"PLY missing required vertex property {required}")

    xi, yi, zi = [props.index(p) for p in ("x", "y", "z")]
    X = ply.vertices[:, [xi, yi, zi]].astype(np.float64)

    if "confidence" in props:
        w = ply.vertices[:, props.index("confidence")].astype(np.float64)
        w = np.clip(w, 1e-5, None)
    elif "opacity" in props:
        # For Gaussian splat PLYs, opacity is often logit-like, but we only need positive weights.
        op = ply.vertices[:, props.index("opacity")].astype(np.float64)
        w = 1.0 / (1.0 + np.exp(-np.clip(op, -20, 20)))
        w = np.clip(w, 1e-5, None)
    else:
        w = np.ones(X.shape[0], dtype=np.float64)

    k = min(int(blobs_requested), max(1, X.shape[0] // min_points))
    labels = kmeans_numpy(X, w, k=k, iterations=kmeans_iterations, seed=seed)

    phox_blobs: List[Dict[str, Any]] = []
    for bid in range(k):
        idx = np.flatnonzero(labels == bid)
        if len(idx) < min_points:
            continue
        try:
            phox_blobs.append(fit_blob(len(phox_blobs), X, ply.vertices, props, idx, w, min_points=min_points))
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue

    attach_neighbors(phox_blobs, k=6)
    bench = aggregate_benchmark(str(input_path), ply, phox_blobs, start)

    result = {
        "format": "crypsara.phoxoidal_blob.v0",
        "status": "experimental_research_artifact",
        "source": {
            "path": str(input_path),
            "ply_comments": ply.comments,
            "vertices": int(ply.vertices.shape[0]),
            "faces": int(len(ply.faces) if ply.faces is not None else 0),
            "vertex_properties": props,
        },
        "theory": {
            "primitive": "phoxoidal_blob",
            "analogy": "Gaussian splat : ellipsoidal quadratic exponent :: phoxoidal blob : phoxponential caustic-chart exponent",
            "density": "rho_i(x)=opacity_i*exp(-A_i(x)); A_i uses local chart residual to H_i(s,t) rather than fixed Mahalanobis distance alone.",
            "v0_limitation": "This file fits geometry/evidence from a PLY. It does not yet train view-dependent radiance or render visibility-sorted splats.",
        },
        "benchmark": bench,
        "blobs": phox_blobs,
    }

    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if benchmark_out:
        with Path(benchmark_out).open("w", encoding="utf-8") as f:
            json.dump(bench, f, indent=2)

    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert ASCII PLY / Gaussian-splat-like PLY into experimental phoxoidal blobs.")
    ap.add_argument("input", help="Input ASCII PLY file")
    ap.add_argument("--out", default=None, help="Output .phox.json path")
    ap.add_argument("--benchmark-out", default=None, help="Output benchmark JSON path")
    ap.add_argument("--blobs", type=int, default=512, help="Requested blob count / cluster count")
    ap.add_argument("--iters", type=int, default=12, help="K-means iterations")
    ap.add_argument("--seed", type=int, default=17, help="Random seed")
    ap.add_argument("--min-points", type=int, default=12, help="Minimum points per blob")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.out) if args.out else inp.with_suffix(".phox.json")
    bench = Path(args.benchmark_out) if args.benchmark_out else inp.with_suffix(".phox_benchmark.json")

    result = convert(
        input_path=inp,
        out_path=out,
        benchmark_out=bench,
        blobs_requested=args.blobs,
        kmeans_iterations=args.iters,
        seed=args.seed,
        min_points=args.min_points,
    )

    print(json.dumps({
        "wrote": str(out),
        "benchmark": str(bench),
        "vertices": result["benchmark"]["vertices"],
        "faces": result["benchmark"]["faces"],
        "blob_count": result["benchmark"]["blob_count"],
        "weighted_global_gaussian_flat_rms": result["benchmark"]["weighted_global_gaussian_flat_rms"],
        "weighted_global_phoxoidal_rms": result["benchmark"]["weighted_global_phoxoidal_rms"],
        "weighted_relative_rms_improvement": result["benchmark"]["weighted_relative_rms_improvement"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
