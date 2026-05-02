"""Fit Gaussian and phoxoidal blobs to a scene point cloud.

A "blob" here is a single primitive (one Gaussian or one phoxoid). The
clusters of points it explains are determined by k-means.

For each cluster:
    Gaussian fit: mean + 3x3 covariance via PCA.
    Phoxoid fit:  mean + local frame (PCA) + 5-coefficient germ (least squares).

Per-cluster RMSE is the residual of n - H_theta(s,t) for points in that cluster
(Gaussian's "germ" is implicitly H = 0, so its RMSE is just the normal-direction
spread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None


@dataclass
class GaussianBlob:
    center: np.ndarray            # (3,)
    R: np.ndarray                 # (3, 3) local frame columns = principal dirs (third = normal)
    sigma: np.ndarray             # (3,) spreads along each PCA axis (s, t, n)
    fit_rmse: float               # residual of (n_i) for points in cluster (Gaussian assumes H=0)
    n_pts: int

    def predict_z(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(s)


@dataclass
class PhoxoidBlob:
    center: np.ndarray
    R: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray             # (5,) (k1, k2, chi, omega, zeta)
    fit_rmse: float               # residual of n_i - H_theta(s_i, t_i)
    n_pts: int

    def predict_z(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        from crypsorender.math.germ import germ_eval
        return germ_eval(self.theta, s, t)


def _kmeans_clusters(pts: np.ndarray, n_blobs: int, seed: int = 0) -> np.ndarray:
    """Returns (n_pts,) cluster ids.  Uses MiniBatchKMeans for speed."""
    if MiniBatchKMeans is None:
        raise RuntimeError("scikit-learn required for clustering")
    km = MiniBatchKMeans(
        n_clusters=n_blobs,
        random_state=seed,
        batch_size=max(256, 4 * n_blobs),
        n_init=3, max_iter=20,
    )
    return km.fit_predict(pts)


def _pca_frame(cluster_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA -> (center, R, sigma).

    R columns are principal directions (largest, second, smallest variance);
    smallest-variance direction is the local "normal".
    """
    center = cluster_pts.mean(axis=0)
    centered = cluster_pts - center
    if centered.shape[0] < 3:
        # Degenerate: return identity
        R = np.eye(3, dtype=np.float32)
        sigma = np.array([0.05, 0.05, 0.01], dtype=np.float32)
        return center.astype(np.float32), R, sigma
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)            # ascending
    # We want columns in order (largest, second, smallest):
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    sigma = np.sqrt(np.clip(eigvals, 1e-8, None)).astype(np.float32)
    return center.astype(np.float32), eigvecs.astype(np.float32), sigma


def _fit_germ_5(s: np.ndarray, t: np.ndarray, n: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Per-cluster 5-coefficient germ fit.  Returns (theta, rmse)."""
    s2, t2 = s * s, t * t
    M = np.stack([
        s2,
        t2,
        s * (s2 - 3.0 * t2),
        t * (3.0 * s2 - t2),
        s2 * s2 + t2 * t2,
    ], axis=-1).astype(np.float64)              # (k, 5)
    b = n.astype(np.float64)
    if M.shape[0] < 6:
        return np.zeros(5, dtype=np.float32), float(np.sqrt((b * b).mean()))
    MtM = M.T @ M + ridge * np.eye(5)
    Mtb = M.T @ b
    theta = np.linalg.solve(MtM, Mtb)
    res = b - M @ theta
    rmse = float(np.sqrt((res * res).mean()))
    return theta.astype(np.float32), rmse


def fit_blobs(pts: np.ndarray, n_blobs: int, seed: int = 0) -> Tuple[List[GaussianBlob], List[PhoxoidBlob], float, float]:
    """Cluster pts into n_blobs and fit both Gaussian and Phoxoid models.

    Returns (gaussian_blobs, phoxoid_blobs, gaussian_total_rmse, phoxoid_total_rmse)
    where the totals are weighted RMSE across all points.
    """
    labels = _kmeans_clusters(pts, n_blobs, seed=seed)
    gauss_blobs, phox_blobs = [], []
    sq_g, sq_p, total_pts = 0.0, 0.0, 0
    for c in range(n_blobs):
        cluster = pts[labels == c]
        if cluster.shape[0] == 0:
            continue
        center, R, sigma = _pca_frame(cluster)
        # Local coords:
        local = (cluster - center) @ R          # columns of R are basis -> local = world @ R
        s = local[:, 0]; t = local[:, 1]; nrm = local[:, 2]
        # Gaussian RMSE = std of normal direction (assumes H=0)
        rmse_g = float(np.sqrt((nrm * nrm).mean()))
        gauss_blobs.append(GaussianBlob(
            center=center, R=R, sigma=sigma, fit_rmse=rmse_g, n_pts=int(cluster.shape[0]),
        ))
        # Phoxoid: fit 5-coefficient germ
        theta, rmse_p = _fit_germ_5(s, t, nrm)
        phox_blobs.append(PhoxoidBlob(
            center=center, R=R, sigma=sigma, theta=theta, fit_rmse=rmse_p,
            n_pts=int(cluster.shape[0]),
        ))
        sq_g += (nrm * nrm).sum()
        # Phoxoid residual squared sum:
        from crypsorender.math.germ import germ_eval
        H = germ_eval(theta, s, t)
        sq_p += ((nrm - H) ** 2).sum()
        total_pts += cluster.shape[0]
    if total_pts == 0:
        return gauss_blobs, phox_blobs, 0.0, 0.0
    gauss_rmse = float(np.sqrt(sq_g / total_pts))
    phox_rmse  = float(np.sqrt(sq_p / total_pts))
    return gauss_blobs, phox_blobs, gauss_rmse, phox_rmse
