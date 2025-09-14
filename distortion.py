# statvis/distortion.py
"""
StatVis Distortion & Correction Module
======================================

Implements density-mismatch diagnostics between the high-dimensional (HD) space
and the low-dimensional (LD) embedding, and produces *visual overlay* corrections
(opacity/contour adjustments) **without altering embedding coordinates**.

This module operationalizes the paper's "Density-Aware Projection Correction"
as a rendering-layer enhancement.

Key Concepts
------------
- rho_hd: local density in HD (kNN or KDE)
- rho_ld: local density in LD (kNN or KDE)
- delta:  density mismatch signal between LD and HD (difference or ratio)
- overlays: derived visual encodings (opacity, line width) from delta

Primary API
-----------
- compute_density_mismatch(...): returns rho_hd, rho_ld, delta
- derive_overlays(delta, alpha): maps delta -> opacity/contour width
- mismatch_grid(...): gridified mismatch heatmap for figure contours
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Optional
import numpy as np

from .density import knn_density, kde_density, normalize_density, density_mismatch, grid_kde


DensityMethod = Literal["knn", "kde"]
MismatchMethod = Literal["difference", "ratio"]


@dataclass
class DistortionConfig:
    """Configuration for density-based distortion diagnostics."""
    k: int = 15                            # neighbors for kNN density
    metric: str = "euclidean"              # distance metric for kNN (HD/LD)
    density_method: DensityMethod = "knn"  # 'knn' or 'kde'
    mismatch: MismatchMethod = "difference"  # 'difference' or 'ratio'
    bw_method: Literal["scott","silverman",float] = "scott"  # KDE bandwidth if used


def _estimate_density(X: np.ndarray, space: str, cfg: DistortionConfig) -> np.ndarray:
    if cfg.density_method == "knn":
        return knn_density(X, k=cfg.k, metric=cfg.metric)
    elif cfg.density_method == "kde":
        return kde_density(X, bw_method=cfg.bw_method)
    else:
        raise ValueError("density_method must be 'knn' or 'kde'")


def compute_density_mismatch(
    X_hd: np.ndarray,
    Y_ld: np.ndarray,
    cfg: Optional[DistortionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local densities in HD and LD, and their mismatch signal.

    Parameters
    ----------
    X_hd : (n, d_hd) array
        High-dimensional features.
    Y_ld : (n, d_ld) array
        Low-dimensional embedding (2D or 3D supported for kNN; KDE grid is 2D).
    cfg : DistortionConfig or None
        Configuration; uses defaults if None.

    Returns
    -------
    rho_hd : (n,) array
    rho_ld : (n,) array
    delta  : (n,) array
        Density mismatch; positive => LD appears denser than HD (for 'difference').
    """
    cfg = cfg or DistortionConfig()

    rho_hd = _estimate_density(X_hd, "HD", cfg)
    rho_ld = _estimate_density(Y_ld, "LD", cfg)

    delta = density_mismatch(rho_hd, rho_ld, method=cfg.mismatch)
    return rho_hd, rho_ld, delta


def derive_overlays(
    delta: np.ndarray,
    alpha: float = 0.6,
    mode: Literal["difference","ratio"] = "difference",
) -> Dict[str, np.ndarray]:
    """
    Map density mismatch signal into *visual overlay encodings*.

    For 'difference' deltas in [-1,1]:
      - regions where LD is denser than HD (delta>0) become more transparent
        to reduce false impression of separation
      - regions where LD is sparser than HD (delta<0) become slightly sharper

    For 'ratio', values are converted to symmetric difference via min-max first.

    Parameters
    ----------
    delta : (n,) array
        Density mismatch values.
    alpha : float in [0,1]
        Correction strength; higher => stronger visual effect.
    mode : {'difference','ratio'}
        Interpret delta as difference or ratio.

    Returns
    -------
    dict with arrays (n,) for per-point encodings:
        - 'opacity'         in [0,1]  (lower => more transparent)
        - 'contour_width'   in [wmin,wmax] (normalized proxy)
        - 'highlight'       binary mask for top-|delta| regions
    """
    eps = 1e-12
    d = delta.copy()
    if mode == "ratio":
        # Convert ratios to symmetric [-1,1] measure
        # r>1 (LD denser), r<1 (LD sparser)
        r = d
        r_norm = (r - r.min()) / (r.max() - r.min() + eps)
        d = 2 * r_norm - 1  # map to [-1,1]

    # Normalize to [-1,1] if not already
    d = np.clip(d, -1.0, 1.0)

    # Opacity: reduce where LD looks too dense vs HD (d>0)
    base_opacity = 1.0 - alpha * np.maximum(0.0, d) * 0.8
    opacity = np.clip(base_opacity, 0.15, 1.0)

    # Contour width: thicker where LD is sparser than HD (d<0) to emphasize cohesion
    wmin, wmax = 0.5, 2.0
    contour_width = wmin + (wmax - wmin) * np.maximum(0.0, -d)

    # Highlight mask for most discrepant regions (top 10% |d|)
    th = np.quantile(np.abs(d), 0.90)
    highlight = (np.abs(d) >= th).astype(np.uint8)

    return {
        "opacity": opacity.astype(float),
        "contour_width": contour_width.astype(float),
        "highlight": highlight,
        "delta_symmetric": d.astype(float),
    }


def mismatch_grid(
    Y_ld: np.ndarray,
    delta: np.ndarray,
    grid_size: int = 200,
    padding: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate per-point mismatch values onto a 2D grid for heatmap/contours.

    Uses a KDE over the *coordinates only* to get a smooth field; then weights that
    field by the average of delta in local neighborhoods (simple Shepard-like scheme).

    Parameters
    ----------
    Y_ld : (n, 2) array
        2D embedding.
    delta : (n,) array
        Symmetric mismatch signal in [-1,1].
    grid_size : int
        Grid resolution per axis.
    padding : float
        Fractional padding around data bounds.

    Returns
    -------
    Xg, Yg, Z : (grid_size, grid_size)
        Grid coordinates and a signed heatmap Z ≈ E[delta | position].
    """
    if Y_ld.shape[1] != 2:
        raise ValueError("mismatch_grid currently supports 2D embeddings only")

    # Build position KDE (unweighted) for smoothing
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(Y_ld.T, bw_method="scott")

    xmin, ymin = Y_ld.min(axis=0)
    xmax, ymax = Y_ld.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    coords = np.vstack([Xg.ravel(), Yg.ravel()])

    # Evaluate density at grid
    base = kde.evaluate(coords) + 1e-12

    # Shepard-like weighting: accumulate contributions from points
    # using Gaussian kernels around each point with bandwidth from KDE.
    # For speed and simplicity we approximate with nearest neighbor weighting.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=30).fit(Y_ld)
    # Sample grid points sparsely to limit memory
    Z = np.zeros_like(base)
    dsum = np.zeros_like(base)

    # Batch the grid for memory safety
    batch = 5000
    for start in range(0, coords.shape[1], batch):
        sl = slice(start, start + batch)
        pts = coords[:, sl].T
        dists, idx = nn.kneighbors(pts)  # (b, k)
        # Weight by inverse distance (add small epsilon)
        w = 1.0 / (dists + 1e-6)
        wnorm = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        # Gather delta
        deltas = delta[idx]  # (b, k)
        z_part = (wnorm * deltas).sum(axis=1)
        Z[sl] = z_part
        dsum[sl] = 1.0  # indicator

    Z = (Z / (dsum + 1e-12)).reshape(Xg.shape)
    return Xg, Yg, Z
