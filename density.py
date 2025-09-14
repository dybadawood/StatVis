# statvis/density.py
"""
StatVis Density Module
======================

Local density estimators used by StatVis in both the high-dimensional (HD)
and low-dimensional (LD) spaces, plus utilities for comparing densities.

Provided estimators
-------------------
- kNN density (inverse average distance to k nearest neighbors)
- Gaussian KDE density (via scipy.stats.gaussian_kde)

Utilities
---------
- normalize_density: safe min-max normalization to [0,1]
- density_mismatch: pointwise mismatch between HD and LD densities
- grid_kde: evaluate KDE on a regular grid for contour overlays

Notes
-----
- kNN density is fast and robust; recommended default for large n.
- KDE can be more expressive but is slower in high dimensions.
- All functions are NumPy-friendly and stateless.
"""

from __future__ import annotations

from typing import Tuple, Literal, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde


def knn_density(X: np.ndarray, k: int = 15, metric: str = "euclidean") -> np.ndarray:
    """
    Estimate local density using inverse average distance to k nearest neighbors.

    Parameters
    ----------
    X : (n_samples, n_features) array
    k : int
        Number of neighbors (excluding self). Typical values: 10–50.
    metric : str
        Distance metric for NearestNeighbors.

    Returns
    -------
    rho : (n_samples,) array
        Density proxy; higher is denser. Non-negative.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= X.shape[0]:
        raise ValueError("k must be < n_samples")

    nn = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
    dists, _ = nn.kneighbors(X)  # (n, k+1) including self at col 0
    avg = dists[:, 1:].mean(axis=1)
    with np.errstate(divide="ignore"):
        rho = 1.0 / (avg + 1e-12)
    return rho


def kde_density(X: np.ndarray, bw_method: Literal["scott","silverman",float]="scott") -> np.ndarray:
    """
    Estimate local density via Gaussian KDE.

    Parameters
    ----------
    X : (n_samples, n_features) array
        Data matrix.
    bw_method : {'scott','silverman', float}
        Bandwidth rule or scalar factor.

    Returns
    -------
    rho : (n_samples,) array
        KDE density evaluated at each sample.
    """
    kde = gaussian_kde(X.T, bw_method=bw_method)
    return kde.evaluate(X.T)


def normalize_density(rho: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max normalize densities to [0,1] with numerical safety.
    """
    lo = float(np.min(rho))
    hi = float(np.max(rho))
    if hi - lo < eps:
        return np.zeros_like(rho)
    return (rho - lo) / (hi - lo + eps)


def density_mismatch(
    rho_hd: np.ndarray,
    rho_ld: np.ndarray,
    method: Literal["difference","ratio"] = "difference",
) -> np.ndarray:
    """
    Compute a per-point mismatch between HD and LD density estimates.

    Parameters
    ----------
    rho_hd : (n_samples,) array
        Density in high-dimensional space.
    rho_ld : (n_samples,) array
        Density in low-dimensional embedding.
    method : {'difference','ratio'}
        - 'difference': δ = rho_ld_norm - rho_hd_norm in [-1,1]
        - 'ratio':      δ = (rho_ld + eps) / (rho_hd + eps)

    Returns
    -------
    δ : (n_samples,) array
        Density mismatch (positive => LD appears denser than HD).
    """
    if rho_hd.shape != rho_ld.shape:
        raise ValueError("rho_hd and rho_ld must have the same shape")

    eps = 1e-12
    if method == "difference":
        a = normalize_density(rho_hd, eps)
        b = normalize_density(rho_ld, eps)
        return b - a
    elif method == "ratio":
        return (rho_ld + eps) / (rho_hd + eps)
    else:
        raise ValueError("method must be 'difference' or 'ratio'")


def grid_kde(
    Y: np.ndarray,
    bw_method: Literal["scott","silverman",float]="scott",
    grid_size: int = 200,
    padding: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate 2D KDE on a regular grid for contour overlays in LD space.

    Parameters
    ----------
    Y : (n_samples, 2) array
        2D embedding coordinates (e.g., UMAP).
    bw_method : {'scott','silverman', float}
        KDE bandwidth rule or scalar.
    grid_size : int
        Number of grid steps per axis (default 200).
    padding : float in [0,0.5)
        Fractional padding around data bounds.

    Returns
    -------
    Xg, Yg, Z : (grid_size, grid_size) arrays
        Grid coordinates and KDE values on the grid.
    """
    if Y.shape[1] != 2:
        raise ValueError("grid_kde currently supports 2D embeddings only")

    kde = gaussian_kde(Y.T, bw_method=bw_method)

    xmin, ymin = Y.min(axis=0)
    xmax, ymax = Y.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    coords = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde.evaluate(coords).reshape(grid_size, grid_size)
    return Xg, Yg, Z


__all__ = [
    "knn_density",
    "kde_density",
    "normalize_density",
    "density_mismatch",
    "grid_kde",
]
