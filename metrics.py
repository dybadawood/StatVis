# statvis/metrics.py
"""
StatVis Metrics Module
======================

Implements internal cluster validation and projection distortion metrics used in StatVis:
- Silhouette (per-point & average)
- Davies–Bouldin Index (DBI)
- Dunn Index (DI)
- Trustworthiness (neighborhood preservation from HD→LD)
- Continuity (neighborhood preservation from LD→HD)
- Neighborhood Preservation@k (overlap ratio of k-NN sets)

Design goals
------------
- Clear, well-documented API with NumPy arrays
- Safe defaults and informative docstrings
- Efficient implementations leveraging scikit-learn when possible
"""

from __future__ import annotations

from typing import Tuple, Optional, Iterable, Dict
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as _trustworthiness


# ------------------------------
# Core internal validation
# ------------------------------

def silhouette(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Silhouette scores.

    Parameters
    ----------
    X : (n_samples, n_features) array
        Original (high-dimensional) features.
    labels : (n_samples,) array
        Cluster assignments (integers). Noise points can be set to -1;
        scikit-learn will ignore singletons for silhouette.

    Returns
    -------
    s_i : (n_samples,) array
        Per-point silhouette values in [-1, 1].
    s_avg : float
        Average silhouette across all points.
    """
    s_i = silhouette_samples(X, labels)
    s_avg = float(np.mean(s_i))
    return s_i, s_avg


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies–Bouldin Index (lower is better).

    Notes
    -----
    Uses scikit-learn's efficient implementation.
    """
    return float(davies_bouldin_score(X, labels))


def dunn_index(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Dunn Index (higher is better).

    Definition
    ----------
    DI = (min inter-cluster distance) / (max intra-cluster diameter)

    Implementation
    --------------
    - Computes a full pairwise distance matrix in the chosen metric.
    - For large n, this can be memory-intensive (O(n^2)); callers may
      pre-downsample or pass a subset.

    Parameters
    ----------
    X : (n_samples, n_features) array
    labels : (n_samples,) array of ints
    metric : str
        Distance metric for pairwise_distances.

    Returns
    -------
    float
        Dunn index value (>= 0). Returns 0 if degenerate.
    """
    D = pairwise_distances(X, metric=metric)
    unique = np.unique(labels)

    # Intra-cluster diameters (max pairwise distance within a cluster)
    diameters = []
    for c in unique:
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            diameters.append(0.0)
        else:
            diameters.append(np.max(D[np.ix_(idx, idx)]))
    max_intra = np.max(diameters) if len(diameters) else 0.0

    # Inter-cluster separations (min distance between points across clusters)
    inter = []
    for i, ci in enumerate(unique):
        for cj in unique[i+1:]:
            ii = np.where(labels == ci)[0]
            jj = np.where(labels == cj)[0]
            inter.append(np.min(D[np.ix_(ii, jj)]))
    min_inter = np.min(inter) if len(inter) else 0.0

    if max_intra == 0.0:
        return 0.0
    return float(min_inter / max_intra)


# ------------------------------
# Distortion / neighborhood preservation metrics
# ------------------------------

def trustworthiness(X_hd: np.ndarray, X_ld: np.ndarray, n_neighbors: int = 5, metric: str = "euclidean") -> float:
    """
    Trustworthiness \n
    Measures how many low-dimensional neighbors are not true neighbors in high-d space.
    Higher is better; range [0, 1].

    Parameters
    ----------
    X_hd : (n_samples, d_hd) array
        High-dimensional data.
    X_ld : (n_samples, d_ld) array
        Low-dimensional embedding (e.g., UMAP/t-SNE).
    n_neighbors : int
        Neighborhood size k.
    metric : str
        Distance metric used for HD neighbor computation.

    Returns
    -------
    float in [0,1]
    """
    return float(_trustworthiness(X_hd, X_ld, n_neighbors=n_neighbors, metric=metric))


def continuity(X_hd: np.ndarray, X_ld: np.ndarray, n_neighbors: int = 5, metric: str = "euclidean") -> float:
    """
    Continuity \n
    Measures how many high-dimensional neighbors are preserved in low-d space.
    Higher is better; range [0, 1].

    Reference
    ---------
    Kaski & Venna neighborhood preservation metric.

    Parameters
    ----------
    X_hd : (n_samples, d_hd) array
    X_ld : (n_samples, d_ld) array
    n_neighbors : int
    metric : str
        Metric used by NearestNeighbors for both spaces.

    Returns
    -------
    float in [0,1]
    """
    n = X_hd.shape[0]
    k = int(n_neighbors)

    nn_hd = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X_hd)
    nn_ld = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X_ld)

    hd_ind = nn_hd.kneighbors(return_distance=False)[:, 1:]  # drop self
    ld_ind = nn_ld.kneighbors(return_distance=False)[:, 1:]

    # Build rank maps for LD neighborhoods
    ranks_ld = []
    for i in range(n):
        ranks_ld.append({idx: r for r, idx in enumerate(ld_ind[i])})

    penalty = 0.0
    for i in range(n):
        U_i = set(hd_ind[i])     # true HD neighbors
        V_i = set(ld_ind[i])     # LD neighbors
        missing = list(U_i - V_i)
        ranks = ranks_ld[i]
        for m in missing:
            # If missing, approximate its rank in LD as k+1
            r_m = ranks.get(m, k+1)
            penalty += (r_m - k)

    denom = n * k * (2 * n - 3 * k - 1)
    return float(1 - (2.0 / denom) * penalty) if denom > 0 else 0.0


def knn_overlap_ratio(X_hd: np.ndarray, X_ld: np.ndarray, n_neighbors: int = 10, metric: str = "euclidean") -> float:
    """
    Neighborhood Preservation@k (overlap ratio).

    Computes the average Jaccard overlap between the k-NN sets in HD and LD.

    Returns
    -------
    float in [0,1]
    """
    k = int(n_neighbors)
    nn_hd = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X_hd)
    nn_ld = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X_ld)

    hd_ind = nn_hd.kneighbors(return_distance=False)[:, 1:]
    ld_ind = nn_ld.kneighbors(return_distance=False)[:, 1:]

    n = X_hd.shape[0]
    overlaps = []
    for i in range(n):
        U = set(hd_ind[i])
        V = set(ld_ind[i])
        inter = len(U & V)
        union = len(U | V)
        overlaps.append(inter / union if union > 0 else 0.0)
    return float(np.mean(overlaps))


# ------------------------------
# Convenience summary
# ------------------------------

def summarize_internal_metrics(X: np.ndarray, labels: np.ndarray, *, metric: str = "euclidean") -> Dict[str, float]:
    """
    Compute a summary of internal validation metrics.

    Returns
    -------
    dict with keys:
        - silhouette_avg
        - davies_bouldin
        - dunn
    """
    _, s_avg = silhouette(X, labels)
    dbi = davies_bouldin(X, labels)
    di = dunn_index(X, labels, metric=metric)
    return {
        "silhouette_avg": float(s_avg),
        "davies_bouldin": float(dbi),
        "dunn": float(di),
    }


def summarize_distortion_metrics(X_hd: np.ndarray, X_ld: np.ndarray, *, n_neighbors: int = 10, metric: str = "euclidean") -> Dict[str, float]:
    """
    Compute a summary of projection reliability metrics.

    Returns
    -------
    dict with keys:
        - trustworthiness
        - continuity
        - knn_overlap
    """
    t = trustworthiness(X_hd, X_ld, n_neighbors=n_neighbors, metric=metric)
    c = continuity(X_hd, X_ld, n_neighbors=n_neighbors, metric=metric)
    j = knn_overlap_ratio(X_hd, X_ld, n_neighbors=n_neighbors, metric=metric)
    return {
        "trustworthiness": float(t),
        "continuity": float(c),
        "knn_overlap": float(j),
    }


__all__ = [
    "silhouette",
    "davies_bouldin",
    "dunn_index",
    "trustworthiness",
    "continuity",
    "knn_overlap_ratio",
    "summarize_internal_metrics",
    "summarize_distortion_metrics",
]
