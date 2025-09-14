# statvis/embedding.py
"""
StatVis Embedding Module
========================

Implements manifold and linear embeddings used by StatVis:
- UMAP (umap-learn)
- t-SNE (scikit-learn)
- PCA (scikit-learn)
- Isomap (scikit-learn)

Design goals
------------
- Consistent, dataclass-based configuration (`EmbedConfig`)
- Sensible defaults with full parameter override via `params`
- Reproducible embeddings with `random_state`
- Optional preprocessing (standardization) switch
- Return both the low-dimensional embedding and the fitted model (when applicable)

Example
-------
>>> import numpy as np
>>> from sklearn.datasets import load_iris
>>> from statvis.embedding import EmbedConfig, embed
>>> X, _ = load_iris(return_X_y=True)
>>> Y, model = embed(X, EmbedConfig(method="umap", n_components=2, random_state=42))
>>> Y.shape
(150, 2)

Notes
-----
- UMAP and t-SNE are stochastic; set `random_state` for reproducibility.
- For t-SNE, the recommended perplexity is between 5 and 50.
- UMAP parameters commonly tuned: `n_neighbors`, `min_dist`, `metric`.
- Isomap parameter commonly tuned: `n_neighbors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np

try:
    # umap-learn (preferred)
    import umap
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler


EmbedMethod = Literal["umap", "tsne", "pca", "isomap"]


@dataclass
class EmbedConfig:
    """
    Configuration for an embedding run.

    Attributes
    ----------
    method : {'umap','tsne','pca','isomap'}
        Embedding algorithm to use.
    n_components : int
        Target dimensionality (2 or 3 typical).
    random_state : int
        Seed for reproducibility where supported.
    params : dict | None
        Extra keyword args forwarded to the underlying implementation.
        Examples:
            - UMAP: {'n_neighbors': 30, 'min_dist': 0.1, 'metric': 'euclidean'}
            - t-SNE: {'perplexity': 30, 'learning_rate': 'auto', 'init': 'pca'}
            - Isomap: {'n_neighbors': 10, 'metric': 'minkowski'}
            - PCA: {'svd_solver': 'auto', 'whiten': False}
    standardize : bool
        If True, apply StandardScaler to features before embedding.
        Recommended for PCA/Isomap and often helpful for UMAP/t-SNE.
    return_model : bool
        If True, return the fitted embedding model along with coordinates.
    """
    method: EmbedMethod = "umap"
    n_components: int = 2
    random_state: int = 42
    params: Optional[Dict[str, Any]] = field(default_factory=dict)
    standardize: bool = True
    return_model: bool = True


def _standardize_if_needed(X: np.ndarray, do_scale: bool) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if not do_scale:
        return X, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def embed(X: np.ndarray, cfg: EmbedConfig) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Compute a low-dimensional embedding.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    cfg : EmbedConfig
        Embedding configuration.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Low-dimensional coordinates.
    model : object or None
        Fitted model object (e.g., UMAP/PCA/Isomap instance). For t-SNE,
        scikit-learn does not expose a transform API; we return the TSNE
        instance for completeness. Returned only if cfg.return_model=True.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got shape {X.shape}")

    X_proc, scaler = _standardize_if_needed(X, cfg.standardize)
    p = dict(cfg.params or {})  # copy to avoid in-place mutation

    method = cfg.method.lower()
    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. `pip install umap-learn`")
        # sensible defaults that can be overridden by params
        default = dict(n_neighbors=15, min_dist=0.1, metric="euclidean")
        for k, v in default.items():
            p.setdefault(k, v)

        reducer = umap.UMAP(
            n_components=cfg.n_components,
            random_state=cfg.random_state,
            **p,
        )
        Y = reducer.fit_transform(X_proc)
        # Attach scaler for potential inverse-transform in downstream pipelines
        reducer._statvis_scaler = scaler  # type: ignore[attr-defined]

        return (Y, reducer) if cfg.return_model else (Y, None)

    elif method == "tsne":
        # scikit-learn TSNE
        perplexity = p.pop("perplexity", 30)
        learning_rate = p.pop("learning_rate", "auto")
        init = p.pop("init", "pca")

        tsne = TSNE(
            n_components=cfg.n_components,
            random_state=cfg.random_state,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            **p,
        )
        Y = tsne.fit_transform(X_proc)
        tsne._statvis_scaler = scaler  # type: ignore[attr-defined]
        return (Y, tsne) if cfg.return_model else (Y, None)

    elif method == "pca":
        # Allow passing PCA kwargs (e.g., whiten, svd_solver)
        p.setdefault("svd_solver", "auto")
        p.setdefault("whiten", False)
        pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state, **p)
        Y = pca.fit_transform(X_proc)
        pca._statvis_scaler = scaler  # type: ignore[attr-defined]
        return (Y, pca) if cfg.return_model else (Y, None)

    elif method == "isomap":
        # Allow passing Isomap kwargs (e.g., n_neighbors, metric)
        p.setdefault("n_neighbors", 10)
        isomap = Isomap(n_neighbors=p["n_neighbors"], n_components=cfg.n_components,
                        metric=p.get("metric", "minkowski"))
        Y = isomap.fit_transform(X_proc)
        isomap._statvis_scaler = scaler  # type: ignore[attr-defined]
        return (Y, isomap) if cfg.return_model else (Y, None)

    else:
        raise ValueError(f"Unknown method: {cfg.method!r}. "
                         "Choose from {'umap','tsne','pca','isomap'}.")


def embed_3d(X: np.ndarray, cfg: Optional[EmbedConfig] = None) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Convenience wrapper to produce 3D embeddings.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    cfg : EmbedConfig or None
        If None, uses UMAP with n_components=3 and default params.

    Returns
    -------
    Y : ndarray (n_samples, 3)
    model : fitted model or None
    """
    if cfg is None:
        cfg = EmbedConfig(method="umap", n_components=3, random_state=42)
    else:
        cfg = EmbedConfig(**{**cfg.__dict__, "n_components": 3})
    return embed(X, cfg)


__all__ = [
    "EmbedConfig",
    "embed",
    "embed_3d",
]
