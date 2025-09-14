# statvis/visualize.py
"""
StatVis Visualization Module
============================

Matplotlib-based plotting utilities for StatVis visualizations.

Capabilities
------------
- Scatter plots of embeddings with metric overlays (Silhouette, DBI, Dunn)
- Density mismatch overlays (opacity, contours, heatmaps)
- Standardized legends, colorbars, and accessibility-friendly colormaps
- Publication-ready figure exports (high DPI, labeled axes)

Design
------
- Functions return (fig, ax) for further customization
- Parameters allow saving to file paths (`out` argument)
- Colormaps: ColorBrewer / Matplotlib colorblind-safe schemes

"""

from __future__ import annotations

from typing import Optional, Tuple, Literal, Dict
import numpy as np
import matplotlib.pyplot as plt

from .distortion import derive_overlays, mismatch_grid


# ------------------------------
# Embedding scatter with Silhouette overlay
# ------------------------------

def plot_embedding_with_scores(
    Y: np.ndarray,
    scores: np.ndarray,
    title: str = "Embedding with Metric Overlay",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    score_label: str = "Score",
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot embedding with per-point scores (e.g., Silhouette).

    Parameters
    ----------
    Y : (n,2) array
        2D embedding coordinates.
    scores : (n,) array
        Per-point scores to map to color.
    title : str
    xlabel, ylabel : str
        Axis labels (e.g., "UMAP1","UMAP2").
    score_label : str
        Colorbar label.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color scale range. If None, inferred from data.
    out : str or None
        If given, path to save figure (PNG/PDF).

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    sc = ax.scatter(Y[:,0], Y[:,1], c=scores, s=10, cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label(score_label)

    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig, ax


# ------------------------------
# Density mismatch visualization
# ------------------------------

def plot_density_mismatch(
    Y: np.ndarray,
    delta: np.ndarray,
    overlays: Optional[Dict[str,np.ndarray]] = None,
    title: str = "Density-Aware Correction Overlay",
    xlabel: str = "UMAP1",
    ylabel: str = "UMAP2",
    out: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot embedding with density-aware correction overlays.

    Parameters
    ----------
    Y : (n,2) array
        2D embedding coordinates.
    delta : (n,) array
        Density mismatch values in [-1,1].
    overlays : dict or None
        From derive_overlays(). Should include 'opacity'.
    """
    fig, ax = plt.subplots(figsize=(6,5), dpi=200)

    if overlays is None:
        from .distortion import derive_overlays
        overlays = derive_overlays(delta)

    sc = ax.scatter(Y[:,0], Y[:,1], c=delta, cmap="RdBu_r", s=10,
                    alpha=overlays["opacity"], vmin=-1, vmax=1, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
    cbar.set_label("Density mismatch (−1 … +1)")

    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig, ax


def plot_density_mismatch_grid(
    Y: np.ndarray,
    delta: np.ndarray,
    grid_size: int = 200,
    title: str = "Density Mismatch Heatmap",
    xlabel: str = "UMAP1",
    ylabel: str = "UMAP2",
    out: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of density mismatch interpolated on a grid.

    Parameters
    ----------
    Y : (n,2) array
    delta : (n,) array
        Density mismatch values.
    """
    Xg, Yg, Z = mismatch_grid(Y, delta, grid_size=grid_size)

    fig, ax = plt.subplots(figsize=(6,5), dpi=200)
    im = ax.imshow(Z, origin="lower", extent=(Xg.min(), Xg.max(), Yg.min(), Yg.max()),
                   cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Density mismatch (−1 … +1)")

    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig, ax


__all__ = [
    "plot_embedding_with_scores",
    "plot_density_mismatch",
    "plot_density_mismatch_grid",
]
