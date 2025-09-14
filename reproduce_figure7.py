# scripts/reproduce_figure7.py
"""
Reproduce Figure 7: UMAP embedding of Dry Bean dataset with Silhouette overlays.

This script demonstrates StatVis functionality:
- Loads or simulates Dry Bean-like data (7 clusters, 16 features).
- Runs KMeans clustering (K=7).
- Computes UMAP embedding (2D).
- Computes Silhouette scores on HD space with cluster labels.
- Visualizes UMAP embedding color-coded by Silhouette score with standardized legend.

Usage
-----
$ python scripts/reproduce_figure7.py

Output
------
- Console: prints average Silhouette, DBI, Dunn index
- File: 'figure7_umap_silhouette.png' in working directory
"""

import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from statvis.embedding import embed, EmbedConfig
from statvis.metrics import silhouette, davies_bouldin, dunn_index
from statvis.visualize import plot_embedding_with_scores


def main():
    # For reproducibility, we simulate Dry Bean-like data with 7 classes
    X, y_true = make_blobs(
        n_samples=8000,
        centers=7,
        n_features=16,
        cluster_std=2.0,
        random_state=7,
    )

    # KMeans clustering
    km = KMeans(n_clusters=7, n_init=10, random_state=7)
    y_pred = km.fit_predict(X)

    # UMAP embedding (2D)
    Y, reducer = embed(
        X,
        EmbedConfig(method="umap", n_components=2, random_state=7, params={"n_neighbors": 30, "min_dist": 0.1})
    )

    # Compute metrics in HD space
    s_i, s_avg = silhouette(X, y_pred)
    dbi = davies_bouldin(X, y_pred)
    di = dunn_index(X, y_pred)

    print(f"Avg Silhouette: {s_avg:.3f} | DBI: {dbi:.3f} | Dunn: {di:.3f}")

    # Plot
    fig, ax = plot_embedding_with_scores(
        Y,
        s_i,
        title="Figure 7: UMAP embedding — Silhouette scores (KMeans, K=7)",
        xlabel="UMAP1",
        ylabel="UMAP2",
        score_label="Silhouette Score",
        cmap="RdYlBu_r",
        vmin=-1, vmax=1,
        out="figure7_umap_silhouette.png",
    )

    print("Saved: figure7_umap_silhouette.png")


if __name__ == "__main__":
    main()
