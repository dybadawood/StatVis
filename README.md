[README.md](https://github.com/user-attachments/files/22320477/README.md)
# StatVis: A Visual Analytics Framework for Statistical Cluster Validation in High Dimensions

StatVis is a Python toolkit implementing the methods described in the manuscript:
**"StatVis: A Visual Analytics Framework for Statistical Cluster Validation in High Dimensions"**.

It provides a unified workflow for:
- Computing **embeddings** (UMAP, t-SNE, PCA, Isomap)
- Assessing cluster validity via **internal metrics** (Silhouette, DBI, Dunn)
- Estimating **density** in HD and LD spaces (kNN, KDE)
- Detecting and visualizing **projection distortions** (trustworthiness, continuity, density mismatch)
- Generating **publication-ready figures** with standardized legends and accessible colormaps

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/statvis.git
cd statvis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage Examples

### Reproduce Figure 7 (UMAP + Silhouette)

```bash
python scripts/reproduce_figure7.py
```

- Runs UMAP embedding on a Dry Bean–like dataset
- Computes KMeans clusters and Silhouette scores
- Saves `figure7_umap_silhouette.png`

### Reproduce Figure Y (Density-Aware Correction)

```bash
python scripts/reproduce_figureY.py
```

- Computes density mismatch between HD and LD
- Applies correction overlays (opacity/contour adjustments)
- Saves before-and-after plots

---

## Datasets

- **Dry Bean Dataset**: [UCI Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)  
- **MNIST**: [Kaggle Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
- **20 Newsgroups**: [UCI Repository](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups)  

Scripts include links or loaders for these datasets.

---

## Directory Structure

```
statvis/              # Core package
  embedding.py        # Embedding methods
  metrics.py          # Cluster validation metrics
  density.py          # Density estimation
  distortion.py       # Distortion & correction routines
  visualize.py        # Visualization utilities
scripts/              # Reproducibility scripts
  reproduce_figure7.py
  reproduce_figureY.py
requirements.txt      # Dependencies
README.md             # This file
LICENSE
```

---

## Citation

If you use StatVis in academic work, please cite:

```
@article{YourCitation2025,
  title   = {StatVis: A Visual Analytics Framework for Statistical Cluster Validation in High Dimensions},
  author  = {Your Name and Co-Authors},
  journal = {Array},
  year    = {2025},
  doi     = {10.xxxx/array.xxxxx}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
