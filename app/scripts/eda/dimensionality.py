# scripts/eda/dimensionality.py
"""
Dimensionality reduction for EDA: PCA, t-SNE, optional UMAP.

Automatically maps categorical targets to numeric/color for plotting.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import UMAP optionally
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def _get_color_array(df, target):
    """Convert categorical target to numeric/color for plotting."""
    if target is None or target not in df.columns:
        return None
    if df[target].dtype == "object":
        # Map 'yes'/'no' or other string categories to integers
        return df[target].map({k: i for i, k in enumerate(df[target].unique())})
    return df[target]


# ----------------------
# PCA
# ----------------------
def plot_pca(df, target=None, outdir: Path = Path(".")):
    numeric_cols = df.select_dtypes(include="number").columns
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[numeric_cols])
    plt.figure(figsize=(6, 6))
    colors = _get_color_array(df, target)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap="coolwarm", alpha=0.7)
    plt.title("PCA: 2D projection")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "pca_2d.png", dpi=300)
    plt.close()


# ----------------------
# t-SNE
# ----------------------
def plot_tsne(df, target=None, outdir: Path = Path(".")):
    numeric_cols = df.select_dtypes(include="number").columns
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(df[numeric_cols])
    plt.figure(figsize=(6, 6))
    colors = _get_color_array(df, target)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap="coolwarm", alpha=0.7)
    plt.title("t-SNE: 2D projection")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "tsne_2d.png", dpi=300)
    plt.close()


# ----------------------
# UMAP (optional)
# ----------------------
def plot_umap(df, target=None, outdir: Path = Path(".")):
    if not UMAP_AVAILABLE:
        print("[INFO] UMAP not available, skipping UMAP plots.")
        return

    numeric_cols = df.select_dtypes(include="number").columns
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(df[numeric_cols])
    plt.figure(figsize=(6, 6))
    colors = _get_color_array(df, target)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap="coolwarm", alpha=0.7)
    plt.title("UMAP: 2D projection")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "umap_2d.png", dpi=300)
    plt.close()
