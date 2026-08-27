"""Phase 10-11: PCA (all major feature levels) and t-SNE/UMAP (input,
latent, AFLB outputs) colored by degradation. Qualitative/exploratory only
-- NOT primary quantitative evidence (that's linear_probe.py / distance_analysis.py).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python dimensionality_reduction.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

TEST02 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST02 / "results" / "features"
VIZ_DIR = TEST02 / "results" / "visualizations"
DEG_COLORS = {"Rain": "#3b7dd8", "Haze": "#d8853b", "Noise": "#3bb273"}
DEG_ORDER = ["Rain", "Haze", "Noise"]

PCA_FEATURES = ["input", "shallow_Y0", "encoder_level1", "encoder_level2", "encoder_level3",
                "latent", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out", "output"]
TSNE_FEATURES = ["input", "latent", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False


def load(feature_name):
    d = np.load(FEATURES_DIR / f"{feature_name}.npz", allow_pickle=True)
    return d["X"], d["degradation"]


def scatter(ax, xy, degs, title):
    for deg in DEG_ORDER:
        mask = degs == deg
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.7, label=deg, color=DEG_COLORS[deg])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    pca_dir = VIZ_DIR / "pca"
    tsne_dir = VIZ_DIR / "tsne"
    umap_dir = VIZ_DIR / "umap"
    pca_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    print("=== PCA ===", flush=True)
    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    for ax, fname in zip(axes.flat, PCA_FEATURES):
        X, degs = load(fname)
        Xs = StandardScaler().fit_transform(X)
        xy = PCA(n_components=2, random_state=0).fit_transform(Xs)
        scatter(ax, xy, degs, fname)
    fig.suptitle("PCA of pooled features, colored by degradation (analysis-only labels)")
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_all_stages.png", dpi=120)
    plt.close(fig)
    print(f"wrote {pca_dir / 'pca_all_stages.png'}")

    print("=== t-SNE ===", flush=True)
    fig, axes = plt.subplots(1, len(TSNE_FEATURES), figsize=(5 * len(TSNE_FEATURES), 5))
    for ax, fname in zip(axes, TSNE_FEATURES):
        X, degs = load(fname)
        Xs = StandardScaler().fit_transform(X)
        xy = TSNE(n_components=2, random_state=0, perplexity=min(30, len(Xs) // 4), init="pca").fit_transform(Xs)
        scatter(ax, xy, degs, fname)
    fig.suptitle("t-SNE (qualitative only -- see linear_probe_results.csv for quantitative evidence)")
    fig.tight_layout()
    fig.savefig(tsne_dir / "tsne_key_stages.png", dpi=120)
    plt.close(fig)
    print(f"wrote {tsne_dir / 'tsne_key_stages.png'}")

    if HAVE_UMAP:
        print("=== UMAP ===", flush=True)
        umap_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(TSNE_FEATURES), figsize=(5 * len(TSNE_FEATURES), 5))
        for ax, fname in zip(axes, TSNE_FEATURES):
            X, degs = load(fname)
            Xs = StandardScaler().fit_transform(X)
            xy = umap.UMAP(n_components=2, random_state=0).fit_transform(Xs)
            scatter(ax, xy, degs, fname)
        fig.suptitle("UMAP (qualitative only)")
        fig.tight_layout()
        fig.savefig(umap_dir / "umap_key_stages.png", dpi=120)
        plt.close(fig)
        print(f"wrote {umap_dir / 'umap_key_stages.png'}")
    else:
        print("umap-learn not installed -- skipping UMAP (t-SNE covers the qualitative requirement)")


if __name__ == "__main__":
    main()
