"""TEST03 Phase 12: within-scene representation visualization. For several
scenes, plot Rain/Haze/Noise latent positions (PCA-reduced) and connect
same-scene points, to visually show whether degradation moves the
representation while scene content stays fixed.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python within_scene_viz.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TEST03 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST03 / "results" / "features"
VIZ_DIR = TEST03 / "results" / "visualizations"
DEG_MARKERS = {"Rain": ("o", "#3b7dd8"), "Haze": ("^", "#d8853b"), "Noise": ("s", "#3bb273")}
N_SCENES_SHOW = 12


def main():
    d = np.load(FEATURES_DIR / "latent.npz", allow_pickle=True)
    X, degs, scenes = d["X"], d["degradation"], d["scene_id"]
    Xs = StandardScaler().fit_transform(X)
    xy = PCA(n_components=2, random_state=0).fit_transform(Xs)

    unique_scenes = sorted(set(scenes))[:N_SCENES_SHOW]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20")
    for si, scene_id in enumerate(unique_scenes):
        mask = scenes == scene_id
        pts = {deg: xy[mask & (degs == deg)][0] for deg in DEG_MARKERS if (mask & (degs == deg)).any()}
        color = cmap(si / len(unique_scenes))
        # connect same-scene points with a faint line (triangle)
        order = ["Rain", "Haze", "Noise"]
        coords = [pts[d] for d in order if d in pts]
        if len(coords) == 3:
            tri = np.array(coords + [coords[0]])
            ax.plot(tri[:, 0], tri[:, 1], color=color, alpha=0.35, lw=1, zorder=1)
        for deg, (marker, _) in DEG_MARKERS.items():
            if deg in pts:
                ax.scatter(*pts[deg], marker=marker, s=70, color=color, edgecolor="black", linewidth=0.4, zorder=2)

    legend_handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                                  markeredgecolor="black", markersize=10, label=deg)
                       for deg, (m, _) in DEG_MARKERS.items()]
    ax.legend(handles=legend_handles, title="Degradation (shape)")
    ax.set_title(f"Within-scene latent representation (PCA), {len(unique_scenes)} scenes\n"
                 "each color = one scene; triangle connects that scene's Rain/Haze/Noise latents")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR / "within_scene_latent_pca.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'within_scene_latent_pca.png'}")


if __name__ == "__main__":
    main()
