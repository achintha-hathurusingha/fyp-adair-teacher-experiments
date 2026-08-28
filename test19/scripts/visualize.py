"""TEST19 visualization -- the numbers from same_image_pca16.py, made visible.

Two figures:
  1. scatter: PCA-16 (projected to its own top-2 components for plotting)
     scatter, colored by degradation, with each scene's 3 points connected
     by a thin line -- if degradation dominates, these "triangles" should
     visibly cross between color clusters rather than sit inside one.
  2. distance_hist: same-scene/different-degradation vs
     different-scene/same-degradation pairwise distance distributions,
     overlaid -- the direct visual counterpart of the 1.73x ratio finding.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

OUT = "/home/minura/teacher-experiments/test19/results"
import os
os.makedirs(OUT, exist_ok=True)

d = np.load('/home/minura/teacher-experiments/test05/results/feature_analysis/latent_pre.npz',
            allow_pickle=True)
X_full = np.concatenate([d['X_gap'], d['X_gmp']], axis=1)
deg = np.array(d['degradation'])
scene = np.array(d['scene_id'])
n = len(deg)

scaler = StandardScaler()
Xs = scaler.fit_transform(X_full)
pca16 = PCA(n_components=16, random_state=0)
X16 = pca16.fit_transform(Xs)

COLORS = {"Rain": "#5b8def", "Haze": "#e8a33d", "Noise": "#5fbf8f"}

# ---- Figure 1: scatter with same-scene triangles ----
# Project the already-fit PCA-16 space to its own top-2 axes for plotting
# (not a fresh 2D fit -- this shows the SAME space the 99% classification
# result lives in, just viewed along its two highest-variance directions).
x2 = X16[:, 0]
y2 = X16[:, 1]

fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="#0a0d13")
ax.set_facecolor("#0a0d13")

for sc in sorted(set(scene)):
    idx = np.where(scene == sc)[0]
    order = {"Rain": 0, "Haze": 1, "Noise": 2}
    idx = sorted(idx, key=lambda i: order[deg[i]])
    pts = [(x2[i], y2[i]) for i in idx]
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color="#3a4258", linewidth=0.5, alpha=0.5, zorder=1)

for d_name, c in COLORS.items():
    idx = np.where(deg == d_name)[0]
    ax.scatter(x2[idx], y2[idx], s=36, c=c, label=d_name, zorder=2,
               edgecolors="#0a0d13", linewidths=0.4)

ax.set_xlabel(f"PC1 ({pca16.explained_variance_ratio_[0]*100:.1f}% var)", color="#8891a8")
ax.set_ylabel(f"PC2 ({pca16.explained_variance_ratio_[1]*100:.1f}% var)", color="#8891a8")
ax.set_title("PCA-16(latent_pre), top 2 axes -- same-scene triples connected",
             color="#e8eaf0", fontsize=13)
ax.tick_params(colors="#5c6478")
for spine in ax.spines.values():
    spine.set_color("#262c3d")
leg = ax.legend(facecolor="#12161f", edgecolor="#262c3d", labelcolor="#e8eaf0")
fig.tight_layout()
fig.savefig(f"{OUT}/scatter_same_scene_triples.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"wrote {OUT}/scatter_same_scene_triples.png")

# ---- Figure 2: distance distributions ----
D = cdist(X16, X16, metric='euclidean')
np.fill_diagonal(D, np.nan)
same_scene_dists, same_deg_dists = [], []
for i in range(n):
    for j in range(i + 1, n):
        if scene[i] == scene[j] and deg[i] != deg[j]:
            same_scene_dists.append(D[i, j])
        elif scene[i] != scene[j] and deg[i] == deg[j]:
            same_deg_dists.append(D[i, j])
same_scene_dists = np.array(same_scene_dists)
same_deg_dists = np.array(same_deg_dists)

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0a0d13")
ax.set_facecolor("#0a0d13")
bins = np.linspace(0, max(same_scene_dists.max(), same_deg_dists.max()), 40)
ax.hist(same_deg_dists, bins=bins, alpha=0.65, color="#5b8def",
        label=f"different scene, SAME degradation (mean={same_deg_dists.mean():.1f})",
        density=True)
ax.hist(same_scene_dists, bins=bins, alpha=0.65, color="#f472b6",
        label=f"same scene, DIFFERENT degradation (mean={same_scene_dists.mean():.1f})",
        density=True)
ax.axvline(same_deg_dists.mean(), color="#5b8def", linestyle="--", linewidth=1.2)
ax.axvline(same_scene_dists.mean(), color="#f472b6", linestyle="--", linewidth=1.2)
ax.set_xlabel("pairwise Euclidean distance in PCA-16 space", color="#8891a8")
ax.set_ylabel("density", color="#8891a8")
ax.set_title(f"Same-image variants land {same_scene_dists.mean()/same_deg_dists.mean():.2f}x "
             f"farther apart than same-degradation, different images",
             color="#e8eaf0", fontsize=12)
ax.tick_params(colors="#5c6478")
for spine in ax.spines.values():
    spine.set_color("#262c3d")
leg = ax.legend(facecolor="#12161f", edgecolor="#262c3d", labelcolor="#e8eaf0", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/distance_distributions.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"wrote {OUT}/distance_distributions.png")
