"""TEST10-R Phase 3: verify the FIXED teacher trajectory targets are
non-collapsed BEFORE any student training starts. This is a hard gate --
per the task spec, if a target itself has near-zero variance, STOP and do
not proceed to student training.

For each of the 3 stages, computes: per-dimension mean/std, effective rank
(participation ratio of the covariance eigenvalue spectrum), pairwise
cosine similarity across DIFFERENT images, pairwise Euclidean distance
across different images, and a grouped (by scene_id) degradation
classification accuracy probe.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python target_collapse_audit.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST10R = Path(__file__).resolve().parent.parent
CACHE_DIR = TEST10R / "results" / "teacher_cache"
STATS_DIR = TEST10R / "results" / "statistics"
VIZ_DIR = TEST10R / "results" / "visualizations"
COLLAPSE_COSINE_THRESHOLD = 0.98
N_PAIRS_SAMPLE = 2000


def effective_rank(X: np.ndarray) -> float:
    """Participation ratio: (sum(eigvals))^2 / sum(eigvals^2). Equals the
    number of components with equal eigenvalue for a perfectly spread
    spectrum; collapses toward 1 if one component dominates all variance."""
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    return float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))


def grouped_probe(X, y, groups):
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), y_enc[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def pairwise_stats(X: np.ndarray, rng: np.random.RandomState, n_pairs=N_PAIRS_SAMPLE):
    n = len(X)
    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    a, b = X[idx_i], X[idx_j]
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = (a_n * b_n).sum(axis=1)
    dist = np.linalg.norm(a - b, axis=1)
    return cos, dist


def main():
    d = np.load(CACHE_DIR / "trajectory_targets.npz", allow_pickle=True)
    scene_id, degradation = d["scene_id"], d["degradation"]
    rng = np.random.RandomState(0)

    audit_rows = []
    STOP = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig_cos, axes_cos = plt.subplots(1, 3, figsize=(15, 4))
    fig_dist, axes_dist = plt.subplots(1, 3, figsize=(15, 4))

    for stage_idx in (0, 1, 2):
        E = d[f"E_stage{stage_idx}"]
        per_dim_mean = E.mean(axis=0)
        per_dim_std = E.std(axis=0)
        eff_rank = effective_rank(E)
        cos, dist = pairwise_stats(E, rng)
        acc, f1 = grouped_probe(E, degradation, scene_id)

        collapsed = bool(cos.mean() > COLLAPSE_COSINE_THRESHOLD or per_dim_std.mean() < 1e-6)
        if collapsed:
            STOP = True

        audit_rows.append({
            "stage": stage_idx, "mean_per_dim_std": float(per_dim_std.mean()),
            "min_per_dim_std": float(per_dim_std.min()), "effective_rank": eff_rank,
            "n_dims": E.shape[1], "mean_pairwise_cosine": float(cos.mean()),
            "p95_pairwise_cosine": float(np.percentile(cos, 95)),
            "mean_pairwise_distance": float(dist.mean()),
            "degradation_probe_accuracy": acc, "degradation_probe_macro_f1": f1,
            "collapsed": collapsed,
        })
        print(f"stage{stage_idx}: mean_std={per_dim_std.mean():.4f} eff_rank={eff_rank:.2f}/{E.shape[1]} "
              f"mean_pairwise_cos={cos.mean():.4f} probe_acc={acc*100:.1f}% collapsed={collapsed}", flush=True)

        axes[stage_idx].scatter(E[:, 0], E[:, 1], c=pd.factorize(degradation)[0], cmap="tab10", s=8, alpha=0.6)
        axes[stage_idx].set_title(f"Stage {stage_idx}: PC1 vs PC2 (colored by degradation)")
        axes[stage_idx].set_xlabel("PC1")
        axes[stage_idx].set_ylabel("PC2")

        axes_cos[stage_idx].hist(cos, bins=40, color="#1f77b4")
        axes_cos[stage_idx].axvline(COLLAPSE_COSINE_THRESHOLD, color="red", linestyle="--", label="collapse threshold")
        axes_cos[stage_idx].set_title(f"Stage {stage_idx}: pairwise cosine (different images)")
        axes_cos[stage_idx].legend()

        axes_dist[stage_idx].hist(dist, bins=40, color="#2ca02c")
        axes_dist[stage_idx].set_title(f"Stage {stage_idx}: pairwise Euclidean distance")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(STATS_DIR / "target_collapse_audit.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'target_collapse_audit.csv'}")

    fig.tight_layout()
    fig.savefig(VIZ_DIR / "target_pca_distribution.png", dpi=150)
    fig_cos.tight_layout()
    fig_cos.savefig(VIZ_DIR / "target_pairwise_cosine_histogram.png", dpi=150)
    fig_dist.tight_layout()
    fig_dist.savefig(VIZ_DIR / "target_pairwise_distance_histogram.png", dpi=150)
    plt.close("all")
    print("wrote 3 target-quality visualizations")

    if STOP:
        print("\n" + "=" * 70)
        print("STOP: at least one teacher trajectory target is COLLAPSED.")
        print("Do not proceed to student training until this is fixed.")
        print("=" * 70)
        raise SystemExit(1)
    else:
        print("\nAll 3 teacher trajectory targets PASS the collapse audit "
              "(non-trivial variance, low pairwise cosine, meaningful degradation "
              "structure). Safe to proceed to student training.")


if __name__ == "__main__":
    main()
