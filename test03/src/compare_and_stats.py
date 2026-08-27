"""TEST03 Phase 18-19: compare TEST02 (reference, read-only) vs TEST03
trajectory, and run scene-aware statistical testing on the key
degradation-vs-scene distance result.

Reads test02's feature_trajectory.csv READ-ONLY -- never writes to test02/.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python compare_and_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TEST03 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST03.parent
TEST02_TRAJECTORY = TEACHER_EXP / "test02" / "results" / "classifiers" / "feature_trajectory.csv"
STATS_DIR = TEST03 / "results" / "statistics"
CLASSIFIERS_DIR = TEST03 / "results" / "classifiers"


def compare_trajectories():
    t2 = pd.read_csv(TEST02_TRAJECTORY)
    t3 = pd.read_csv(CLASSIFIERS_DIR / "feature_trajectory.csv")

    merged = t2[["feature_key", "stage", "accuracy_mean"]].merge(
        t3[["feature_key", "accuracy_mean"]], on="feature_key", suffixes=("_test02", "_test03"))
    merged["accuracy_mean_test02_pct"] = merged["accuracy_mean_test02"] * 100
    merged["accuracy_mean_test03_pct"] = merged["accuracy_mean_test03"] * 100
    merged["difference_pct_points"] = merged["accuracy_mean_test03_pct"] - merged["accuracy_mean_test02_pct"]
    merged = merged[["stage", "feature_key", "accuracy_mean_test02_pct", "accuracy_mean_test03_pct",
                      "difference_pct_points"]]

    out_path = STATS_DIR / "test02_vs_test03.csv"
    merged.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(merged.to_string(index=False))
    return merged


def scene_aware_stats():
    """Bootstrap CI (resampling by SCENE, the experimental unit) on the
    degradation_ratio for the latent feature -- the headline result."""
    paired = pd.read_csv(STATS_DIR / "paired_distance_analysis.csv")
    sub = paired[(paired.feature == "latent") & (paired.metric == "euclidean") &
                 (paired.comparison_type == "same_scene_diff_degradation")]
    scene_ids = sub.scene_id.unique()

    rng = np.random.RandomState(0)
    boot_means = []
    for _ in range(2000):
        sample_scenes = rng.choice(scene_ids, size=len(scene_ids), replace=True)
        vals = np.concatenate([sub[sub.scene_id == s]["distance"].to_numpy() for s in sample_scenes])
        boot_means.append(vals.mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    result = {
        "feature": "latent", "metric": "euclidean",
        "n_scenes": len(scene_ids),
        "D_degradation_mean": float(sub["distance"].mean()),
        "bootstrap_ci95_lo": float(ci_lo), "bootstrap_ci95_hi": float(ci_hi),
        "note": "bootstrap resampled by SCENE (the experimental unit, not by individual image), "
                "2000 resamples, 95% percentile CI",
    }
    out_path = STATS_DIR / "scene_aware_bootstrap_ci.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(result)


if __name__ == "__main__":
    compare_trajectories()
    scene_aware_stats()
