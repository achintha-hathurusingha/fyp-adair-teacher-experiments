"""TEST05 Phase 3-5: pooled-level (GAP+GMP) grouped linear probe (reproduces
TEST03's methodology for the extended candidate set) + same-scene vs
cross-scene distance ratio analysis (reproduces TEST03's Phase 10-11 for
the extended candidate set). This is the "full tensor" baseline that
Phase 6+ channel-level analysis will be compared against.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python primary_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST05 / "results" / "feature_analysis"
STATS_DIR = TEST05 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
DEG_PAIRS = [("Rain", "Haze"), ("Rain", "Noise"), ("Haze", "Noise")]
RANDOM_BASELINE = 100.0 / 3.0
RNG = np.random.RandomState(0)
MAX_SCENE_PAIRS = 500


def pooled_vec(d):
    return np.concatenate([d["X_gap"], d["X_gmp"]], axis=1)


def grouped_probe(X, y, groups):
    X = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=5)
    fold_acc, y_pred_all = [], np.empty_like(y_enc)
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y_enc[train_idx])
        pred = clf.predict(X[test_idx])
        y_pred_all[test_idx] = pred
        fold_acc.append(accuracy_score(y_enc[test_idx], pred))
    return {
        "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
        "balanced_accuracy": balanced_accuracy_score(y_enc, y_pred_all),
        "macro_f1": f1_score(y_enc, y_pred_all, average="macro"),
    }


def dist(a, b, metric):
    if metric == "euclidean":
        return euclidean(a, b)
    return cosine(a, b) if (np.any(a) and np.any(b)) else np.nan


def distance_ratio(X, degs, scenes):
    by_scene_deg = {(s, g): X[i] for i, (s, g) in enumerate(zip(scenes, degs))}
    scene_ids = sorted(set(scenes))
    out = {}
    for metric in ["euclidean", "cosine"]:
        same_scene = []
        for scene_id in scene_ids:
            for da, db in DEG_PAIRS:
                if (scene_id, da) in by_scene_deg and (scene_id, db) in by_scene_deg:
                    same_scene.append(dist(by_scene_deg[(scene_id, da)], by_scene_deg[(scene_id, db)], metric))
        cross_scene = []
        for deg in DEGS:
            deg_scenes = [s for s in scene_ids if (s, deg) in by_scene_deg]
            n = len(deg_scenes)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            if len(pairs) > MAX_SCENE_PAIRS:
                idxs = RNG.choice(len(pairs), size=MAX_SCENE_PAIRS, replace=False)
                pairs = [pairs[k] for k in idxs]
            for i, j in pairs:
                cross_scene.append(dist(by_scene_deg[(deg_scenes[i], deg)], by_scene_deg[(deg_scenes[j], deg)], metric))
        d_deg, d_scene = float(np.nanmean(same_scene)), float(np.nanmean(cross_scene))
        out[metric] = {"D_degradation": d_deg, "D_scene": d_scene,
                        "ratio": d_deg / d_scene if d_scene > 0 else float("nan")}
    return out


def main():
    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    probe_rows, dist_rows = [], []
    print(f"{len(npz_files)} candidate features", flush=True)

    for i, npz_path in enumerate(npz_files):
        fname = npz_path.stem
        d = np.load(npz_path, allow_pickle=True)
        X, y, groups = pooled_vec(d), d["degradation"], d["scene_id"]

        r = grouped_probe(X, y, groups)
        probe_rows.append({"feature": fname, "n_dim": X.shape[1], **r, "random_baseline": RANDOM_BASELINE})

        dr = distance_ratio(X, y, groups)
        for metric, vals in dr.items():
            dist_rows.append({"feature": fname, "metric": metric, **vals})

        if (i + 1) % 10 == 0 or i == len(npz_files) - 1:
            print(f"  [{i + 1}/{len(npz_files)}] {fname}: acc={r['accuracy_mean']*100:.1f}% "
                  f"ratio(euclid)={dr['euclidean']['ratio']:.2f}", flush=True)

    probe_df = pd.DataFrame(probe_rows).sort_values("accuracy_mean", ascending=False)
    dist_df = pd.DataFrame(dist_rows).sort_values(["metric", "ratio"], ascending=[True, False])

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(STATS_DIR / "linear_probe.csv", index=False)
    dist_df.to_csv(STATS_DIR / "scene_sensitivity.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'linear_probe.csv'} ({len(probe_df)} rows)")
    print(f"wrote {STATS_DIR / 'scene_sensitivity.csv'} ({len(dist_df)} rows)")

    print("\nTop 10 by accuracy:")
    print(probe_df.head(10)[["feature", "accuracy_mean"]].to_string(index=False))
    print("\nTop 10 by degradation_scene_ratio (euclidean):")
    print(dist_df[dist_df.metric == "euclidean"].head(10)[["feature", "D_degradation", "D_scene", "ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
