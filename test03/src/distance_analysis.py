"""TEST03 Phase 10-11: THE key new analysis TEST02 could not do.

For every feature, for every scene, compute same-scene cross-degradation
distances (rain_i vs haze_i, rain_i vs noise_i, haze_i vs noise_i -- i.e.
"same content, different degradation") AND same-degradation cross-scene
distances (rain_i vs rain_j for i!=j -- "same degradation, different
content"). Then:

    degradation_ratio = mean(D_degradation) / mean(D_scene)

If degradation_ratio >> 1: changing ONLY the degradation moves the
representation more than changing the scene does -- strong evidence for
genuine degradation-specific encoding, not dataset/domain artifact.
If degradation_ratio ~= 1: scene identity dominates the representation at
least as much as degradation does.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python distance_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import StandardScaler

TEST03 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST03 / "results" / "features"
STATS_DIR = TEST03 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
DEG_PAIRS = [("Rain", "Haze"), ("Rain", "Noise"), ("Haze", "Noise")]
RNG = np.random.RandomState(0)
MAX_SCENE_PAIRS_PER_DEG = 500  # cap cross-scene pair sampling for speed on large feature sets


def dist(a, b, metric):
    if metric == "euclidean":
        return euclidean(a, b)
    return cosine(a, b) if (np.any(a) and np.any(b)) else np.nan


def main():
    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    paired_rows = []
    ratio_rows = []

    for npz_path in npz_files:
        feature_name = npz_path.stem
        d = np.load(npz_path, allow_pickle=True)
        X_raw, degs, scenes = d["X"], d["degradation"], d["scene_id"]
        X = StandardScaler().fit_transform(X_raw)

        by_scene_deg = {(s, g): X[i] for i, (s, g) in enumerate(zip(scenes, degs))}
        scene_ids = sorted(set(scenes))

        for metric in ["euclidean", "cosine"]:
            same_scene_dists = []
            for scene_id in scene_ids:
                for da, db in DEG_PAIRS:
                    if (scene_id, da) in by_scene_deg and (scene_id, db) in by_scene_deg:
                        dd = dist(by_scene_deg[(scene_id, da)], by_scene_deg[(scene_id, db)], metric)
                        same_scene_dists.append(dd)
                        paired_rows.append({"feature": feature_name, "metric": metric, "scene_id": scene_id,
                                             "comparison_type": "same_scene_diff_degradation",
                                             "pair": f"{da}-{db}", "distance": dd})

            cross_scene_dists = []
            for deg in DEGS:
                deg_scene_ids = [s for s in scene_ids if (s, deg) in by_scene_deg]
                n = len(deg_scene_ids)
                all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                if len(all_pairs) > MAX_SCENE_PAIRS_PER_DEG:
                    idxs = RNG.choice(len(all_pairs), size=MAX_SCENE_PAIRS_PER_DEG, replace=False)
                    sampled_pairs = [all_pairs[k] for k in idxs]
                else:
                    sampled_pairs = all_pairs
                for i, j in sampled_pairs:
                    s1, s2 = deg_scene_ids[i], deg_scene_ids[j]
                    dd = dist(by_scene_deg[(s1, deg)], by_scene_deg[(s2, deg)], metric)
                    cross_scene_dists.append(dd)

            d_degradation = float(np.nanmean(same_scene_dists))
            d_scene = float(np.nanmean(cross_scene_dists))
            ratio = d_degradation / d_scene if d_scene > 0 else float("nan")
            ratio_rows.append({
                "feature": feature_name, "metric": metric,
                "D_degradation_same_scene": d_degradation, "D_scene_same_degradation": d_scene,
                "degradation_ratio": ratio,
                "n_same_scene_pairs": len(same_scene_dists), "n_cross_scene_pairs": len(cross_scene_dists),
            })

    paired_df = pd.DataFrame(paired_rows)
    ratio_df = pd.DataFrame(ratio_rows).sort_values(["metric", "degradation_ratio"], ascending=[True, False])

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(STATS_DIR / "paired_distance_analysis.csv", index=False)
    ratio_df.to_csv(STATS_DIR / "degradation_vs_scene.csv", index=False)

    print(f"wrote {STATS_DIR / 'paired_distance_analysis.csv'} ({len(paired_df)} rows)")
    print(f"wrote {STATS_DIR / 'degradation_vs_scene.csv'} ({len(ratio_df)} rows)")
    print("\nTop 15 by degradation_ratio (euclidean) -- D_degradation >> D_scene means "
          "degradation dominates over scene content:")
    print(ratio_df[ratio_df.metric == "euclidean"].head(15)[
        ["feature", "D_degradation_same_scene", "D_scene_same_degradation", "degradation_ratio"]
    ].to_string(index=False))
    print("\nBottom 5 by degradation_ratio (euclidean) -- weakest degradation-vs-scene separation:")
    print(ratio_df[ratio_df.metric == "euclidean"].tail(5)[
        ["feature", "D_degradation_same_scene", "D_scene_same_degradation", "degradation_ratio"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
