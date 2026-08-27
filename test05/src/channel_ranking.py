"""TEST05 Phase 6: channel-level analysis -- the most important correlational
analysis in this experiment. For every candidate feature's per-channel GAP
vector, rank each channel by:
  - degradation_probe_accuracy: single-feature (1-D) grouped-CV logistic
    regression accuracy using ONLY that channel's GAP value
  - degradation_distance / scene_distance / degradation_scene_ratio: same
    same-scene vs cross-scene distance ratio as Phase 4-5, computed on the
    1-D channel value
  - variance, energy: raw descriptive stats

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python channel_ranking.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST05 / "results" / "feature_analysis"
STATS_DIR = TEST05 / "results" / "channel_analysis"
DEGS = ["Rain", "Haze", "Noise"]
DEG_PAIRS = [("Rain", "Haze"), ("Rain", "Noise"), ("Haze", "Noise")]
RANDOM_BASELINE = 100.0 / 3.0
RNG = np.random.RandomState(0)

# Only run full channel ranking on these candidates -- the ones with real
# spatial extent and plausible distillation relevance. Skip 1-channel
# scalar-ish features (hl_spatial_weight after pooling is nearly scalar)
# and raw_low (negative control, handled separately/trivially since it's
# exactly zero -- see Raw_Low note in the report, not worth per-channel
# compute).
TARGET_FEATURES = [
    "latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out",
    "AFLB1_mined_low", "AFLB1_mined_high", "AFLB1_fmom_agg", "AFLB1_raw_high",
    "AFLB2_mined_low", "AFLB2_mined_high", "AFLB2_fmom_agg",
    "AFLB3_mined_low", "AFLB3_mined_high", "AFLB3_fmom_agg",
    "AFLB1_lh_channel_weight", "AFLB1_cross_agg_out",
]


def channel_probe_accuracy(x_1d: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    X = StandardScaler().fit_transform(x_1d.reshape(-1, 1))
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=5)
    accs = []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(accs))


def channel_distance_ratio(x_1d: np.ndarray, degs: np.ndarray, scenes: np.ndarray) -> float:
    by_scene_deg = {(s, g): v for s, g, v in zip(scenes, degs, x_1d)}
    scene_ids = sorted(set(scenes))
    same_scene = []
    for scene_id in scene_ids:
        for da, db in DEG_PAIRS:
            if (scene_id, da) in by_scene_deg and (scene_id, db) in by_scene_deg:
                same_scene.append(abs(by_scene_deg[(scene_id, da)] - by_scene_deg[(scene_id, db)]))
    cross_scene = []
    for deg in DEGS:
        deg_scenes = [s for s in scene_ids if (s, deg) in by_scene_deg]
        vals = np.array([by_scene_deg[(s, deg)] for s in deg_scenes])
        n = len(vals)
        idxs = RNG.choice(n, size=min(200, n * (n - 1) // 2 or 1), replace=True) if n > 1 else []
        # sample random pairs for speed at per-channel granularity
        for _ in range(min(200, n * (n - 1) // 2) if n > 1 else 0):
            i, j = RNG.choice(n, size=2, replace=False)
            cross_scene.append(abs(vals[i] - vals[j]))
    d_deg = float(np.mean(same_scene)) if same_scene else float("nan")
    d_scene = float(np.mean(cross_scene)) if cross_scene else float("nan")
    return d_deg, d_scene, (d_deg / d_scene if d_scene and d_scene > 0 else float("nan"))


def main():
    all_rows = []
    for fname in TARGET_FEATURES:
        npz_path = FEATURES_DIR / f"{fname}.npz"
        if not npz_path.exists():
            print(f"SKIP {fname}: not found")
            continue
        d = np.load(npz_path, allow_pickle=True)
        X_gap, degs, scenes = d["X_gap"], d["degradation"], d["scene_id"]
        n_channels = X_gap.shape[1]
        print(f"{fname}: {n_channels} channels", flush=True)

        for c in range(n_channels):
            x_1d = X_gap[:, c]
            acc = channel_probe_accuracy(x_1d, degs, scenes)
            d_deg, d_scene, ratio = channel_distance_ratio(x_1d, degs, scenes)
            all_rows.append({
                "feature": fname, "channel": c,
                "degradation_probe_accuracy": acc,
                "degradation_distance": d_deg, "scene_distance": d_scene,
                "degradation_scene_ratio": ratio,
                "variance": float(np.var(x_1d)), "energy": float(np.sum(x_1d ** 2)),
            })

    df = pd.DataFrame(all_rows).sort_values("degradation_probe_accuracy", ascending=False)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATS_DIR / "channel_rank.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'channel_rank.csv'} ({len(df)} rows)")
    print("\nTop 20 channels overall by degradation_probe_accuracy:")
    print(df.head(20)[["feature", "channel", "degradation_probe_accuracy", "degradation_scene_ratio"]].to_string(index=False))

    print("\nTop 10 channels by degradation_scene_ratio (>random-accuracy channels only):")
    good = df[df.degradation_probe_accuracy > RANDOM_BASELINE / 100 + 0.1]
    print(good.sort_values("degradation_scene_ratio", ascending=False).head(10)[
        ["feature", "channel", "degradation_probe_accuracy", "degradation_scene_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
