"""Phase 6: for every pooled feature (results/features/*.npz), compute
within-class (Rain-Rain, Haze-Haze, Noise-Noise) and between-class
(Rain-Haze, Rain-Noise, Haze-Noise) pairwise distances (Euclidean + cosine),
their means, and a separation_ratio = mean(inter) / mean(intra).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python distance_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from sklearn.preprocessing import StandardScaler

TEST02 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST02 / "results" / "features"
OUT_PATH = TEST02 / "results" / "statistics" / "degradation_separation.csv"

DEGS = ["Rain", "Haze", "Noise"]
PAIRS_WITHIN = [("Rain", "Rain"), ("Haze", "Haze"), ("Noise", "Noise")]
PAIRS_BETWEEN = [("Rain", "Haze"), ("Rain", "Noise"), ("Haze", "Noise")]


def mean_pdist(X: np.ndarray, metric: str) -> float:
    if len(X) < 2:
        return float("nan")
    return float(np.mean(pdist(X, metric=metric)))


def mean_cdist(Xa: np.ndarray, Xb: np.ndarray, metric: str) -> float:
    if len(Xa) == 0 or len(Xb) == 0:
        return float("nan")
    return float(np.mean(cdist(Xa, Xb, metric=metric)))


def main():
    rows = []
    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    for npz_path in npz_files:
        feature_name = npz_path.stem
        d = np.load(npz_path, allow_pickle=True)
        X, degs = d["X"], d["degradation"]
        X = StandardScaler().fit_transform(X)  # standardize so distance scales are comparable across features

        by_deg = {deg: X[degs == deg] for deg in DEGS}

        for metric in ["euclidean", "cosine"]:
            intra_vals = [mean_pdist(by_deg[a], metric) for a, b in PAIRS_WITHIN]
            inter_vals = [mean_cdist(by_deg[a], by_deg[b], metric) for a, b in PAIRS_BETWEEN]
            intra_mean = float(np.nanmean(intra_vals))
            inter_mean = float(np.nanmean(inter_vals))
            sep_ratio = inter_mean / intra_mean if intra_mean > 0 else float("nan")

            row = {"feature": feature_name, "metric": metric,
                   "intra_class_mean_distance": intra_mean, "inter_class_mean_distance": inter_mean,
                   "separation_ratio": sep_ratio}
            for (a, b), v in zip(PAIRS_WITHIN, intra_vals):
                row[f"intra_{a}"] = v
            for (a, b), v in zip(PAIRS_BETWEEN, inter_vals):
                row[f"inter_{a}_{b}"] = v
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["metric", "separation_ratio"], ascending=[True, False])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} rows)")
    print("\nTop 10 by separation_ratio (euclidean):")
    print(out[out.metric == "euclidean"].head(10)[["feature", "separation_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
