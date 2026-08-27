"""TEST03 Phase 13b: alpha/beta distributions + GROUPED (by scene_id)
classifier using ONLY [alpha,beta] as input. Mirrors test02's approach but
with GroupKFold instead of plain StratifiedKFold.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python alpha_beta_analysis.py
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST03 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST03 / "results" / "statistics"
VIZ_DIR = TEST03 / "results" / "visualizations"
CLASSIFIERS_DIR = TEST03 / "results" / "classifiers"
DEG_COLORS = {"Rain": "#3b7dd8", "Haze": "#d8853b", "Noise": "#3bb273"}
DEG_ORDER = ["Rain", "Haze", "Noise"]
RANDOM_BASELINE = 100.0 / 3.0


def grouped_accuracy(X, y, groups):
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


def main():
    df = pd.read_csv(STATS_DIR / "alpha_beta.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col_idx, aflb in enumerate(["AFLB1", "AFLB2", "AFLB3"]):
        sub = df[df.AFLB == aflb]
        for row_idx, param in enumerate(["alpha", "beta"]):
            ax = axes[row_idx, col_idx]
            data = [sub[sub.degradation == d][param] for d in DEG_ORDER]
            bp = ax.boxplot(data, tick_labels=DEG_ORDER, patch_artist=True)
            for patch, d in zip(bp["boxes"], DEG_ORDER):
                patch.set_facecolor(DEG_COLORS[d]); patch.set_alpha(0.6)
            ax.set_title(f"{aflb}: {param}")
            ax.grid(alpha=0.3)
    fig.suptitle("Alpha/Beta distributions by degradation and AFLB (controlled same-scene design)")
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR / "alpha_beta_distributions.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'alpha_beta_distributions.png'}")

    results = []
    for aflb in ["AFLB1", "AFLB2", "AFLB3", "ALL_AFLB_COMBINED"]:
        if aflb == "ALL_AFLB_COMBINED":
            wide = df.pivot_table(index=["scene_id", "degradation"], columns="AFLB", values=["alpha", "beta"])
            wide.columns = [f"{a}_{b}" for a, b in wide.columns]
            wide = wide.reset_index()
            X = wide[[c for c in wide.columns if c not in ("scene_id", "degradation")]].values
            y = wide["degradation"].values
            groups = wide["scene_id"].values
        else:
            sub = df[df.AFLB == aflb]
            X = sub[["alpha", "beta"]].values
            y = sub["degradation"].values
            groups = sub["scene_id"].values

        r = grouped_accuracy(X, y, groups)
        results.append({"input": aflb, "n_dim": X.shape[1], **r,
                         "random_baseline": RANDOM_BASELINE,
                         "above_random": r["accuracy_mean"] * 100 - RANDOM_BASELINE})

    out = pd.DataFrame(results)
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLASSIFIERS_DIR / "alpha_beta_probe_results.csv", index=False)
    print(f"\nwrote {CLASSIFIERS_DIR / 'alpha_beta_probe_results.csv'}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
