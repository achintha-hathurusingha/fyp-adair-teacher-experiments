"""Phase 13: does [alpha, beta] alone contain measurable degradation
information? Distribution plots per degradation, per AFLB, and a linear
probe using ONLY [alpha, beta] (2 features) as input.

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

TEST02 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST02 / "results" / "statistics"
VIZ_DIR = TEST02 / "results" / "visualizations"
CLASSIFIERS_DIR = TEST02 / "results" / "classifiers"
DEG_COLORS = {"Rain": "#3b7dd8", "Haze": "#d8853b", "Noise": "#3bb273"}
DEG_ORDER = ["Rain", "Haze", "Noise"]
RANDOM_BASELINE = 100.0 / 3.0


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
    fig.suptitle("Alpha/Beta distributions by degradation and AFLB")
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR / "alpha_beta_distributions.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'alpha_beta_distributions.png'}")

    # linear probe using [alpha, beta] per AFLB, and all 6 (3 AFLB x 2) combined
    results = []
    for aflb in ["AFLB1", "AFLB2", "AFLB3", "ALL_AFLB_COMBINED"]:
        if aflb == "ALL_AFLB_COMBINED":
            piv = df.pivot(index="image_id", columns="AFLB", values=["alpha", "beta"])
            piv.columns = [f"{a}_{b}" for a, b in piv.columns]
            labels = df.drop_duplicates("image_id").set_index("image_id")["degradation"]
            X = piv.values
            y = labels.loc[piv.index].values
        else:
            sub = df[df.AFLB == aflb]
            X = sub[["alpha", "beta"]].values
            y = sub["degradation"].values

        X = StandardScaler().fit_transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        clf = LogisticRegression(max_iter=2000)
        fold_acc = []
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            fold_acc.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        results.append({
            "input": aflb, "n_dim": X.shape[1],
            "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
            "random_baseline": RANDOM_BASELINE,
            "above_random": float(np.mean(fold_acc)) * 100 - RANDOM_BASELINE,
        })

    out = pd.DataFrame(results)
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLASSIFIERS_DIR / "alpha_beta_probe_results.csv", index=False)
    print(f"\nwrote {CLASSIFIERS_DIR / 'alpha_beta_probe_results.csv'}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
