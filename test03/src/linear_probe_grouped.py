"""TEST03 Phase 8-9: linear probe with GROUPED cross-validation (group =
scene_id) -- the critical methodological difference from TEST02's plain
StratifiedKFold. No fold ever trains on one degraded version of a scene
while testing on another version of the SAME scene, eliminating any
leakage through shared scene content.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python linear_probe_grouped.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST03 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST03 / "results" / "features"
CLASSIFIERS_DIR = TEST03 / "results" / "classifiers"
VIZ_DIR = TEST03 / "results" / "visualizations"

RANDOM_BASELINE = 100.0 / 3.0
N_SPLITS = 5

TRAJECTORY_ORDER = [
    "input", "shallow_Y0", "encoder_level1", "encoder_level2", "encoder_level3",
    "latent", "AFLB1_aflb_out", "decoder_level3", "AFLB2_aflb_out", "decoder_level2",
    "AFLB3_aflb_out", "decoder_level1", "refinement", "output",
]
TRAJECTORY_LABELS = {
    "input": "Input", "shallow_Y0": "Shallow (Y0)", "encoder_level1": "Encoder L1",
    "encoder_level2": "Encoder L2", "encoder_level3": "Encoder L3", "latent": "Latent",
    "AFLB1_aflb_out": "AFLB 1", "decoder_level3": "Decoder L3 (post-AFLB1)",
    "AFLB2_aflb_out": "AFLB 2", "decoder_level2": "Decoder L2 (post-AFLB2)",
    "AFLB3_aflb_out": "AFLB 3", "decoder_level1": "Decoder L1 (post-AFLB3)",
    "refinement": "Refinement", "output": "Output (restored)",
}


def run_grouped_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    X = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=N_SPLITS)

    results = {}
    for clf_name, clf_ctor in [
        ("logreg", lambda: LogisticRegression(max_iter=2000)),
        ("linear_svm", lambda: LinearSVC(max_iter=5000)),
    ]:
        fold_acc = []
        y_pred_all = np.empty_like(y_enc)
        for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
            # hard leakage assertion: no group in both train and test
            assert not (set(groups[train_idx]) & set(groups[test_idx])), \
                "scene leakage across train/test fold!"
            clf = clf_ctor()
            clf.fit(X[train_idx], y_enc[train_idx])
            pred = clf.predict(X[test_idx])
            y_pred_all[test_idx] = pred
            fold_acc.append(accuracy_score(y_enc[test_idx], pred))

        results[clf_name] = {
            "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
            "balanced_accuracy": balanced_accuracy_score(y_enc, y_pred_all),
            "macro_f1": f1_score(y_enc, y_pred_all, average="macro"),
            "precision": precision_score(y_enc, y_pred_all, average="macro", zero_division=0),
            "recall": recall_score(y_enc, y_pred_all, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_enc, y_pred_all),
        }
    return results


def main():
    rows = []
    cm_records = {}
    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    print(f"{len(npz_files)} features to probe (GroupKFold by scene_id)", flush=True)

    for i, npz_path in enumerate(npz_files):
        feature_name = npz_path.stem
        d = np.load(npz_path, allow_pickle=True)
        X, y, groups = d["X"], d["degradation"], d["scene_id"]
        results = run_grouped_probe(X, y, groups)
        for clf_name, r in results.items():
            rows.append({
                "feature": feature_name, "classifier": clf_name, "n_dim": X.shape[1],
                "n_scenes": len(set(groups)),
                "accuracy_mean": r["accuracy_mean"], "accuracy_std": r["accuracy_std"],
                "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
                "precision": r["precision"], "recall": r["recall"],
                "random_baseline": RANDOM_BASELINE,
            })
            cm_records[(feature_name, clf_name)] = r["confusion_matrix"]
        if (i + 1) % 10 == 0 or i == len(npz_files) - 1:
            print(f"  [{i + 1}/{len(npz_files)}] {feature_name}: "
                  f"logreg_acc={results['logreg']['accuracy_mean']*100:.1f}%", flush=True)

    df = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLASSIFIERS_DIR / "linear_probe_results.csv", index=False)
    print(f"\nwrote {CLASSIFIERS_DIR / 'linear_probe_results.csv'} ({len(df)} rows)")
    print("\nTop 10 (logreg, by accuracy_mean, GROUPED CV):")
    print(df[df.classifier == "logreg"].head(10)[
        ["feature", "accuracy_mean", "balanced_accuracy", "macro_f1"]].to_string(index=False))

    logreg_df = df[df.classifier == "logreg"].set_index("feature")
    traj_rows = []
    for key in TRAJECTORY_ORDER:
        if key not in logreg_df.index:
            continue
        traj_rows.append({
            "stage": TRAJECTORY_LABELS[key], "feature_key": key,
            "accuracy_mean": logreg_df.loc[key, "accuracy_mean"],
            "accuracy_std": logreg_df.loc[key, "accuracy_std"],
            "balanced_accuracy": logreg_df.loc[key, "balanced_accuracy"],
        })
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(CLASSIFIERS_DIR / "feature_trajectory.csv", index=False)
    print(f"\nwrote {CLASSIFIERS_DIR / 'feature_trajectory.csv'}")
    print(traj_df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(traj_df))
    ax.plot(x, traj_df.accuracy_mean * 100, marker="o", color="#c0392b", lw=2,
            label="logreg accuracy (5-fold GROUPED CV, group=scene_id)")
    ax.fill_between(x, (traj_df.accuracy_mean - traj_df.accuracy_std) * 100,
                     (traj_df.accuracy_mean + traj_df.accuracy_std) * 100, alpha=0.2, color="#c0392b")
    ax.axhline(RANDOM_BASELINE, color="gray", ls="--", lw=1, label=f"random baseline ({RANDOM_BASELINE:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(traj_df.stage, rotation=45, ha="right")
    ax.set_ylabel("Rain/Haze/Noise classification accuracy (%)")
    ax.set_title("Controlled same-scene degradation-information trajectory through AdaIR\n"
                 "(grouped by scene_id -- no same-scene leakage between train/test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR / "controlled_degradation_trajectory.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'controlled_degradation_trajectory.png'}")

    best_feature = df[df.classifier == "logreg"].iloc[0]["feature"]
    cm_rows = []
    labels = ["Haze", "Noise", "Rain"]  # LabelEncoder alphabetical order
    for (fname, clf_name), cm in cm_records.items():
        if fname not in TRAJECTORY_ORDER and fname != best_feature:
            continue
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm_rows.append({"feature": fname, "classifier": clf_name,
                                 "true_label": true_label, "predicted_label": pred_label,
                                 "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv(CLASSIFIERS_DIR / "confusion_matrices.csv", index=False)
    print(f"wrote {CLASSIFIERS_DIR / 'confusion_matrices.csv'} (best feature: {best_feature})")


if __name__ == "__main__":
    main()
