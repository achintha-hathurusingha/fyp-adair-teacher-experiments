"""Phase 7-9 + 15: for every pooled feature, train a LINEAR classifier
(Logistic Regression and Linear SVM -- both external to AdaIR, not part of
the model) to predict Rain/Haze/Noise from the feature vector, via 5-fold
STRATIFIED cross-validation. Reports accuracy, balanced accuracy, macro F1,
precision, recall (macro-averaged), and saves a confusion matrix for the
best-performing feature per classifier.

This measures how LINEARLY separable degradation information is in each
representation -- not whether AdaIR "knows" the degradation type.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python linear_probe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

TEST02 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST02 / "results" / "features"
CLASSIFIERS_DIR = TEST02 / "results" / "classifiers"
VIZ_DIR = TEST02 / "results" / "visualizations"

RANDOM_BASELINE = 100.0 / 3.0
N_SPLITS = 5
SEED = 0

# Main pipeline order for the trajectory table/plot (Phase 9). Not every
# extracted feature belongs on this single-axis "depth" plot -- AFLB
# internals (FMiM/FMoM sub-features) are analysed separately in Phase 12.
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


def run_probe(X: np.ndarray, y: np.ndarray) -> dict:
    X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {}
    for clf_name, clf in [
        ("logreg", LogisticRegression(max_iter=2000)),
        ("linear_svm", LinearSVC(max_iter=5000)),
    ]:
        fold_acc = []
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            fold_acc.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        results[clf_name] = {
            "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
            "precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred, labels=["Rain", "Haze", "Noise"]),
        }
    return results


def main():
    rows = []
    cm_records = {}
    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    print(f"{len(npz_files)} features to probe", flush=True)

    for i, npz_path in enumerate(npz_files):
        feature_name = npz_path.stem
        d = np.load(npz_path, allow_pickle=True)
        X, y = d["X"], d["degradation"]
        results = run_probe(X, y)
        for clf_name, r in results.items():
            rows.append({
                "feature": feature_name, "classifier": clf_name, "n_dim": X.shape[1],
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
    print("\nTop 10 (logreg, by accuracy_mean):")
    print(df[df.classifier == "logreg"].head(10)[
        ["feature", "accuracy_mean", "balanced_accuracy", "macro_f1"]].to_string(index=False))

    # -------- Phase 9: feature trajectory table + plot --------
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
    ax.plot(x, traj_df.accuracy_mean * 100, marker="o", color="#3b7dd8", lw=2, label="logreg accuracy (5-fold CV)")
    ax.fill_between(x, (traj_df.accuracy_mean - traj_df.accuracy_std) * 100,
                     (traj_df.accuracy_mean + traj_df.accuracy_std) * 100, alpha=0.2, color="#3b7dd8")
    ax.axhline(RANDOM_BASELINE, color="gray", ls="--", lw=1, label=f"random baseline ({RANDOM_BASELINE:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(traj_df.stage, rotation=45, ha="right")
    ax.set_ylabel("Rain/Haze/Noise classification accuracy (%)")
    ax.set_title("Degradation-information trajectory through AdaIR\n(linear probe on GAP+GMP pooled features)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ_DIR / "degradation_information_trajectory.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'degradation_information_trajectory.png'}")

    # -------- Phase 15: confusion matrices for best features --------
    best_feature = df[df.classifier == "logreg"].iloc[0]["feature"]
    cm_rows = []
    for (fname, clf_name), cm in cm_records.items():
        if fname not in TRAJECTORY_ORDER and fname != best_feature:
            continue
        for i, true_label in enumerate(["Rain", "Haze", "Noise"]):
            for j, pred_label in enumerate(["Rain", "Haze", "Noise"]):
                cm_rows.append({"feature": fname, "classifier": clf_name,
                                 "true_label": true_label, "predicted_label": pred_label,
                                 "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv(CLASSIFIERS_DIR / "confusion_matrices.csv", index=False)
    print(f"wrote {CLASSIFIERS_DIR / 'confusion_matrices.csv'} (best feature: {best_feature})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key in zip(axes, [TRAJECTORY_ORDER[0], best_feature, TRAJECTORY_ORDER[-1]]):
        cm = cm_records[(key, "logreg")]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(["Rain", "Haze", "Noise"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["Rain", "Haze", "Noise"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{key}\n(logreg)")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "confusion_matrices_input_best_output.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VIZ_DIR / 'confusion_matrices_input_best_output.png'}")


if __name__ == "__main__":
    main()
