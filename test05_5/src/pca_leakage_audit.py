"""TEST05.5 Phase 2: STRICT leakage-safe PCA audit, correcting TEST05's
loophole (StandardScaler + PCA were fit on the full 300-image set before
grouped CV, even though the classifier itself was fit per-fold -- a real
leakage risk since PCA components could "see" validation-fold images
during fitting).

For every fold: fit scaler+PCA on TRAINING scenes only, transform both
splits, fit classifier on training only, evaluate on validation only.
Never fit any transform on data outside the current training fold.

Reads TEST05's already-extracted latent_pre.npz (read-only reference, not
modified) -- the pooled GAP+GMP feature vectors themselves are unchanged;
only the CV procedure applied to them is corrected here.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python pca_leakage_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
LATENT_NPZ = TEACHER_EXP / "test05" / "results" / "feature_analysis" / "latent_pre.npz"
OUT_DIR = TEST05_5 / "results" / "pca_audit"
DIMS = [4, 8, 16, 32, 64, 128]
RANDOM_BASELINE = 100.0 / 3.0
N_SPLITS = 5


def leakage_safe_pca_cv(X_raw, y, groups, n_components):
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_accs, fold_f1s = [], []

    for train_idx, test_idx in gkf.split(X_raw, y_enc, groups=groups):
        assert not (set(groups[train_idx]) & set(groups[test_idx])), "scene leakage!"

        scaler = StandardScaler().fit(X_raw[train_idx])
        X_train_scaled = scaler.transform(X_raw[train_idx])
        X_test_scaled = scaler.transform(X_raw[test_idx])

        pca = PCA(n_components=n_components, random_state=0).fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train_pca, y_enc[train_idx])
        pred = clf.predict(X_test_pca)

        fold_accs.append(accuracy_score(y_enc[test_idx], pred))
        fold_f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))

    fold_accs = np.array(fold_accs)
    mean_acc = fold_accs.mean()
    se = fold_accs.std(ddof=1) / np.sqrt(N_SPLITS)
    ci_lo, ci_hi = mean_acc - 1.96 * se, mean_acc + 1.96 * se
    return {
        "mean_accuracy": float(mean_acc), "std_accuracy": float(fold_accs.std(ddof=1)),
        "ci95_lo": float(max(0, ci_lo)), "ci95_hi": float(min(1, ci_hi)),
        "macro_f1_mean": float(np.mean(fold_f1s)),
    }


def main():
    d = np.load(LATENT_NPZ, allow_pickle=True)
    X_raw = np.concatenate([d["X_gap"], d["X_gmp"]], axis=1)
    y, groups = d["degradation"], d["scene_id"]
    print(f"latent_pre pooled vector: {X_raw.shape}", flush=True)

    # ---- also redo the FULL (no PCA) leakage-safe baseline for comparison ----
    rows = []
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=N_SPLITS)
    full_accs = []
    for train_idx, test_idx in gkf.split(X_raw, y_enc, groups=groups):
        scaler = StandardScaler().fit(X_raw[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X_raw[train_idx]), y_enc[train_idx])
        pred = clf.predict(scaler.transform(X_raw[test_idx]))
        full_accs.append(accuracy_score(y_enc[test_idx], pred))
    full_accs = np.array(full_accs)
    se = full_accs.std(ddof=1) / np.sqrt(N_SPLITS)
    rows.append({"representation": "full_768d_leakage_safe", "dim": X_raw.shape[1],
                  "mean_accuracy": float(full_accs.mean()), "std_accuracy": float(full_accs.std(ddof=1)),
                  "ci95_lo": float(full_accs.mean() - 1.96 * se), "ci95_hi": float(full_accs.mean() + 1.96 * se)})
    print(f"Full 768-dim (leakage-safe, no PCA): {full_accs.mean()*100:.2f}% "
          f"+/- {full_accs.std(ddof=1)*100:.2f}%", flush=True)

    for dim in DIMS:
        r = leakage_safe_pca_cv(X_raw, y, groups, dim)
        rows.append({"representation": f"pca_{dim}_leakage_safe", "dim": dim,
                      "mean_accuracy": r["mean_accuracy"], "std_accuracy": r["std_accuracy"],
                      "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"], "macro_f1": r["macro_f1_mean"]})
        print(f"PCA-{dim} (leakage-safe): {r['mean_accuracy']*100:.2f}% "
              f"(95% CI [{r['ci95_lo']*100:.2f}, {r['ci95_hi']*100:.2f}]) macro_f1={r['macro_f1_mean']:.3f}",
              flush=True)

    df_out = pd.DataFrame(rows)
    df_out["random_baseline"] = RANDOM_BASELINE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_DIR / "pca_leakage_safe_results.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'pca_leakage_safe_results.csv'}")

    # ---- comparison against TEST05's original (leaky) numbers, read-only reference ----
    orig_path = TEACHER_EXP / "test05" / "results" / "statistics" / "compact_embedding.csv"
    if orig_path.exists():
        orig = pd.read_csv(orig_path)
        orig_pca = orig[orig.representation.str.startswith("pca_")]
        print("\nComparison vs TEST05's original (leaky) PCA numbers:")
        for _, r in orig_pca.iterrows():
            dim = int(r["representation"].replace("pca_", ""))
            safe_row = df_out[df_out.representation == f"pca_{dim}_leakage_safe"]
            if len(safe_row):
                print(f"  PCA-{dim}: TEST05 (leaky)={r['accuracy']*100:.1f}%  "
                      f"TEST05.5 (leakage-safe)={safe_row.iloc[0]['mean_accuracy']*100:.1f}%  "
                      f"diff={  (safe_row.iloc[0]['mean_accuracy'] - r['accuracy'])*100:+.2f}pp")


if __name__ == "__main__":
    main()
