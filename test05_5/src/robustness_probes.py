"""TEST05.5 Phase 3-4 analysis: degradation-FAMILY probe (both severity
bands pooled, grouped CV by scene) and cross-severity generalization
(train on band A, test on band B, and vice versa).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python robustness_probes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05_5 = Path(__file__).resolve().parent.parent
ROBUST_DIR = TEST05_5 / "results" / "robustness"
CANDIDATES = ["input", "latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]
RANDOM_BASELINE = 100.0 / 3.0


def family_probe(X, y, groups):
    Xs = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(Xs, y_enc, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xs[train_idx], y_enc[train_idx])
        pred = clf.predict(Xs[test_idx])
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def cross_severity(X, y, band, groups, train_band, test_band):
    scaler = StandardScaler().fit(X[band == train_band])
    y_enc_full = LabelEncoder().fit(y)
    train_mask, test_mask = band == train_band, band == test_band
    X_train, X_test = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])
    y_train, y_test = y_enc_full.transform(y[train_mask]), y_enc_full.transform(y[test_mask])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred)), float(f1_score(y_test, pred, average="macro"))


def main():
    family_rows, severity_rows = [], []
    for cand in CANDIDATES:
        npz_path = ROBUST_DIR / f"{cand}.npz"
        if not npz_path.exists():
            print(f"SKIP {cand}: not found")
            continue
        d = np.load(npz_path, allow_pickle=True)
        X, y, band, groups = d["X"], d["degradation"], d["band"], d["scene_id"]

        acc, std, f1 = family_probe(X, y, groups)
        family_rows.append({"candidate": cand, "n_dim": X.shape[1], "accuracy_mean": acc,
                             "accuracy_std": std, "macro_f1": f1, "n_images": len(X),
                             "random_baseline": RANDOM_BASELINE})
        print(f"{cand}: family-probe (both bands pooled) accuracy = {acc*100:.1f}% (+/-{std*100:.1f})", flush=True)

        acc_ab, f1_ab = cross_severity(X, y, band, groups, "A", "B")
        acc_ba, f1_ba = cross_severity(X, y, band, groups, "B", "A")
        severity_rows.append({"candidate": cand, "train_band": "A", "test_band": "B",
                               "accuracy": acc_ab, "macro_f1": f1_ab})
        severity_rows.append({"candidate": cand, "train_band": "B", "test_band": "A",
                               "accuracy": acc_ba, "macro_f1": f1_ba})
        print(f"  train=A/test=B: {acc_ab*100:.1f}%   train=B/test=A: {acc_ba*100:.1f}%", flush=True)

    fam_df = pd.DataFrame(family_rows).sort_values("accuracy_mean", ascending=False)
    sev_df = pd.DataFrame(severity_rows)
    fam_df.to_csv(ROBUST_DIR / "family_probe_results.csv", index=False)
    sev_df.to_csv(ROBUST_DIR / "severity_generalization_results.csv", index=False)
    print(f"\nwrote family_probe_results.csv, severity_generalization_results.csv")
    print(fam_df.to_string(index=False))
    print()
    print(sev_df.to_string(index=False))


if __name__ == "__main__":
    main()
