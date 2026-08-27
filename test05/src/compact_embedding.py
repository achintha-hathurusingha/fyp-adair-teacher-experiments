"""TEST05 Phase 16-17: compact embedding experiment. PCA-project latent_pre
(the top pooled candidate) down to 16/32/64/128 dims and re-run the grouped
linear probe + scene-sensitivity ratio at each size, to test whether
degradation-aware information survives aggressive compression. Also
directly compares against alpha/beta (Phase 17).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python compact_embedding.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05 = Path(__file__).resolve().parent.parent
FEATURES_DIR = TEST05 / "results" / "feature_analysis"
STATS_DIR = TEST05 / "results" / "statistics"
DIMS = [16, 32, 64, 128]
RANDOM_BASELINE = 100.0 / 3.0


def grouped_probe_acc(X, y, groups):
    X = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=5)
    accs = []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(accs))


def main():
    d = np.load(FEATURES_DIR / "latent_pre.npz", allow_pickle=True)
    X_full = np.concatenate([d["X_gap"], d["X_gmp"]], axis=1)  # 768-dim, matches primary_analysis
    y, groups = d["degradation"], d["scene_id"]

    full_acc = grouped_probe_acc(X_full, y, groups)
    print(f"Full latent_pre (768-dim, GAP+GMP): grouped-CV accuracy = {full_acc*100:.1f}%", flush=True)

    rows = [{"representation": "full_latent_pre", "dim": X_full.shape[1], "accuracy": full_acc,
             "explained_variance_pct": 100.0}]

    Xs = StandardScaler().fit_transform(X_full)
    for dim in DIMS:
        pca = PCA(n_components=dim, random_state=0)
        X_proj = pca.fit_transform(Xs)
        acc = grouped_probe_acc(X_proj, y, groups)
        explained = float(pca.explained_variance_ratio_.sum() * 100)
        rows.append({"representation": f"pca_{dim}", "dim": dim, "accuracy": acc,
                     "explained_variance_pct": explained})
        print(f"  PCA-{dim}: accuracy = {acc*100:.1f}%  (explains {explained:.1f}% variance)", flush=True)

    # alpha/beta comparison (Phase 17)
    ab = pd.read_csv(STATS_DIR / "alpha_beta.csv")
    for aflb in ["AFLB1", "AFLB2", "AFLB3"]:
        sub = ab[ab.AFLB == aflb]
        X_ab = sub[["alpha", "beta"]].to_numpy()
        acc_ab = grouped_probe_acc(X_ab, sub["degradation"].to_numpy(), sub["scene_id"].to_numpy())
        rows.append({"representation": f"{aflb}_alpha_beta", "dim": 2, "accuracy": acc_ab,
                     "explained_variance_pct": np.nan})
        print(f"  {aflb} [alpha,beta] (2-dim): accuracy = {acc_ab*100:.1f}%", flush=True)

    wide = ab.pivot_table(index=["scene_id", "degradation"], columns="AFLB", values=["alpha", "beta"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    X_ab_all = wide[[c for c in wide.columns if c not in ("scene_id", "degradation")]].to_numpy()
    acc_ab_all = grouped_probe_acc(X_ab_all, wide["degradation"].to_numpy(), wide["scene_id"].to_numpy())
    rows.append({"representation": "all_AFLB_alpha_beta", "dim": 6, "accuracy": acc_ab_all,
                 "explained_variance_pct": np.nan})
    print(f"  All AFLB [alpha,beta] combined (6-dim): accuracy = {acc_ab_all*100:.1f}%", flush=True)

    out = pd.DataFrame(rows)
    out["random_baseline"] = RANDOM_BASELINE
    out["above_random_pp"] = out["accuracy"] * 100 - RANDOM_BASELINE
    out.to_csv(STATS_DIR / "compact_embedding.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'compact_embedding.csv'}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
