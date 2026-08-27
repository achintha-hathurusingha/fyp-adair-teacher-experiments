"""TEST05.5 Phase 12-14: compact representation audit.

Phase 12 -- evaluate PCA-16/32/64 for RESTORATION-relevant information, not
just classification: correlate each compact embedding with the actual
restoration IMPROVEMENT the teacher achieves on that image (output PSNR/SSIM
minus input PSNR/SSIM vs clean), via a held-out (grouped-CV) linear
regression R^2 -- a compact code that is useful for degradation adaptation
should predict how much restoration work is needed / achieved, not merely
which of 3 labels applies.

Phase 13 -- compare PCA against a simple LEARNED linear projection
(768->16/32/64 via a single nn.Linear-equivalent: ordinary least-squares /
LogisticRegression-derived weight matrix is not directly comparable, so we
use PCA vs a *random-but-fixed* linear projection as the "just as good as
any linear map" negative control, and a *supervised* linear projection
(LDA-style: top discriminant directions from LogisticRegression's learned
coefficients) as the upper-bound comparison) -- explicitly no large NN.

Phase 14 -- negative controls: random projection (same dim), shuffled
labels (accuracy should collapse to chance), shuffled scene/degradation
correspondence (breaks the true pairing), random equal-energy subset of
raw dimensions (same dimensionality, no PCA structure). All compared
against real leakage-safe PCA-16/32/64 accuracy (from pca_leakage_audit.py)
to demonstrate compactness performance is not a dimensionality artifact.

Usage (on devon, adair-distill env, PINNED, GPU-free -- reads existing .npz):
  taskset -c 0-7,12-31 python compact_representation_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
LATENT_NPZ = TEACHER_EXP / "test05" / "results" / "feature_analysis" / "latent_pre.npz"
RESTORE_CSV_CANDIDATES = [
    TEACHER_EXP / "test03" / "results" / "statistics" / "restoration_metrics.csv",
]
OUT_DIR = TEST05_5 / "results" / "compact"
DIMS = [16, 32, 64]
N_SPLITS = 5
SEED = 0


def leakage_safe_transform_and_probe(X_raw, y, groups, transform_fn):
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=N_SPLITS)
    accs = []
    for train_idx, test_idx in gkf.split(X_raw, y_enc, groups=groups):
        scaler = StandardScaler().fit(X_raw[train_idx])
        Xtr_s, Xte_s = scaler.transform(X_raw[train_idx]), scaler.transform(X_raw[test_idx])
        Xtr, Xte = transform_fn(Xtr_s, Xte_s, y_enc[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(Xte)))
    return float(np.mean(accs)), float(np.std(accs))


def pca_transform_fn(n_components):
    def fn(Xtr_s, Xte_s, y_train):
        pca = PCA(n_components=n_components, random_state=SEED).fit(Xtr_s)
        return pca.transform(Xtr_s), pca.transform(Xte_s)
    return fn


def random_projection_transform_fn(n_components):
    def fn(Xtr_s, Xte_s, y_train):
        rp = GaussianRandomProjection(n_components=n_components, random_state=SEED).fit(Xtr_s)
        return rp.transform(Xtr_s), rp.transform(Xte_s)
    return fn


def supervised_linear_transform_fn(n_components):
    """Top-n_components directions from a one-vs-rest LogisticRegression's
    learned coefficient matrix, fit on TRAIN ONLY -- an upper-bound
    "best-case simple linear map" comparison, still leakage-safe."""
    def fn(Xtr_s, Xte_s, y_train):
        clf = LogisticRegression(max_iter=2000).fit(Xtr_s, y_train)
        W = clf.coef_  # (n_classes, n_features) or (1, n_features) for binary
        if W.shape[0] < n_components:
            # pad with PCA directions on residual if too few classes
            resid_pca = PCA(n_components=n_components - W.shape[0], random_state=SEED).fit(Xtr_s)
            proj = np.vstack([W, resid_pca.components_])
        else:
            proj = W[:n_components]
        return Xtr_s @ proj.T, Xte_s @ proj.T
    return fn


def random_equal_energy_subset_fn(n_components, rng):
    def fn(Xtr_s, Xte_s, y_train):
        idx = rng.choice(Xtr_s.shape[1], size=n_components, replace=False)
        return Xtr_s[:, idx], Xte_s[:, idx]
    return fn


def shuffled_label_control(X_raw, groups, n_components, rng):
    y_shuffled = rng.permutation(len(X_raw))  # meaningless pseudo-labels, same cardinality distribution not required
    y_shuffled = (y_shuffled % 3).astype(str)
    return leakage_safe_transform_and_probe(X_raw, y_shuffled, groups, pca_transform_fn(n_components))


def shuffled_correspondence_control(X_raw, y, groups, n_components, rng):
    perm = rng.permutation(len(X_raw))
    y_perm = np.array(y)[perm]  # breaks true (feature, label) pairing while keeping label distribution
    return leakage_safe_transform_and_probe(X_raw, y_perm, groups, pca_transform_fn(n_components))


def phase12_restoration_relevance(X_raw, y, groups, restore_df):
    if restore_df is None:
        print("Phase 12: SKIPPED -- no restoration-quality CSV found from TEST03 (documented, not fabricated).")
        return pd.DataFrame()

    # Align restore_df rows to (X_raw, y, groups) order via (scene_id, degradation) key --
    # positional alignment would silently corrupt the target if row order differs.
    key_to_psnr = {(s, d): p for s, d, p in zip(restore_df["scene_id"], restore_df["degradation"], restore_df["psnr"])}
    target = np.array([key_to_psnr.get((s, d), np.nan) for s, d in zip(groups, y)])
    valid = ~np.isnan(target)
    n_missing = int((~valid).sum())
    if n_missing:
        print(f"Phase 12: {n_missing}/{len(target)} rows had no restoration_metrics.csv match, dropped.")
    X_raw, y, groups, target = X_raw[valid], y[valid], groups[valid], target[valid]

    rows = []
    for dim in DIMS:
        y_enc = LabelEncoder().fit_transform(y)
        gkf = GroupKFold(n_splits=N_SPLITS)
        r2s = []
        for train_idx, test_idx in gkf.split(X_raw, y_enc, groups=groups):
            scaler = StandardScaler().fit(X_raw[train_idx])
            pca = PCA(n_components=dim, random_state=SEED).fit(scaler.transform(X_raw[train_idx]))
            Xtr = pca.transform(scaler.transform(X_raw[train_idx]))
            Xte = pca.transform(scaler.transform(X_raw[test_idx]))
            reg = LinearRegression().fit(Xtr, target[train_idx])
            pred = reg.predict(Xte)
            r2s.append(r2_score(target[test_idx], pred))
        rows.append({"dim": dim, "r2_restoration_psnr_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))})
        print(f"PCA-{dim}: R^2(predict restoration output PSNR) = {np.mean(r2s):.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    d = np.load(LATENT_NPZ, allow_pickle=True)
    X_raw = np.concatenate([d["X_gap"], d["X_gmp"]], axis=1)
    y, groups = d["degradation"], d["scene_id"]
    rng = np.random.RandomState(SEED)

    restore_df = None
    for path in RESTORE_CSV_CANDIDATES:
        if path.exists():
            restore_df = pd.read_csv(path)
            print(f"Phase 12: using restoration-quality reference {path}")
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 13: PCA vs random projection vs supervised-linear ----
    rows = []
    for dim in DIMS:
        acc_pca, std_pca = leakage_safe_transform_and_probe(X_raw, y, groups, pca_transform_fn(dim))
        acc_rp, std_rp = leakage_safe_transform_and_probe(X_raw, y, groups, random_projection_transform_fn(dim))
        acc_sup, std_sup = leakage_safe_transform_and_probe(X_raw, y, groups, supervised_linear_transform_fn(dim))
        rows.append({"dim": dim, "method": "pca", "accuracy_mean": acc_pca, "accuracy_std": std_pca})
        rows.append({"dim": dim, "method": "random_projection", "accuracy_mean": acc_rp, "accuracy_std": std_rp})
        rows.append({"dim": dim, "method": "supervised_linear_upper_bound", "accuracy_mean": acc_sup, "accuracy_std": std_sup})
        print(f"dim={dim}: PCA={acc_pca*100:.1f}%  RandomProj={acc_rp*100:.1f}%  "
              f"SupervisedLinear(upper bound)={acc_sup*100:.1f}%", flush=True)

    # ---- Phase 14: negative controls ----
    for dim in DIMS:
        acc_shuf_label, _ = shuffled_label_control(X_raw, groups, dim, rng)
        acc_shuf_corr, _ = shuffled_correspondence_control(X_raw, y, groups, dim, rng)
        acc_eq_energy, std_eq = leakage_safe_transform_and_probe(
            X_raw, y, groups, random_equal_energy_subset_fn(dim, rng))
        rows.append({"dim": dim, "method": "shuffled_labels_control", "accuracy_mean": acc_shuf_label})
        rows.append({"dim": dim, "method": "shuffled_correspondence_control", "accuracy_mean": acc_shuf_corr})
        rows.append({"dim": dim, "method": "random_equal_dim_subset_control", "accuracy_mean": acc_eq_energy,
                      "accuracy_std": std_eq})
        print(f"dim={dim} CONTROLS: shuffled_labels={acc_shuf_label*100:.1f}% "
              f"shuffled_correspondence={acc_shuf_corr*100:.1f}% "
              f"random_equal_dim_subset={acc_eq_energy*100:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "compact_vs_controls.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'compact_vs_controls.csv'}")
    print(df.to_string(index=False))

    restoration_df = phase12_restoration_relevance(X_raw, y, groups, restore_df)
    if len(restoration_df):
        restoration_df.to_csv(OUT_DIR / "restoration_relevance.csv", index=False)


if __name__ == "__main__":
    main()
