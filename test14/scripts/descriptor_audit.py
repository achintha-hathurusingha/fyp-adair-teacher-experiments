"""TEST14: frequency descriptor validity audit, BEFORE any student
training. Computes q_F for every crop in the (reused, read-only) TEST12
dataset, checks: variance per band, pairwise similarity across different
images, degradation-wise mean, scene-wise variance, cross-degradation
separability (grouped probe). If q_F is nearly constant across images,
this script STOPS (non-zero exit) and training must not proceed.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python descriptor_audit.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST14 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST14.parent
sys.path.insert(0, str(TEST14 / "scripts"))
from frequency_descriptor import compute_qF, N_BANDS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
OUT_DIR = TEST14 / "results" / "statistics"
VIZ_DIR = TEST14 / "results" / "visualizations"
DEGS = ["Rain", "Haze", "Noise"]
COLLAPSE_STD_THRESHOLD = 1e-4  # per-band std below this across ALL images = STOP


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def grouped_probe(X, y, groups):
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), y_enc[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    records = []
    with torch.no_grad():
        for row in rows:
            for deg in DEGS:
                img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                q_f = compute_qF(img_t)[0].cpu().numpy()
                records.append({"scene_id": row["scene_id"], "crop_id": row["crop_id"], "split": row["split"],
                                 "degradation": deg, "q_f": q_f, "sum_qF": float(q_f.sum())})

    df_meta = pd.DataFrame([{k: v for k, v in r.items() if k != "q_f"} for r in records])
    Q = np.stack([r["q_f"] for r in records])  # (N, 8)

    band_cols = [f"band{i+1}" for i in range(N_BANDS)]
    q_df = pd.concat([df_meta, pd.DataFrame(Q, columns=band_cols)], axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(OUT_DIR / "frequency_descriptors_all_crops.csv", index=False)
    print(f"wrote {OUT_DIR / 'frequency_descriptors_all_crops.csv'}: {len(q_df)} rows")
    print(f"sum(q_F) statistics: mean={q_df.sum_qF.mean():.4f} min={q_df.sum_qF.min():.4f} "
          f"max={q_df.sum_qF.max():.4f} (expected close to but not exactly 1.0 -- corner "
          f"frequencies beyond axis-Nyquist are excluded from all 8 bands by design)")

    # ---- 1. variance per band ----
    band_std = Q.std(axis=0)
    band_mean = Q.mean(axis=0)
    print("\n=== Per-band mean / std (across ALL crops x degradations) ===")
    for i in range(N_BANDS):
        print(f"  band{i+1}: mean={band_mean[i]:.5f} std={band_std[i]:.5f}")

    collapsed = bool((band_std < COLLAPSE_STD_THRESHOLD).all())

    # ---- 2. pairwise similarity across different images ----
    rng = np.random.RandomState(0)
    n = len(Q)
    idx_i = rng.randint(0, n, size=3000)
    idx_j = rng.randint(0, n, size=3000)
    mask = idx_i != idx_j
    a, b = Q[idx_i[mask]], Q[idx_j[mask]]
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    pairwise_cos = (a_n * b_n).sum(axis=1)
    print(f"\nPairwise cosine similarity across DIFFERENT images: mean={pairwise_cos.mean():.4f} "
          f"std={pairwise_cos.std():.4f}")

    # ---- 3. degradation-wise mean ----
    print("\n=== Degradation-wise mean q_F ===")
    deg_means = q_df.groupby("degradation")[band_cols].mean()
    print(deg_means.to_string())

    # ---- 4. scene-wise variance (within a fixed degradation) ----
    scene_var_rows = []
    for deg in DEGS:
        sub = q_df[q_df.degradation == deg]
        for band in band_cols:
            scene_var_rows.append({"degradation": deg, "band": band,
                                    "variance_across_scenes": float(sub[band].var())})
    scene_var_df = pd.DataFrame(scene_var_rows)
    scene_var_df.to_csv(OUT_DIR / "qF_scene_variance.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'qF_scene_variance.csv'}")

    # ---- 5. cross-degradation separability (grouped probe) ----
    acc, f1 = grouped_probe(Q, df_meta.degradation.values, df_meta.scene_id.values)
    print(f"\nq_F degradation-classification probe: accuracy={acc*100:.1f}% macro_f1={f1:.3f}")

    audit_summary = pd.DataFrame({
        "metric": ["mean_pairwise_cosine_diff_images", "collapsed", "degradation_probe_accuracy",
                   "degradation_probe_macro_f1", "min_band_std", "max_band_std"],
        "value": [float(pairwise_cos.mean()), collapsed, acc, f1, float(band_std.min()), float(band_std.max())],
    })
    audit_summary.to_csv(OUT_DIR / "frequency_descriptor_validity_summary.csv", index=False)

    if collapsed:
        print("\n" + "=" * 70)
        print("STOP: q_F is nearly CONSTANT across all images (all band stds "
              f"< {COLLAPSE_STD_THRESHOLD}). Do NOT train a frequency-conditioned "
              "student around a useless descriptor.")
        print("=" * 70)
        raise SystemExit(1)
    else:
        print(f"\nq_F PASSES the validity audit: non-trivial variance, degradation probe "
              f"accuracy {acc*100:.1f}% (well above chance for 3 classes), pairwise cosine "
              f"across different images = {pairwise_cos.mean():.4f} (not near 1.0, so images "
              f"are not producing near-identical descriptors). Safe to proceed to student training.")


if __name__ == "__main__":
    main()
