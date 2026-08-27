"""TEST05.5 Phase 1: simple, non-learned statistical baselines on TEST03's
exact 300 images. Purpose: establish how much of the "AdaIR representation
is degradation-aware" story is already recoverable from hand-crafted pixel
statistics alone, with NO neural network involved -- the sharpest possible
test of loophole L1.

Computed per image: RGB mean/std (per channel), overall variance,
luminance mean/std, gradient-magnitude mean/std (Sobel), Laplacian
mean/std/energy, edge density (Canny-like threshold on gradient magnitude),
8-bin RGB histogram (per channel, 24 bins total), spatial autocorrelation
(lag-1, horizontal+vertical, luminance), FFT radial low/mid/high energy %
of the luminance channel.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python simple_statistics.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
MANIFEST_PATH = TEACHER_EXP / "test03" / "results" / "manifest" / "scene_manifest.csv"
OUT_DIR = TEST05_5 / "results" / "simple_stats"
DEGS = ["Rain", "Haze", "Noise"]
RANDOM_BASELINE = 100.0 / 3.0


def radial_band_energy(gray: np.ndarray, n_bins=3):
    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag2 = np.abs(fft) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / r.max()
    bins = np.linspace(0, 1, n_bins + 1)
    energies = []
    for i in range(n_bins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        energies.append(float(mag2[mask].sum()))
    total = sum(energies) + 1e-12
    return [e / total * 100 for e in energies]


def compute_features(img_u8: np.ndarray) -> dict:
    img = img_u8.astype(np.float32)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)

    feats = {}
    for c, name in enumerate(["R", "G", "B"]):
        feats[f"{name}_mean"] = float(img[:, :, c].mean())
        feats[f"{name}_std"] = float(img[:, :, c].std())
        feats[f"{name}_var"] = float(img[:, :, c].var())

    feats["luminance_mean"] = float(gray.mean())
    feats["luminance_std"] = float(gray.std())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    feats["gradient_mag_mean"] = float(grad_mag.mean())
    feats["gradient_mag_std"] = float(grad_mag.std())

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    feats["laplacian_mean"] = float(lap.mean())
    feats["laplacian_std"] = float(lap.std())
    feats["laplacian_energy"] = float((lap ** 2).mean())

    edges = cv2.Canny(img_u8, 100, 200)
    feats["edge_density"] = float((edges > 0).mean())

    for c, name in enumerate(["R", "G", "B"]):
        hist, _ = np.histogram(img_u8[:, :, c], bins=8, range=(0, 255))
        hist = hist / hist.sum()
        for i, v in enumerate(hist):
            feats[f"hist_{name}_{i}"] = float(v)

    gray_norm = (gray - gray.mean())
    denom = (gray_norm ** 2).sum() + 1e-12
    autocorr_h = float((gray_norm[:, :-1] * gray_norm[:, 1:]).sum() / denom)
    autocorr_v = float((gray_norm[:-1, :] * gray_norm[1:, :]).sum() / denom)
    feats["autocorr_horizontal_lag1"] = autocorr_h
    feats["autocorr_vertical_lag1"] = autocorr_v

    low, mid, high = radial_band_energy(gray)
    feats["fft_low_pct"] = low
    feats["fft_mid_pct"] = mid
    feats["fft_high_pct"] = high

    return feats


def grouped_probe(X, y, groups, clf_ctor):
    X = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    gkf = GroupKFold(n_splits=5)
    fold_acc, y_pred_all = [], np.empty_like(y_enc)
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        clf = clf_ctor()
        clf.fit(X[train_idx], y_enc[train_idx])
        pred = clf.predict(X[test_idx])
        y_pred_all[test_idx] = pred
        fold_acc.append(accuracy_score(y_enc[test_idx], pred))
    return {
        "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc)),
        "balanced_accuracy": balanced_accuracy_score(y_enc, y_pred_all),
        "macro_f1": f1_score(y_enc, y_pred_all, average="macro"),
        "confusion_matrix": confusion_matrix(y_enc, y_pred_all).tolist(),
    }


def main():
    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    from PIL import Image
    rows = []
    for scene_row in scene_rows:
        scene_id = scene_row["scene_id"]
        for deg in DEGS:
            img = np.array(Image.open(scene_row[f"{deg.lower()}_image_path"]).convert("RGB"))
            feats = compute_features(img)
            rows.append({"scene_id": scene_id, "degradation": deg, **feats})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "simple_statistics.csv", index=False)
    print(f"wrote {OUT_DIR / 'simple_statistics.csv'} ({len(df)} rows, {len(df.columns) - 2} features)", flush=True)

    feature_cols = [c for c in df.columns if c not in ("scene_id", "degradation")]
    X = df[feature_cols].to_numpy()
    y = df["degradation"].to_numpy()
    groups = df["scene_id"].to_numpy()

    results = []
    for clf_name, clf_ctor in [("logreg", lambda: LogisticRegression(max_iter=2000)),
                                ("linear_svm", lambda: LinearSVC(max_iter=5000))]:
        r = grouped_probe(X, y, groups, clf_ctor)
        results.append({"classifier": clf_name, "n_features": len(feature_cols),
                         "accuracy_mean": r["accuracy_mean"], "accuracy_std": r["accuracy_std"],
                         "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
                         "random_baseline": RANDOM_BASELINE, "confusion_matrix": r["confusion_matrix"]})
        print(f"{clf_name}: accuracy={r['accuracy_mean']*100:.1f}% "
              f"balanced={r['balanced_accuracy']*100:.1f}% macro_f1={r['macro_f1']:.3f}", flush=True)
        print(f"  confusion matrix (Haze/Noise/Rain order): {r['confusion_matrix']}", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "simple_statistics_probe.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'simple_statistics_probe.csv'}")

    # also test with just the FFT band features (closest analog to "frequency-only" baseline)
    fft_cols = ["fft_low_pct", "fft_mid_pct", "fft_high_pct"]
    X_fft = df[fft_cols].to_numpy()
    r_fft = grouped_probe(X_fft, y, groups, lambda: LogisticRegression(max_iter=2000))
    print(f"\nFFT-band-only (3 features): accuracy={r_fft['accuracy_mean']*100:.1f}%")
    pd.DataFrame([{"classifier": "logreg_fft_only", "n_features": 3,
                   "accuracy_mean": r_fft["accuracy_mean"], "accuracy_std": r_fft["accuracy_std"],
                   "balanced_accuracy": r_fft["balanced_accuracy"], "macro_f1": r_fft["macro_f1"],
                   "random_baseline": RANDOM_BASELINE,
                   "confusion_matrix": r_fft["confusion_matrix"]}]).to_csv(
        OUT_DIR / "simple_statistics_fft_only_probe.csv", index=False)


if __name__ == "__main__":
    main()
