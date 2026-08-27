"""TEST07-Pilot: after training, extract each model's compact/bottleneck
representation for every pilot image, and probe degradation classification
(grouped CV by scene_id, matching the established TEST02-06 methodology).
Compares teacher PCA-16 against Models A-D.

Model A has no explicit projection head (by design -- no distillation
target) -- its "representation" for this probe is the raw GAP-pooled
256-dim bottleneck, a fair like-for-like comparison point (some feature
vector every model provides "for free"), not literally the same
dimensionality as B/C/D's 16-dim head, which is fine: each gets its own
classifier.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python probe_student_representations.py
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

TEST07 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST07 / "src"))
from models import MODELS  # noqa: E402

MANIFEST_PATH = TEST07 / "results" / "pilot_manifest.csv"
CACHE_DIR = TEST07 / "results" / "teacher_cache"
CKPT_DIR = TEST07 / "results" / "checkpoints"
OUT_DIR = TEST07 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]


def load_rgb_tensor(path, device):
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

    results = []

    # ---- teacher PCA-16 ----
    d = np.load(CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    acc, f1 = grouped_probe(d["E"], d["degradation"], d["scene_id"])
    results.append({"representation": "teacher_PCA16", "dim": d["E"].shape[1], "accuracy": acc, "macro_f1": f1})
    print(f"teacher_PCA16: acc={acc*100:.1f}% f1={f1:.3f}", flush=True)

    # ---- each trained student model ----
    for model_name, cls in MODELS.items():
        ckpt_path = CKPT_DIR / f"model_{model_name}.pt"
        if not ckpt_path.exists():
            print(f"SKIP {model_name}: checkpoint not found", flush=True)
            continue
        model = cls().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        feats, degs, scenes = [], [], []
        with torch.no_grad():
            for row in rows:
                for deg in DEGS:
                    img_t = load_rgb_tensor(row[f"{deg.lower()}_path"], device)
                    out, e_s = model(img_t)
                    if e_s is not None:
                        vec = e_s[0].cpu().numpy()
                    else:
                        # Model A: raw GAP-pooled bottleneck (see module docstring)
                        bottleneck, _, _ = model._encode_to_bottleneck(img_t)
                        vec = bottleneck.mean(dim=(2, 3))[0].cpu().numpy()
                    feats.append(vec)
                    degs.append(deg)
                    scenes.append(row["scene_id"])
        if not feats:
            print(f"SKIP {model_name}: no embedding available (see note below)", flush=True)
            continue
        X = np.stack(feats)
        acc, f1 = grouped_probe(X, np.array(degs), np.array(scenes))
        results.append({"representation": f"model_{model_name}", "dim": X.shape[1], "accuracy": acc, "macro_f1": f1})
        print(f"model_{model_name}: acc={acc*100:.1f}% f1={f1:.3f}", flush=True)

    df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "representation_probe.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'representation_probe.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
