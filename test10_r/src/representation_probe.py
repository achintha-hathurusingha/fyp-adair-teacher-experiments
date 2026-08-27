"""TEST10-R Phase 12: grouped degradation-classification probe for the
teacher's fixed stage targets, and for F/G's student bottleneck + e_S
(final 16-dim compact embedding), aggregated across seeds.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python representation_probe.py
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
sys.path.insert(0, str(TEST10R / "src"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
KD_CACHE_DIR = TEST07B_RESULTS / "teacher_cache"
CKPT_DIR = TEST10R / "results" / "checkpoints"
OUT_DIR = TEST10R / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
MODEL_NAMES = ["A", "F", "G"]
KD_MODELS = ["F", "G"]


def load_rgb_tensor(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def grouped_probe(X, y, groups):
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, bal_accs, f1s = [], [], []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), y_enc[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append(accuracy_score(y_enc[test_idx], pred))
        bal_accs.append(balanced_accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(bal_accs)), float(np.mean(f1s))


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    probe_rows = []
    d = np.load(KD_CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    acc, bal_acc, f1 = grouped_probe(d["E"], d["degradation"], d["scene_id"])
    probe_rows.append({"representation": "teacher_PCA16", "model": "teacher", "seed": None, "dim": d["E"].shape[1],
                        "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1})
    print(f"teacher_PCA16: acc={acc*100:.1f}%", flush=True)

    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"SKIP {model_name}/seed{seed}: checkpoint not found", flush=True)
                continue
            model = MODELS[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            bottleneck_feats, es_feats, degs, scenes = [], [], [], []
            with torch.no_grad():
                for row in rows:
                    for deg in DEGS:
                        img_t = load_rgb_tensor(row[f"{deg.lower()}_path"], device)
                        bpool = model.bottleneck_pooled(img_t)[0].cpu().numpy()
                        bottleneck_feats.append(bpool)
                        degs.append(deg)
                        scenes.append(row["scene_id"])
                        if model_name in KD_MODELS:
                            _, e_s = model(img_t)
                            es_feats.append(e_s[0].cpu().numpy())

            X_bottleneck = np.stack(bottleneck_feats)
            acc, bal_acc, f1 = grouped_probe(X_bottleneck, np.array(degs), np.array(scenes))
            probe_rows.append({"representation": f"model_{model_name}_bottleneck", "model": model_name, "seed": seed,
                                "dim": X_bottleneck.shape[1], "accuracy": acc, "balanced_accuracy": bal_acc,
                                "macro_f1": f1})
            print(f"model_{model_name}_bottleneck seed{seed}: acc={acc*100:.1f}%", flush=True)

            if model_name in KD_MODELS:
                X_es = np.stack(es_feats)
                acc, bal_acc, f1 = grouped_probe(X_es, np.array(degs), np.array(scenes))
                probe_rows.append({"representation": f"model_{model_name}_eS", "model": model_name, "seed": seed,
                                    "dim": X_es.shape[1], "accuracy": acc, "balanced_accuracy": bal_acc,
                                    "macro_f1": f1})
                print(f"model_{model_name}_eS seed{seed}: acc={acc*100:.1f}%", flush=True)

    probe_df = pd.DataFrame(probe_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(OUT_DIR / "representation_probe.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'representation_probe.csv'}")

    agg = probe_df.groupby("representation")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    agg.to_csv(OUT_DIR / "representation_probe_aggregated.csv")
    print(agg.to_string())


if __name__ == "__main__":
    main()
