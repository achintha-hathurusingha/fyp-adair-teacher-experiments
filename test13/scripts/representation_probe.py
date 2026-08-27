"""TEST13: grouped degradation-classification probe for teacher PCA-16 and
each model's e_D (F2, T12), aggregated across seeds.

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

TEST13 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST13.parent
sys.path.insert(0, str(TEST13 / "scripts"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CACHE_DIR = TEST07B_RESULTS / "teacher_cache"
CKPT_DIR = TEST13 / "results" / "checkpoints"
OUT_DIR = TEST13 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
KD_MODELS = ["F2", "T13"]


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

    probe_rows, align_rows = [], []
    d = np.load(CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    e_t_lookup = {(crop_id, deg): d["E"][i] for i, (crop_id, deg) in
                  enumerate(zip(d["crop_id"], d["degradation"]))}
    acc, bal_acc, f1 = grouped_probe(d["E"], d["degradation"], d["scene_id"])
    probe_rows.append({"representation": "teacher_PCA16", "model": "teacher", "seed": None,
                        "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1})
    print(f"teacher_PCA16: acc={acc*100:.1f}%", flush=True)

    for model_name in KD_MODELS:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                continue
            model = MODELS[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            es_feats, degs, scenes = [], [], []
            cos_sims, mse_vals = [], []
            with torch.no_grad():
                for row in rows:
                    for deg in DEGS:
                        img_t = load_rgb_tensor(row[f"{deg.lower()}_path"], device)
                        _, e_d = model(img_t)
                        e_d_np = e_d[0].cpu().numpy()
                        es_feats.append(e_d_np)
                        degs.append(deg)
                        scenes.append(row["scene_id"])
                        e_t = e_t_lookup.get((row["crop_id"], deg))
                        if e_t is not None:
                            cos = float(np.dot(e_d_np, e_t) / (np.linalg.norm(e_d_np) * np.linalg.norm(e_t) + 1e-12))
                            mse = float(np.mean((e_d_np - e_t) ** 2))
                            cos_sims.append(cos)
                            mse_vals.append(mse)

            X_es = np.stack(es_feats)
            acc, bal_acc, f1 = grouped_probe(X_es, np.array(degs), np.array(scenes))
            probe_rows.append({"representation": f"model_{model_name}_eD", "model": model_name, "seed": seed,
                                "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1})
            align_rows.append({"model": model_name, "seed": seed,
                                "mean_cosine_similarity": float(np.mean(cos_sims)),
                                "mean_mse": float(np.mean(mse_vals))})
            print(f"{model_name}/seed{seed}: probe_acc={acc*100:.1f}% cosine={np.mean(cos_sims):.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(OUT_DIR / "representation_probe.csv", index=False)
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(OUT_DIR / "teacher_alignment.csv", index=False)
    print(f"\nwrote representation_probe.csv and teacher_alignment.csv")

    print("\n=== Probe accuracy by model (mean across seeds) ===")
    print(probe_df[probe_df.model != "teacher"].groupby("model")["accuracy"].mean().to_string())


if __name__ == "__main__":
    main()
