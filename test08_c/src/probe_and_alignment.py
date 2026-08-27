"""TEST08-C: after all 9 training runs, (1) grouped degradation-classification
probe for teacher PCA-16, Model A bottleneck, Model B e_S, Model C e_S
(per seed, then aggregated mean+-std across seeds); (2) teacher-student
embedding alignment (cosine similarity, MSE) for Models B and C, per seed.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python probe_and_alignment.py
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

TEST08C = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST08C.parent
sys.path.insert(0, str(TEST08C / "src"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CACHE_DIR = TEST07B_RESULTS / "teacher_cache"
CKPT_DIR = TEST08C / "results" / "checkpoints"
OUT_DIR = TEST08C / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]


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

    d = np.load(CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    e_t_lookup = {(crop_id, deg): d["E"][i] for i, (crop_id, deg) in
                  enumerate(zip(d["crop_id"], d["degradation"]))}
    acc, bal_acc, f1 = grouped_probe(d["E"], d["degradation"], d["scene_id"])
    probe_rows.append({"representation": "teacher_PCA16", "model": "teacher", "seed": None, "dim": d["E"].shape[1],
                        "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1})
    print(f"teacher_PCA16: acc={acc*100:.1f}% bal_acc={bal_acc*100:.1f}% f1={f1:.3f}", flush=True)

    align_rows = []
    for model_name in ["A", "B", "C"]:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"SKIP {model_name}/seed{seed}: checkpoint not found", flush=True)
                continue
            model = MODELS[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            bottleneck_feats, es_feats, degs, scenes = [], [], [], []
            cos_sims, mse_vals, per_comp_sq_errs = [], [], []

            with torch.no_grad():
                for row in rows:
                    for deg in DEGS:
                        img_t = load_rgb_tensor(row[f"{deg.lower()}_path"], device)
                        bpool = model.bottleneck_pooled(img_t)[0].cpu().numpy()
                        bottleneck_feats.append(bpool)
                        degs.append(deg)
                        scenes.append(row["scene_id"])

                        if model_name in ("B", "C"):
                            _, e_s = model(img_t)
                            e_s_np = e_s[0].cpu().numpy()
                            es_feats.append(e_s_np)
                            e_t = e_t_lookup.get((row["crop_id"], deg))
                            if e_t is not None:
                                cos = float(np.dot(e_s_np, e_t) / (np.linalg.norm(e_s_np) * np.linalg.norm(e_t) + 1e-12))
                                mse = float(np.mean((e_s_np - e_t) ** 2))
                                cos_sims.append(cos)
                                mse_vals.append(mse)
                                per_comp_sq_errs.append((e_s_np - e_t) ** 2)

            X_bottleneck = np.stack(bottleneck_feats)
            acc, bal_acc, f1 = grouped_probe(X_bottleneck, np.array(degs), np.array(scenes))
            probe_rows.append({"representation": f"model_{model_name}_bottleneck", "model": model_name, "seed": seed,
                                "dim": X_bottleneck.shape[1], "accuracy": acc, "balanced_accuracy": bal_acc,
                                "macro_f1": f1})
            print(f"model_{model_name}_bottleneck seed{seed}: acc={acc*100:.1f}% f1={f1:.3f}", flush=True)

            if model_name in ("B", "C"):
                X_es = np.stack(es_feats)
                acc, bal_acc, f1 = grouped_probe(X_es, np.array(degs), np.array(scenes))
                probe_rows.append({"representation": f"model_{model_name}_eS", "model": model_name, "seed": seed,
                                    "dim": X_es.shape[1], "accuracy": acc, "balanced_accuracy": bal_acc,
                                    "macro_f1": f1})
                print(f"model_{model_name}_eS seed{seed}: acc={acc*100:.1f}% f1={f1:.3f}", flush=True)

                per_comp = np.stack(per_comp_sq_errs).mean(axis=0)
                align_rows.append({"model": model_name, "seed": seed,
                                    "mean_cosine_similarity": float(np.mean(cos_sims)),
                                    "mean_mse": float(np.mean(mse_vals)),
                                    "per_component_mse": per_comp.tolist(),
                                    "n_examples": len(cos_sims)})
                print(f"{model_name}/seed{seed} alignment: cosine={np.mean(cos_sims):.4f} mse={np.mean(mse_vals):.4f}",
                      flush=True)

    probe_df = pd.DataFrame(probe_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(OUT_DIR / "representation_probe.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'representation_probe.csv'}")

    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(OUT_DIR / "teacher_student_alignment.csv", index=False)
    print(f"wrote {OUT_DIR / 'teacher_student_alignment.csv'}")

    agg = probe_df.groupby("representation")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    agg.to_csv(OUT_DIR / "representation_probe_aggregated.csv")
    print("\nAggregated across seeds:")
    print(agg.to_string())


if __name__ == "__main__":
    main()
