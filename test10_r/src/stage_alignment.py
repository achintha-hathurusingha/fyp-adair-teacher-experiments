"""TEST10-R Phase 13: stage-wise teacher/student trajectory alignment for
Model G, against the FIXED precomputed teacher targets. Mandatory
cross-input diversity check per the task spec: same-sample cosine alone is
not sufficient -- must also confirm cross-sample diversity remains high
(this is the check that caught TEST10's collapse).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python stage_alignment.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
sys.path.insert(0, str(TEST10R / "src"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
TRAJ_CACHE_DIR = TEST10R / "results" / "teacher_cache"
CKPT_DIR = TEST10R / "results" / "checkpoints"
OUT_DIR = TEST10R / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
N_PAIRS = 2000


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pairwise_stats(X, rng, n_pairs=N_PAIRS):
    n = len(X)
    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    a, b = X[idx_i], X[idx_j]
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = (a_n * b_n).sum(axis=1)
    dist = np.linalg.norm(a - b, axis=1)
    return cos, dist


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    d = np.load(TRAJ_CACHE_DIR / "trajectory_targets.npz", allow_pickle=True)
    val_mask = d["split"] == "val"
    teacher_lookup = {(cid, deg): i for i, (cid, deg) in enumerate(zip(d["crop_id"], d["degradation"]))}

    stage_rows, diversity_rows = [], []
    rng = np.random.RandomState(0)

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"model_G_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"SKIP G/seed{seed}: checkpoint not found (likely aborted)", flush=True)
            continue
        model = MODELS["G"]().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        stage_es = {0: [], 1: [], 2: []}
        stage_et = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for row in val_rows:
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    out, e_s, e_s_traj = model.forward_trajectory(img_t)
                    idx = teacher_lookup[(row["crop_id"], deg)]
                    for stage_idx in (0, 1, 2):
                        stage_es[stage_idx].append(e_s_traj[stage_idx][0].cpu().numpy())
                        stage_et[stage_idx].append(d[f"E_stage{stage_idx}"][idx])

        for stage_idx in (0, 1, 2):
            es = np.stack(stage_es[stage_idx])
            et = np.stack(stage_et[stage_idx])
            es_n = es / (np.linalg.norm(es, axis=1, keepdims=True) + 1e-12)
            et_n = et / (np.linalg.norm(et, axis=1, keepdims=True) + 1e-12)
            same_sample_cos = (es_n * et_n).sum(axis=1)
            norm_mse = ((es_n - et_n) ** 2).mean()

            cos_es, dist_es = pairwise_stats(es_n, rng)
            cos_et, dist_et = pairwise_stats(et_n, rng)

            stage_rows.append({
                "seed": seed, "stage": stage_idx,
                "same_sample_cosine_mean": float(same_sample_cos.mean()),
                "same_sample_cosine_std": float(same_sample_cos.std()),
                "normalized_mse": float(norm_mse),
            })
            diversity_rows.append({
                "seed": seed, "stage": stage_idx,
                "student_cross_input_mean_cosine": float(cos_es.mean()),
                "student_cross_input_p95_cosine": float(np.percentile(cos_es, 95)),
                "student_cross_input_mean_distance": float(dist_es.mean()),
                "teacher_cross_input_mean_cosine": float(cos_et.mean()),
                "teacher_cross_input_p95_cosine": float(np.percentile(cos_et, 95)),
                "teacher_cross_input_mean_distance": float(dist_et.mean()),
                "student_collapsed": bool(cos_es.mean() > 0.98),
                "teacher_collapsed": bool(cos_et.mean() > 0.98),
                "valid_trajectory_model": bool(same_sample_cos.mean() > 0.3 and cos_es.mean() < 0.98),
            })
            print(f"seed{seed} stage{stage_idx}: same_sample_cos={same_sample_cos.mean():.4f} "
                  f"student_cross_input_cos={cos_es.mean():.4f} teacher_cross_input_cos={cos_et.mean():.4f}",
                  flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(OUT_DIR / "stage_alignment.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'stage_alignment.csv'}")

    diversity_df = pd.DataFrame(diversity_rows)
    diversity_df.to_csv(OUT_DIR / "cross_input_diversity.csv", index=False)
    print(f"wrote {OUT_DIR / 'cross_input_diversity.csv'}")
    print("\n=== VALIDITY SUMMARY (same-sample alignment high AND cross-sample diversity high) ===")
    print(diversity_df[["seed", "stage", "valid_trajectory_model", "student_collapsed",
                         "teacher_collapsed"]].to_string(index=False))


if __name__ == "__main__":
    main()
