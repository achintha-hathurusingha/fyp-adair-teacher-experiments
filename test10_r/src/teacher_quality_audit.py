"""TEST10-R Phase 0: teacher-vs-baseline quality audit, BEFORE any new
student training. Evaluates the frozen AdaIR teacher and the (already
validated, deterministic) baseline Model A on the exact validation set,
per degradation, and classifies each degradation as teacher_better /
teacher_similar / teacher_worse -- WITHOUT reference to any student.

Model A's checkpoints are reused READ-ONLY from test10/ (A's training is
fully deterministic given identical seed/data/architecture -- confirmed
bit-identical across TEST08-C/09/10 -- so no retraining is needed to answer
this question; TEST10-R will still train its OWN fresh A/F/G in Phase 4-10
for self-containment, which will reproduce these same numbers as a bonus
consistency check).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python teacher_quality_audit.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
sys.path.insert(0, str(TEST10R / "src"))
from models import MODELS  # noqa: E402
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import load_adair  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
TEST10_CKPT_DIR = TEACHER_EXP / "test10" / "results" / "checkpoints"  # READ-ONLY reuse
ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
OUT_PATH = TEST10R / "results" / "teacher_quality.csv"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
SIMILAR_THRESHOLD_DB = 0.5  # within +-0.5dB counted as "similar"


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def psnr_ssim(pred, target):
    pred_u8 = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    tgt_u8 = (target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    psnr = float(peak_signal_noise_ratio(tgt_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(tgt_u8, pred_u8, data_range=255, channel_axis=2))
    return psnr, ssim


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    teacher = load_adair(ADAIR_DIR, CKPT_PATH, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    rows_out = []
    with torch.no_grad():
        for row in val_rows:
            gt = load_rgb(row["clean_path"], device)
            for deg in DEGS:
                degraded = load_rgb(row[f"{deg.lower()}_path"], device)
                teacher_out = teacher(degraded)
                t_psnr, t_ssim = psnr_ssim(teacher_out[0], gt[0])
                rows_out.append({"source": "teacher", "seed": None, "scene_id": row["scene_id"],
                                  "degradation": deg, "psnr": t_psnr, "ssim": t_ssim})

                for seed in SEEDS:
                    ckpt_path = TEST10_CKPT_DIR / f"model_A_seed{seed}.pt"
                    if not ckpt_path.exists():
                        continue
                    model = MODELS["A"]().to(device)
                    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                    model.eval()
                    out, _ = model(degraded)
                    a_psnr, a_ssim = psnr_ssim(out[0], gt[0])
                    rows_out.append({"source": "baseline_A", "seed": seed, "scene_id": row["scene_id"],
                                      "degradation": deg, "psnr": a_psnr, "ssim": a_ssim})

    df = pd.DataFrame(rows_out)
    summary = df.groupby(["source", "degradation"])[["psnr", "ssim"]].mean().reset_index()
    teacher_summary = summary[summary.source == "teacher"].set_index("degradation")
    baseline_summary = summary[summary.source == "baseline_A"].set_index("degradation")

    final_rows = []
    for deg in DEGS:
        t_psnr, t_ssim = teacher_summary.loc[deg, "psnr"], teacher_summary.loc[deg, "ssim"]
        b_psnr, b_ssim = baseline_summary.loc[deg, "psnr"], baseline_summary.loc[deg, "ssim"]
        delta = t_psnr - b_psnr
        if abs(delta) <= SIMILAR_THRESHOLD_DB:
            classification = "teacher_similar"
        elif delta > 0:
            classification = "teacher_better"
        else:
            classification = "teacher_worse"
        final_rows.append({"degradation": deg, "teacher_psnr": t_psnr, "teacher_ssim": t_ssim,
                            "baseline_psnr": b_psnr, "baseline_ssim": b_ssim,
                            "delta_psnr_teacher_minus_baseline": delta,
                            "delta_ssim_teacher_minus_baseline": t_ssim - b_ssim,
                            "classification": classification})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
