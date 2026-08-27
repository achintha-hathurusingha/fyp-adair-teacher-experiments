"""TEST10-R Phase 14: restoration-trajectory analysis (explanatory, NOT a
primary quality metric). For validation samples, compares degraded input,
baseline (A), F, G, and teacher outputs: residual to clean GT, change from
input, and change relative to teacher. Per degradation. Models are
pre-loaded once (not per-sample) for efficiency.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python residual_analysis.py
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
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import load_adair  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST10R / "results" / "checkpoints"
ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
OUT_DIR = TEST10R / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
MODEL_NAMES = ["A", "F", "G"]


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    teacher = load_adair(ADAIR_DIR, CKPT_PATH, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    students = {}
    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                continue
            m = MODELS[model_name]().to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.eval()
            students[(model_name, seed)] = m
    print(f"pre-loaded {len(students)} student checkpoints", flush=True)

    rows_out = []
    with torch.no_grad():
        for row in val_rows:
            gt = load_rgb(row["clean_path"], device)
            for deg in DEGS:
                degraded = load_rgb(row[f"{deg.lower()}_path"], device)
                teacher_out = teacher(degraded)
                teacher_resid_gt = float(torch.norm(teacher_out - gt))
                input_gt_dist = float(torch.norm(degraded - gt))

                for (model_name, seed), model in students.items():
                    out, _ = model(degraded)
                    change_from_input = float(torch.norm(out - degraded))
                    resid_to_gt = float(torch.norm(out - gt))
                    change_vs_teacher = float(torch.norm(out - teacher_out))

                    rows_out.append({
                        "model": model_name, "seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                        "input_to_gt_dist": input_gt_dist,
                        "output_to_input_change": change_from_input,
                        "output_to_gt_residual": resid_to_gt,
                        "teacher_output_to_gt_residual": teacher_resid_gt,
                        "output_change_vs_teacher_output": change_vs_teacher,
                        "residual_recovered_fraction": 1.0 - (resid_to_gt / (input_gt_dist + 1e-12)),
                        "residual_gap_to_teacher": resid_to_gt - teacher_resid_gt,
                    })

        print(f"processed {len(val_rows)} scenes x {len(DEGS)} degradations x {len(students)} student runs",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_DIR / "residual_analysis.csv", index=False)
    print(f"wrote {OUT_DIR / 'residual_analysis.csv'}: {len(df)} rows")

    summary = df.groupby(["model", "degradation"])[
        ["output_to_gt_residual", "teacher_output_to_gt_residual", "residual_gap_to_teacher",
         "residual_recovered_fraction", "output_to_input_change", "output_change_vs_teacher_output"]].mean()
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "residual_analysis_summary.csv")


if __name__ == "__main__":
    main()
