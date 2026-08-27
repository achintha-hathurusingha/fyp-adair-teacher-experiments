"""TEST10 Phases 11-12: restoration-CHANGE analysis (explanatory, NOT a
primary quality metric) and teacher/student residual-vs-ground-truth
comparison, for Models A/F/G plus the teacher itself.

Phase 11: ||I_output - I_input||_2 and ||I_output - I_GT||_2, per model,
per degradation -- does trajectory distillation change how far the student
moves from the degraded input, and how close it lands to clean?

Phase 12: teacher residual ||I_T - I_GT|| vs. student residual
||I_S - I_GT||, per degradation -- is G's residual closer to the teacher's
than F's is, especially for Haze?

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python residual_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST10 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10.parent
sys.path.insert(0, str(TEST10 / "src"))
from models import MODELS  # noqa: E402
from teacher_trajectory import load_frozen_teacher  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST10 / "results" / "checkpoints"
OUT_DIR = TEST10 / "results" / "statistics"
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

    teacher_model, teacher_net, _ = load_frozen_teacher(device)

    rows_out = []
    with torch.no_grad():
        for row in val_rows:
            gt = load_rgb(row["clean_path"], device)
            for deg in DEGS:
                degraded = load_rgb(row[f"{deg.lower()}_path"], device)

                teacher_out = teacher_model(degraded)
                teacher_resid_gt = float(torch.norm(teacher_out - gt))
                input_gt_dist = float(torch.norm(degraded - gt))

                for model_name in MODEL_NAMES:
                    for seed in SEEDS:
                        ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
                        if not ckpt_path.exists():
                            continue
                        model = MODELS[model_name]().to(device)
                        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                        model.eval()
                        out, _ = model(degraded)

                        change_from_input = float(torch.norm(out - degraded))
                        resid_to_gt = float(torch.norm(out - gt))

                        rows_out.append({
                            "model": model_name, "seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                            "input_to_gt_dist": input_gt_dist,
                            "output_to_input_change": change_from_input,
                            "output_to_gt_residual": resid_to_gt,
                            "teacher_output_to_gt_residual": teacher_resid_gt,
                            "residual_recovered_fraction": 1.0 - (resid_to_gt / (input_gt_dist + 1e-12)),
                            "residual_gap_to_teacher": resid_to_gt - teacher_resid_gt,
                        })

        print(f"processed {len(val_rows)} scenes x {len(DEGS)} degradations "
              f"x {len(MODEL_NAMES)} models x {len(SEEDS)} seeds", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_DIR / "residual_analysis.csv", index=False)
    print(f"wrote {OUT_DIR / 'residual_analysis.csv'}: {len(df)} rows")

    print("\n=== Mean residual-to-GT and gap-to-teacher, by model/degradation ===")
    summary = df.groupby(["model", "degradation"])[
        ["output_to_gt_residual", "teacher_output_to_gt_residual", "residual_gap_to_teacher",
         "residual_recovered_fraction", "output_to_input_change"]].mean()
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "residual_analysis_summary.csv")


if __name__ == "__main__":
    main()
