"""TEST09: per-stage, per-degradation conditioning analysis for Models
C/D/E/F, on the 20 validation scenes x 3 degradations = 60 crops.

For each conditioned stage (bottleneck, decoder_level3, decoder_level2 --
whichever exist for a given model) records gamma/beta stats (FiLM stages)
or 'a' stats (F's low-rank bottleneck stage), plus the relative modulation
magnitude ||F_cond-F_pre||_2/||F_pre||_2. This directly tests: does Haze
receive stronger/later conditioning than Rain/Noise?

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python conditioning_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST09 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST09.parent
sys.path.insert(0, str(TEST09 / "src"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST09 / "results" / "checkpoints"
OUT_DIR = TEST09 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
CONDITIONED_MODELS = ["C", "D", "E", "F"]


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    stage_rows = []   # gamma/beta or 'a' stats per sample per stage
    change_rows = []  # relative modulation magnitude per sample per stage

    for model_name in CONDITIONED_MODELS:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"SKIP {model_name}/seed{seed}: checkpoint not found", flush=True)
                continue
            model = MODELS[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            with torch.no_grad():
                for row in val_rows:
                    for deg in DEGS:
                        img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                        _, e_s, stage_diag = model.forward_diagnostics(img_t)

                        for stage_name, info in stage_diag.items():
                            F_pre, F_cond = info["F_pre"], info["F_cond"]
                            rel_change = float(torch.norm(F_cond - F_pre) / (torch.norm(F_pre) + 1e-12))
                            change_rows.append({"model": model_name, "seed": seed, "scene_id": row["scene_id"],
                                                 "degradation": deg, "stage": stage_name,
                                                 "relative_modulation_magnitude": rel_change})

                            if "gamma" in info:
                                gamma_np = info["gamma"][0].cpu().numpy()
                                beta_np = info["beta"][0].cpu().numpy()
                                stage_rows.append({"model": model_name, "seed": seed, "scene_id": row["scene_id"],
                                                    "degradation": deg, "stage": stage_name, "op_type": "FiLM",
                                                    "gamma_mean": float(gamma_np.mean()), "gamma_std": float(gamma_np.std()),
                                                    "beta_mean": float(beta_np.mean()), "beta_std": float(beta_np.std()),
                                                    "a_mean": None, "a_std": None})
                            elif "a" in info:
                                a_np = info["a"][0].cpu().numpy()
                                stage_rows.append({"model": model_name, "seed": seed, "scene_id": row["scene_id"],
                                                    "degradation": deg, "stage": stage_name, "op_type": "lowrank",
                                                    "gamma_mean": None, "gamma_std": None, "beta_mean": None,
                                                    "beta_std": None, "a_mean": float(a_np.mean()),
                                                    "a_std": float(a_np.std())})

            print(f"{model_name}/seed{seed}: conditioning analysis done "
                  f"({len(val_rows)} scenes x 3 degradations)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(OUT_DIR / "conditioning_statistics.csv", index=False)
    print(f"wrote {OUT_DIR / 'conditioning_statistics.csv'}")

    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(OUT_DIR / "modulation_magnitude.csv", index=False)
    print(f"wrote {OUT_DIR / 'modulation_magnitude.csv'}")

    print("\n=== Relative modulation magnitude by model/stage/degradation ===")
    print(change_df.groupby(["model", "stage", "degradation"])["relative_modulation_magnitude"]
          .agg(["mean", "std"]).to_string())

    print("\n=== Does Haze get stronger conditioning than Rain/Noise, per model/stage? ===")
    piv = change_df.groupby(["model", "stage", "degradation"])["relative_modulation_magnitude"].mean().unstack()
    print(piv.to_string())


if __name__ == "__main__":
    main()
