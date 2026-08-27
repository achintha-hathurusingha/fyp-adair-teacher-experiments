"""TEST11: modulation/coefficient analysis + effective-rank utilization,
for all F-variants (F2/F4/F8/F16), on the 20 validation scenes x 3
degradations = 60 crops.

For each (rank, seed): collect the coefficient vector a(e_S) in R^R and the
pre/post-conditioning bottleneck (F, F_cond) per sample. Computes:
  - relative modulation magnitude ||F_cond-F||/||F|| per sample, aggregated
    by degradation
  - coefficient L2 magnitude ||a|| per sample, aggregated by degradation
  - coefficient sparsity: number of "active" components (|a_i| > 10% of
    max|a| for that sample) and relative energy of the top-1 component
  - EFFECTIVE RANK of the coefficient matrix A_coeff (N x R) across the
    validation set, via participation ratio of its covariance eigenvalue
    spectrum -- tests whether configured rank R was actually used (a
    rank-16 model using only 2-3 effective dimensions would mean the extra
    configured capacity is not being exploited)

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python coefficient_and_rank_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST11 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST11.parent
sys.path.insert(0, str(TEST11 / "src"))
from models import MODELS, RANK_OF  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST11 / "results" / "checkpoints"
OUT_DIR = TEST11 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]
F_MODELS = ["F2", "F4", "F8", "F16"]


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def effective_rank(X: np.ndarray) -> float:
    if X.shape[1] == 1:
        return 1.0
    cov = np.cov(X, rowvar=False)
    eigvals = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(cov)))
    eigvals = np.clip(eigvals, 0, None)
    return float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    coeff_rows, mod_rows, rank_rows = [], [], []

    for model_name in F_MODELS:
        rank = RANK_OF[model_name]
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"model_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"SKIP {model_name}/seed{seed}: checkpoint not found", flush=True)
                continue
            model = MODELS[model_name]().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()

            a_by_deg = {d: [] for d in DEGS}
            a_all = []
            with torch.no_grad():
                for row in val_rows:
                    for deg in DEGS:
                        img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                        out, e_s, a, F_pre, F_cond = model.forward_diagnostics(img_t)
                        a_np = a[0].cpu().numpy()
                        rel_change = float(torch.norm(F_cond - F_pre) / (torch.norm(F_pre) + 1e-12))

                        abs_a = np.abs(a_np)
                        max_a = abs_a.max() + 1e-12
                        n_active = int((abs_a > 0.1 * max_a).sum())
                        energy = a_np ** 2
                        top1_energy_frac = float(energy.max() / (energy.sum() + 1e-12))

                        coeff_rows.append({"model": model_name, "rank": rank, "seed": seed,
                                            "scene_id": row["scene_id"], "degradation": deg,
                                            "coeff_l2_magnitude": float(np.linalg.norm(a_np)),
                                            "n_active_coeffs": n_active,
                                            "top1_energy_fraction": top1_energy_frac})
                        mod_rows.append({"model": model_name, "rank": rank, "seed": seed,
                                          "scene_id": row["scene_id"], "degradation": deg,
                                          "relative_modulation_magnitude": rel_change})
                        a_by_deg[deg].append(a_np)
                        a_all.append(a_np)

            eff_rank_overall = effective_rank(np.stack(a_all))
            rank_rows.append({"model": model_name, "configured_rank": rank, "seed": seed, "degradation": "ALL",
                               "effective_rank": eff_rank_overall})
            for deg in DEGS:
                eff_rank_deg = effective_rank(np.stack(a_by_deg[deg]))
                rank_rows.append({"model": model_name, "configured_rank": rank, "seed": seed, "degradation": deg,
                                   "effective_rank": eff_rank_deg})

            print(f"{model_name}(R={rank})/seed{seed}: effective_rank(ALL)={eff_rank_overall:.2f}/{rank}",
                  flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeff_df = pd.DataFrame(coeff_rows)
    coeff_df.to_csv(OUT_DIR / "coefficient_analysis.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'coefficient_analysis.csv'}")

    mod_df = pd.DataFrame(mod_rows)
    mod_df.to_csv(OUT_DIR / "modulation_magnitude.csv", index=False)
    print(f"wrote {OUT_DIR / 'modulation_magnitude.csv'}")

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_DIR / "effective_rank.csv", index=False)
    print(f"wrote {OUT_DIR / 'effective_rank.csv'}")

    print("\n=== Coefficient magnitude by rank/degradation (mean across seeds) ===")
    print(coeff_df.groupby(["model", "degradation"])["coeff_l2_magnitude"].mean().unstack().to_string())

    print("\n=== Modulation magnitude by rank/degradation (mean across seeds) ===")
    print(mod_df.groupby(["model", "degradation"])["relative_modulation_magnitude"].mean().unstack().to_string())

    print("\n=== Effective rank vs configured rank (ALL degradations, mean across seeds) ===")
    print(rank_df[rank_df.degradation == "ALL"].groupby("model")[["configured_rank", "effective_rank"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
