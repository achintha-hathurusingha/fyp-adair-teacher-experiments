"""TEST08-C: internal conditioning analysis for Model C, per seed, on the
20 validation scenes x 3 degradations = 60 crops.

Records, per sample: e_S, gamma, beta, bottleneck before/after conditioning,
and the relative bottleneck change ||F_cond-F||_2/||F||_2. Aggregates
gamma/beta mean/std/min/max overall and per-degradation.

Then runs the evaluation-only controls (learned vs shuffled vs random vs
zero embedding) to test whether the conditioning benefit (if any) comes
from the semantic teacher-derived state or merely from an extra affine
transform. These controls do NOT retrain anything -- same checkpoint,
override the conditioning embedding only.

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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TEST08C = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST08C.parent
sys.path.insert(0, str(TEST08C / "src"))
from models import MODELS  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST08C / "results" / "checkpoints"
OUT_DIR = TEST08C / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]


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

    cond_rows = []       # gamma/beta stats per sample
    change_rows = []     # relative bottleneck change per sample
    control_rows = []    # learned/shuffled/random/zero restoration comparison

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"model_C_seed{seed}.pt"
        model = MODELS["C"]().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        samples = []  # (clean_t, degraded_t, deg, scene_id)
        with torch.no_grad():
            for row in val_rows:
                clean_t = load_rgb(row["clean_path"], device)
                for deg in DEGS:
                    degraded_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    samples.append((clean_t, degraded_t, deg, row["scene_id"]))

            e_s_list, gamma_list, beta_list = [], [], []
            for clean_t, degraded_t, deg, scene_id in samples:
                out, e_s, gamma, beta, F, F_cond = model.forward_diagnostics(degraded_t)
                e_s_np = e_s[0].cpu().numpy()
                gamma_np = gamma[0].cpu().numpy()
                beta_np = beta[0].cpu().numpy()
                rel_change = float(torch.norm(F_cond - F) / (torch.norm(F) + 1e-12))
                psnr, ssim = psnr_ssim(out[0], clean_t[0])

                e_s_list.append(e_s_np)
                gamma_list.append(gamma_np)
                beta_list.append(beta_np)

                cond_rows.append({"seed": seed, "scene_id": scene_id, "degradation": deg,
                                   "gamma_mean": float(gamma_np.mean()), "gamma_std": float(gamma_np.std()),
                                   "gamma_min": float(gamma_np.min()), "gamma_max": float(gamma_np.max()),
                                   "beta_mean": float(beta_np.mean()), "beta_std": float(beta_np.std()),
                                   "beta_min": float(beta_np.min()), "beta_max": float(beta_np.max())})
                change_rows.append({"seed": seed, "scene_id": scene_id, "degradation": deg,
                                     "relative_bottleneck_change": rel_change,
                                     "psnr_learned": psnr, "ssim_learned": ssim})

            e_s_arr = np.stack(e_s_list)  # (60, 16)
            emp_mean = e_s_arr.mean(axis=0)
            emp_std = e_s_arr.std(axis=0)

            rng = np.random.RandomState(1000 + seed)
            shuffle_idx = rng.permutation(len(samples))

            for i, (clean_t, degraded_t, deg, scene_id) in enumerate(samples):
                learned_e = torch.from_numpy(e_s_arr[i]).float().unsqueeze(0).to(device)
                shuffled_e = torch.from_numpy(e_s_arr[shuffle_idx[i]]).float().unsqueeze(0).to(device)
                random_e = torch.from_numpy(
                    rng.normal(emp_mean, emp_std).astype(np.float32)).unsqueeze(0).to(device)
                zero_e = torch.zeros(1, e_s_arr.shape[1], device=device)

                for cond_name, e_override in [("learned", learned_e), ("shuffled", shuffled_e),
                                               ("random_matched", random_e), ("zero", zero_e)]:
                    out, _, _ = model.forward_with_override_embedding(degraded_t, e_override)
                    psnr, ssim = psnr_ssim(out[0], clean_t[0])
                    control_rows.append({"seed": seed, "scene_id": scene_id, "degradation": deg,
                                          "condition": cond_name, "psnr": psnr, "ssim": ssim})

        print(f"seed {seed}: conditioning analysis + controls done "
              f"({len(samples)} val crops x 4 conditions)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(OUT_DIR / "conditioning_statistics.csv", index=False)
    print(f"wrote {OUT_DIR / 'conditioning_statistics.csv'}")

    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(OUT_DIR / "bottleneck_change.csv", index=False)
    print(f"wrote {OUT_DIR / 'bottleneck_change.csv'}")

    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(OUT_DIR / "random_control.csv", index=False)
    print(f"wrote {OUT_DIR / 'random_control.csv'}")

    print("\n=== Gamma/Beta summary (overall) ===")
    print(cond_df[["gamma_mean", "gamma_std", "beta_mean", "beta_std"]].describe().to_string())
    print("\n=== Gamma/Beta summary by degradation ===")
    print(cond_df.groupby("degradation")[["gamma_mean", "gamma_std", "beta_mean", "beta_std"]].mean().to_string())
    print("\n=== Relative bottleneck change by degradation ===")
    print(change_df.groupby("degradation")["relative_bottleneck_change"].agg(["mean", "std"]).to_string())
    print("\n=== Learned vs random/shuffled/zero conditioning (mean PSNR/SSIM) ===")
    print(control_df.groupby("condition")[["psnr", "ssim"]].mean().to_string())


if __name__ == "__main__":
    main()
