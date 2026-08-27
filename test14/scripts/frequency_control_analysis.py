"""TEST14: the 4 mandatory frequency causal controls (normal, zero,
mean, shuffled) + coefficient analysis (a_T14 vs a_F2) + degradation-wise
frequency contribution + frequency signature analysis (between-degradation
vs within-scene-cross-degradation distance).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python frequency_control_analysis.py
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

TEST14 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST14.parent
sys.path.insert(0, str(TEST14 / "scripts"))
from models import MODELS  # noqa: E402
from frequency_descriptor import compute_qF  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST14 / "results" / "checkpoints"
OUT_DIR = TEST14 / "results" / "statistics"
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
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    control_rows, coeff_rows, sig_rows = [], [], []

    for seed in SEEDS:
        t14_ckpt = CKPT_DIR / f"model_T14_seed{seed}.pt"
        f2_ckpt = CKPT_DIR / f"model_F2_seed{seed}.pt"
        t14 = MODELS["T14"]().to(device)
        t14.load_state_dict(torch.load(t14_ckpt, map_location=device, weights_only=True))
        t14.eval()
        f2 = MODELS["F2"]().to(device)
        f2.load_state_dict(torch.load(f2_ckpt, map_location=device, weights_only=True))
        f2.eval()

        # ---- mean_qF from TRAINING crops only (leakage-safe) ----
        qf_accum = []
        with torch.no_grad():
            for row in train_rows[:200]:
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    qf_accum.append(compute_qF(img_t)[0].cpu().numpy())
        mean_qF = torch.from_numpy(np.mean(qf_accum, axis=0)).float().unsqueeze(0).to(device)

        samples = []
        with torch.no_grad():
            for row in val_rows:
                gt = load_rgb(row["clean_path"], device)
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    out_t14, e_d_t14, a_t14, phi, q_f = t14.forward_diagnostics(img_t)
                    out_f2, e_d_f2, a_f2 = f2.forward_diagnostics(img_t)

                    psnr_t14, ssim_t14 = psnr_ssim(out_t14[0], gt[0])
                    psnr_f2, ssim_f2 = psnr_ssim(out_f2[0], gt[0])

                    coeff_rows.append({"seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                                        "a0_T14": float(a_t14[0, 0]), "a1_T14": float(a_t14[0, 1]),
                                        "a0_F2": float(a_f2[0, 0]), "a1_F2": float(a_f2[0, 1]),
                                        "delta_a0": float(a_t14[0, 0] - a_f2[0, 0]),
                                        "delta_a1": float(a_t14[0, 1] - a_f2[0, 1]),
                                        "coeff_magnitude_T14": float(torch.norm(a_t14[0])),
                                        "coeff_magnitude_F2": float(torch.norm(a_f2[0])),
                                        "psnr_T14": psnr_t14, "psnr_F2": psnr_f2})

                    samples.append({"scene_id": row["scene_id"], "degradation": deg, "img_t": img_t, "gt": gt,
                                     "q_f": q_f})

        with torch.no_grad():
            for i, s in enumerate(samples):
                zero_qf = torch.zeros(1, 8, device=device)

                out_n, _ = t14.forward_with_override(s["img_t"])
                psnr_n, ssim_n = psnr_ssim(out_n[0], s["gt"][0])

                out_z, _ = t14.forward_with_override(s["img_t"], q_f_override=zero_qf)
                psnr_z, ssim_z = psnr_ssim(out_z[0], s["gt"][0])

                out_m, _ = t14.forward_with_override(s["img_t"], q_f_override=mean_qF)
                psnr_m, ssim_m = psnr_ssim(out_m[0], s["gt"][0])

                j = np.random.RandomState(4000 + i).randint(0, len(samples) - 1)
                if j >= i:
                    j += 1
                out_s, _ = t14.forward_with_override(s["img_t"], q_f_override=samples[j]["q_f"])
                psnr_s, ssim_s = psnr_ssim(out_s[0], s["gt"][0])

                control_rows.append({
                    "seed": seed, "scene_id": s["scene_id"], "degradation": s["degradation"],
                    "psnr_normal": psnr_n, "ssim_normal": ssim_n,
                    "psnr_zero_freq": psnr_z, "ssim_zero_freq": ssim_z,
                    "psnr_mean_freq": psnr_m, "ssim_mean_freq": ssim_m,
                    "psnr_shuffled_freq": psnr_s, "ssim_shuffled_freq": ssim_s,
                    "donor_scene_id": samples[j]["scene_id"], "donor_degradation": samples[j]["degradation"],
                })

        print(f"seed{seed}: controls + coefficient analysis done ({len(samples)} val crops)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(OUT_DIR / "frequency_controls.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'frequency_controls.csv'}")

    coeff_df = pd.DataFrame(coeff_rows)
    coeff_df.to_csv(OUT_DIR / "coefficient_analysis.csv", index=False)
    print(f"wrote {OUT_DIR / 'coefficient_analysis.csv'}")

    print("\n=== Frequency control comparison (mean PSNR across all val crops, all seeds) ===")
    print(control_df[["psnr_normal", "psnr_zero_freq", "psnr_mean_freq", "psnr_shuffled_freq"]].mean().to_string())

    print("\n=== Frequency control comparison BY DEGRADATION ===")
    print(control_df.groupby("degradation")[["psnr_normal", "psnr_zero_freq", "psnr_mean_freq",
                                               "psnr_shuffled_freq"]].mean().to_string())

    print("\n=== Coefficient magnitude and delta by degradation ===")
    print(coeff_df.groupby("degradation")[["coeff_magnitude_T14", "coeff_magnitude_F2", "delta_a0",
                                             "delta_a1"]].mean().to_string())

    print("\n=== T14-F2 restoration delta from paired-sample PSNR (sanity cross-check) ===")
    coeff_df["delta_psnr_sample"] = coeff_df.psnr_T14 - coeff_df.psnr_F2
    print(coeff_df.groupby("degradation")["delta_psnr_sample"].mean().to_string())


if __name__ == "__main__":
    main()
