"""TEST12: coefficient analysis + the mandatory causal controls
(degradation-only, content-only, shuffled-content) + Haze-specific
scene-variance analysis (T12 vs F2), on the 20 validation scenes x 3
degradations = 60 crops.

phi_bar (the fixed dataset mean used for the degradation-only control) is
computed from TRAINING crops only (leakage-safe, matching this project's
established convention for any fitted/summary statistic).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python coefficient_control_analysis.py
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

TEST12 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST12.parent
sys.path.insert(0, str(TEST12 / "scripts"))
from models import MODELS, PCA_DIM  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST12 / "results" / "checkpoints"
OUT_DIR = TEST12 / "results" / "statistics"
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
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    coeff_rows, control_rows, variance_rows, rank_rows = [], [], [], []

    for seed in SEEDS:
        t12_ckpt = CKPT_DIR / f"model_T12_seed{seed}.pt"
        f2_ckpt = CKPT_DIR / f"model_F2_seed{seed}.pt"
        t12 = MODELS["T12"]().to(device)
        t12.load_state_dict(torch.load(t12_ckpt, map_location=device, weights_only=True))
        t12.eval()
        f2 = MODELS["F2"]().to(device)
        f2.load_state_dict(torch.load(f2_ckpt, map_location=device, weights_only=True))
        f2.eval()

        # ---- phi_bar: fixed dataset mean of phi(F), TRAINING crops only, leakage-safe ----
        phi_accum = []
        with torch.no_grad():
            for row in train_rows[:200]:  # subsample for speed; still leakage-safe (train-only)
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    _, e_d, a, phi, F_pre, F_cond = t12.forward_diagnostics(img_t)
                    phi_accum.append(phi[0].cpu().numpy())
        phi_bar = torch.from_numpy(np.mean(phi_accum, axis=0)).float().unsqueeze(0).to(device)

        # ---- normal pass over validation set: collect e_d, phi, a, scene/deg tags ----
        samples = []
        with torch.no_grad():
            for row in val_rows:
                gt = load_rgb(row["clean_path"], device)
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    out, e_d, a, phi, F_pre, F_cond = t12.forward_diagnostics(img_t)
                    psnr, ssim = psnr_ssim(out[0], gt[0])
                    rel_change = float(torch.norm(F_cond - F_pre) / (torch.norm(F_pre) + 1e-12))

                    _, e_d_f2, a_f2 = f2.forward_diagnostics(img_t)

                    samples.append({"scene_id": row["scene_id"], "degradation": deg, "img_t": img_t, "gt": gt,
                                     "e_d": e_d, "phi": phi, "a": a[0].cpu().numpy(), "a_f2": a_f2[0].cpu().numpy()})

                    coeff_rows.append({"seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                                        "model": "T12", "a0": float(a[0, 0]), "a1": float(a[0, 1]),
                                        "coeff_l2_magnitude": float(np.linalg.norm(a[0].cpu().numpy())),
                                        "relative_modulation_magnitude": rel_change,
                                        "psnr_normal": psnr, "ssim_normal": ssim})
                    coeff_rows.append({"seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                                        "model": "F2", "a0": float(a_f2[0, 0]), "a1": float(a_f2[0, 1]),
                                        "coeff_l2_magnitude": float(np.linalg.norm(a_f2[0].cpu().numpy())),
                                        "relative_modulation_magnitude": None,
                                        "psnr_normal": None, "ssim_normal": None})

        # ---- causal controls ----
        with torch.no_grad():
            for i, s in enumerate(samples):
                zero_e_d = torch.zeros(1, PCA_DIM, device=device)

                # degradation-only: a(e_D, phi_bar)
                out_deg, a_deg = t12.forward_with_override(s["img_t"], phi_override=phi_bar)
                psnr_deg, ssim_deg = psnr_ssim(out_deg[0], s["gt"][0])

                # content-only: a(0, phi(F))
                out_cont, a_cont = t12.forward_with_override(s["img_t"], e_d_override=zero_e_d)
                psnr_cont, ssim_cont = psnr_ssim(out_cont[0], s["gt"][0])

                # shuffled content: a(e_D_i, phi(F_j)), j != i random
                j = np.random.RandomState(1000 + i).randint(0, len(samples) - 1)
                if j >= i:
                    j += 1
                out_shuf, a_shuf = t12.forward_with_override(s["img_t"], phi_override=samples[j]["phi"])
                psnr_shuf, ssim_shuf = psnr_ssim(out_shuf[0], s["gt"][0])

                normal_psnr = next(c["psnr_normal"] for c in coeff_rows
                                   if c["model"] == "T12" and c["seed"] == seed and c["scene_id"] == s["scene_id"]
                                   and c["degradation"] == s["degradation"])
                control_rows.append({
                    "seed": seed, "scene_id": s["scene_id"], "degradation": s["degradation"],
                    "psnr_normal": normal_psnr,
                    "psnr_degradation_only": psnr_deg, "ssim_degradation_only": ssim_deg,
                    "psnr_content_only": psnr_cont, "ssim_content_only": ssim_cont,
                    "psnr_shuffled_content": psnr_shuf, "ssim_shuffled_content": ssim_shuf,
                    "a0_degradation_only": float(a_deg[0, 0]), "a1_degradation_only": float(a_deg[0, 1]),
                    "a0_content_only": float(a_cont[0, 0]), "a1_content_only": float(a_cont[0, 1]),
                    "a0_shuffled": float(a_shuf[0, 0]), "a1_shuffled": float(a_shuf[0, 1]),
                    "donor_scene_id": samples[j]["scene_id"], "donor_degradation": samples[j]["degradation"],
                })

        print(f"seed{seed}: coefficient + causal-control analysis done ({len(samples)} val crops)", flush=True)

        # ---- Haze scene-variance analysis: does a vary across scenes within one degradation? ----
        t12_df_seed = pd.DataFrame([c for c in coeff_rows if c["model"] == "T12" and c["seed"] == seed])
        f2_df_seed = pd.DataFrame([c for c in coeff_rows if c["model"] == "F2" and c["seed"] == seed])
        for deg in DEGS:
            t12_a = t12_df_seed[t12_df_seed.degradation == deg][["a0", "a1"]].values
            f2_a = f2_df_seed[f2_df_seed.degradation == deg][["a0", "a1"]].values
            variance_rows.append({"seed": seed, "degradation": deg,
                                   "T12_var_a0_across_scenes": float(np.var(t12_a[:, 0])),
                                   "T12_var_a1_across_scenes": float(np.var(t12_a[:, 1])),
                                   "F2_var_a0_across_scenes": float(np.var(f2_a[:, 0])),
                                   "F2_var_a1_across_scenes": float(np.var(f2_a[:, 1])),
                                   "T12_over_F2_variance_ratio_a0": float(np.var(t12_a[:, 0]) / (np.var(f2_a[:, 0]) + 1e-12)),
                                   "T12_over_F2_variance_ratio_a1": float(np.var(t12_a[:, 1]) / (np.var(f2_a[:, 1]) + 1e-12))})

        # ---- effective rank of T12's coefficient matrix ----
        all_a = t12_df_seed[["a0", "a1"]].values
        rank_rows.append({"seed": seed, "model": "T12", "configured_rank": 2, "effective_rank": effective_rank(all_a)})
        all_a_f2 = f2_df_seed[["a0", "a1"]].values
        rank_rows.append({"seed": seed, "model": "F2", "configured_rank": 2, "effective_rank": effective_rank(all_a_f2)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeff_df = pd.DataFrame(coeff_rows)
    coeff_df.to_csv(OUT_DIR / "coefficient_analysis.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'coefficient_analysis.csv'}")

    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(OUT_DIR / "content_shuffle_controls.csv", index=False)
    print(f"wrote {OUT_DIR / 'content_shuffle_controls.csv'}")

    variance_df = pd.DataFrame(variance_rows)
    variance_df.to_csv(OUT_DIR / "haze_scene_variance.csv", index=False)
    print(f"wrote {OUT_DIR / 'haze_scene_variance.csv'}")

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_DIR / "effective_rank.csv", index=False)
    print(f"wrote {OUT_DIR / 'effective_rank.csv'}")

    print("\n=== Control comparison (mean PSNR across all val crops, all seeds) ===")
    print(control_df[["psnr_normal", "psnr_degradation_only", "psnr_content_only",
                       "psnr_shuffled_content"]].mean().to_string())

    print("\n=== Scene-variance of coefficients by degradation (T12 vs F2, mean across seeds) ===")
    print(variance_df.groupby("degradation")[["T12_var_a0_across_scenes", "F2_var_a0_across_scenes",
                                                "T12_over_F2_variance_ratio_a0"]].mean().to_string())

    print("\n=== Effective rank (T12 vs F2, mean across seeds) ===")
    print(rank_df.groupby("model")["effective_rank"].mean().to_string())


if __name__ == "__main__":
    main()
