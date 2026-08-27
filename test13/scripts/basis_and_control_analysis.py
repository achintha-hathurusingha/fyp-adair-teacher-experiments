"""TEST13: basis-adaptation analysis (||dU||, ||dV||, relative change,
by degradation), effective-basis-complexity (per-sample singular values
of U(e)/V(e), and effective rank / diversity of dU across the validation
set), and the mandatory 5-condition causal control (normal, zero-e_D,
mean-content, shuffled-content, shuffled-basis-state).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python basis_and_control_analysis.py
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

TEST13 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST13.parent
sys.path.insert(0, str(TEST13 / "scripts"))
from models import MODELS, PCA_DIM, BOTTLENECK_CHAN, RANK  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST13 / "results" / "checkpoints"
OUT_DIR = TEST13 / "results" / "statistics"
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

    basis_rows, control_rows, rank_rows = [], [], []

    for seed in SEEDS:
        t13_ckpt = CKPT_DIR / f"model_T13_seed{seed}.pt"
        t13 = MODELS["T13"]().to(device)
        t13.load_state_dict(torch.load(t13_ckpt, map_location=device, weights_only=True))
        t13.eval()

        U0 = t13.U0.detach()
        V0 = t13.V0.detach()
        U0_norm = float(torch.norm(U0))
        V0_norm = float(torch.norm(V0))

        # ---- phi_bar: fixed dataset mean, TRAINING crops only ----
        phi_accum = []
        with torch.no_grad():
            for row in train_rows[:200]:
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    _, e_d, a, phi, du, dv = t13.forward_diagnostics(img_t)
                    phi_accum.append(phi[0].cpu().numpy())
        phi_bar = torch.from_numpy(np.mean(phi_accum, axis=0)).float().unsqueeze(0).to(device)

        samples = []
        du_all, dv_all = [], []
        with torch.no_grad():
            for row in val_rows:
                gt = load_rgb(row["clean_path"], device)
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    out, e_d, a, phi, du, dv = t13.forward_diagnostics(img_t)
                    psnr, ssim = psnr_ssim(out[0], gt[0])

                    du_norm = float(torch.norm(du[0]))
                    dv_norm = float(torch.norm(dv[0]))
                    U_sample = U0 + du[0]
                    V_sample = V0 + dv[0]
                    sv_u = torch.linalg.svdvals(U_sample).cpu().numpy()
                    sv_v = torch.linalg.svdvals(V_sample).cpu().numpy()

                    basis_rows.append({
                        "seed": seed, "scene_id": row["scene_id"], "degradation": deg,
                        "delta_U_norm": du_norm, "delta_V_norm": dv_norm,
                        "relative_basis_change_U": du_norm / (U0_norm + 1e-12),
                        "relative_basis_change_V": dv_norm / (V0_norm + 1e-12),
                        "sv_U_0": float(sv_u[0]), "sv_U_1": float(sv_u[1]),
                        "sv_V_0": float(sv_v[0]), "sv_V_1": float(sv_v[1]),
                        "psnr": psnr, "ssim": ssim,
                    })
                    du_all.append(du[0].flatten().cpu().numpy())
                    dv_all.append(dv[0].flatten().cpu().numpy())

                    samples.append({"scene_id": row["scene_id"], "degradation": deg, "img_t": img_t, "gt": gt,
                                     "e_d": e_d, "phi": phi})

        eff_rank_du = effective_rank(np.stack(du_all))
        eff_rank_dv = effective_rank(np.stack(dv_all))
        rank_rows.append({"seed": seed, "effective_rank_deltaU": eff_rank_du,
                           "effective_rank_deltaV": eff_rank_dv, "configured_rank": RANK})
        print(f"seed{seed}: basis analysis done. eff_rank(dU)={eff_rank_du:.2f} eff_rank(dV)={eff_rank_dv:.2f}",
              flush=True)

        # ---- 5 causal controls ----
        with torch.no_grad():
            for i, s in enumerate(samples):
                zero_e_d = torch.zeros(1, PCA_DIM, device=device)

                out_norm, _ = t13.forward_with_override(s["img_t"])
                psnr_n, ssim_n = psnr_ssim(out_norm[0], s["gt"][0])

                out_zero, _ = t13.forward_with_override(s["img_t"], e_d_override=zero_e_d)
                psnr_z, ssim_z = psnr_ssim(out_zero[0], s["gt"][0])

                out_mean, _ = t13.forward_with_override(s["img_t"], phi_override=phi_bar)
                psnr_m, ssim_m = psnr_ssim(out_mean[0], s["gt"][0])

                j = np.random.RandomState(2000 + i).randint(0, len(samples) - 1)
                if j >= i:
                    j += 1
                out_shuf_c, _ = t13.forward_with_override(s["img_t"], phi_override=samples[j]["phi"])
                psnr_sc, ssim_sc = psnr_ssim(out_shuf_c[0], s["gt"][0])

                k = np.random.RandomState(3000 + i).randint(0, len(samples) - 1)
                if k >= i:
                    k += 1
                out_shuf_b, _ = t13.forward_with_override(s["img_t"], e_d_override=samples[k]["e_d"])
                psnr_sb, ssim_sb = psnr_ssim(out_shuf_b[0], s["gt"][0])

                control_rows.append({
                    "seed": seed, "scene_id": s["scene_id"], "degradation": s["degradation"],
                    "psnr_normal": psnr_n, "ssim_normal": ssim_n,
                    "psnr_zero_eD": psnr_z, "ssim_zero_eD": ssim_z,
                    "psnr_mean_content": psnr_m, "ssim_mean_content": ssim_m,
                    "psnr_shuffled_content": psnr_sc, "ssim_shuffled_content": ssim_sc,
                    "psnr_shuffled_basis_state": psnr_sb, "ssim_shuffled_basis_state": ssim_sb,
                })

        print(f"seed{seed}: causal-control analysis done ({len(samples)} val crops)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basis_df = pd.DataFrame(basis_rows)
    basis_df.to_csv(OUT_DIR / "basis_adaptation.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'basis_adaptation.csv'}")

    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(OUT_DIR / "content_controls.csv", index=False)
    print(f"wrote {OUT_DIR / 'content_controls.csv'}")

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_DIR / "basis_effective_rank.csv", index=False)
    print(f"wrote {OUT_DIR / 'basis_effective_rank.csv'}")

    print("\n=== ||dU||, ||dV||, relative change by degradation (mean across seeds) ===")
    print(basis_df.groupby("degradation")[["delta_U_norm", "delta_V_norm", "relative_basis_change_U",
                                             "relative_basis_change_V"]].mean().to_string())

    print("\n=== Singular value variance by degradation (mean across seeds) ===")
    print(basis_df.groupby("degradation")[["sv_U_0", "sv_U_1", "sv_V_0", "sv_V_1"]].agg(["mean", "std"]).to_string())

    print("\n=== Effective rank of dU / dV (mean across seeds) ===")
    print(rank_df[["effective_rank_deltaU", "effective_rank_deltaV"]].mean().to_string())

    print("\n=== Causal control comparison (mean PSNR across all val crops, all seeds) ===")
    print(control_df[["psnr_normal", "psnr_zero_eD", "psnr_mean_content", "psnr_shuffled_content",
                       "psnr_shuffled_basis_state"]].mean().to_string())


if __name__ == "__main__":
    main()
