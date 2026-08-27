"""TEST08-C: student-side degradation embedding intervention (Model C only
-- the only model where e_S causally participates in restoration).

For each validation scene and each ordered (recipient_degradation,
donor_degradation) pair, take the recipient's degraded crop, but condition
the bottleneck with the DONOR's e_S (computed from the SAME scene's
donor-degradation crop) instead of the recipient's own e_S. This tests
whether the compact embedding actually controls restoration behavior, the
student-side counterpart to the teacher-side interventions in TEST06/06-R.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python embedding_intervention.py
"""
from __future__ import annotations

import csv
import itertools
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


def l2_dist(a, b):
    return float(torch.norm(a - b))


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    intervention_rows = []

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"model_C_seed{seed}.pt"
        model = MODELS["C"]().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            for row in val_rows:
                clean_t = load_rgb(row["clean_path"], device)
                deg_tensors = {deg: load_rgb(row[f"{deg.lower()}_path"], device) for deg in DEGS}

                # normal (own e_S) output + e_S per degradation, this scene
                normal_out, normal_es = {}, {}
                for deg in DEGS:
                    out, e_s = model(deg_tensors[deg])
                    normal_out[deg] = out
                    normal_es[deg] = e_s

                for recipient_deg, donor_deg in itertools.permutations(DEGS, 2):
                    recipient_input = deg_tensors[recipient_deg]
                    donor_e = normal_es[donor_deg]
                    intervened_out, gamma, beta = model.forward_with_override_embedding(recipient_input, donor_e)

                    psnr_vs_clean, ssim_vs_clean = psnr_ssim(intervened_out[0], clean_t[0])
                    dist_to_recipient_normal = l2_dist(intervened_out[0], normal_out[recipient_deg][0])
                    dist_to_donor_normal = l2_dist(intervened_out[0], normal_out[donor_deg][0])
                    recipient_norm = float(torch.norm(normal_out[recipient_deg][0]))

                    intervention_rows.append({
                        "seed": seed, "scene_id": row["scene_id"],
                        "recipient_degradation": recipient_deg, "donor_degradation": donor_deg,
                        "psnr_vs_clean": psnr_vs_clean, "ssim_vs_clean": ssim_vs_clean,
                        "recipient_normal_psnr": psnr_ssim(normal_out[recipient_deg][0], clean_t[0])[0],
                        "dist_to_recipient_normal_output": dist_to_recipient_normal,
                        "dist_to_donor_normal_output": dist_to_donor_normal,
                        "dist_ratio_recipient_vs_donor": dist_to_recipient_normal / (dist_to_donor_normal + 1e-12),
                        "relative_output_change": dist_to_recipient_normal / (recipient_norm + 1e-12),
                    })

        print(f"seed {seed}: embedding intervention done "
              f"({len(val_rows)} scenes x 6 ordered degradation pairs)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(intervention_rows)
    df.to_csv(OUT_DIR / "embedding_intervention.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'embedding_intervention.csv'}: {len(df)} rows")

    print("\n=== Mean effect by (recipient -> donor) direction, across seeds+scenes ===")
    summary = df.groupby(["recipient_degradation", "donor_degradation"])[
        ["psnr_vs_clean", "recipient_normal_psnr", "dist_to_recipient_normal_output",
         "dist_to_donor_normal_output", "relative_output_change"]].mean()
    summary["delta_psnr_vs_recipient_normal"] = summary["psnr_vs_clean"] - summary["recipient_normal_psnr"]
    print(summary.to_string())


if __name__ == "__main__":
    main()
