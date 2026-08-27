"""TEST05.5 Phase 5-6: NORMALIZED causal intervention, correcting TEST04's
loophole L4 (raw L2 output-change values were compared across tensors of
very different dimensionality/scale without normalization, inviting the
unsupported claim "AFLB3 is 3.8x more important").

For every intervention, in addition to raw L2, compute:
  normalized_change = ||Y_swap - Y_normal||_2 / ||Y_normal||_2   (relative)
  rms_change = ||Y_swap - Y_normal||_2 / sqrt(n_pixels)            (per-element RMS)
  psnr_change = PSNR(Y_swap, clean) - PSNR(Y_normal, clean)
  ssim_change = SSIM(Y_swap, clean) - SSIM(Y_normal, clean)

Also reproduces TEST04's controls (random/zero/mean/cross-scene) so the
degradation-specificity ratio = effect(same-scene cross-degradation) /
effect(cross-scene same-degradation) can be computed on NORMALIZED units,
directly answering Phase 6.

Uses TEST04's verified manual_forward (read-only import).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python normalized_intervention.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from instrument import load_adair  # noqa: E402
from intervention import manual_forward  # noqa: E402
from metrics_utils import psnr_ssim_mse  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEACHER_EXP / "test03" / "results" / "manifest" / "scene_manifest.csv"
OUT_DIR = TEST05_5 / "results" / "intervention"
DEGS = ["Rain", "Haze", "Noise"]
POINTS = ["latent_pre", "aflb1_out", "aflb2_out", "aflb3_out"]
N_SCENES = 30


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def run_normal(model, scene_row, device):
    clean_t = to_tensor(load_rgb(scene_row["clean_image_path"]), device)
    cache = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), device)
        out = manual_forward(model, img_t)
        out["input"] = img_t
        cache[deg] = out
    return clean_t, cache


def compute_metrics(swapped, normal, clean):
    diff = (swapped.detach().float() - normal.detach().float())
    l2 = float(torch.linalg.vector_norm(diff.reshape(-1)).item())
    normal_l2 = float(torch.linalg.vector_norm(normal.detach().float().reshape(-1)).item())
    n_pixels = normal.numel()
    normalized_change = l2 / (normal_l2 + 1e-12)
    rms_change = l2 / np.sqrt(n_pixels)
    m_swap = psnr_ssim_mse(swapped, clean)
    m_normal = psnr_ssim_mse(normal, clean)
    psnr_swap, ssim_swap = m_swap["psnr"], m_swap["ssim"]
    psnr_normal, ssim_normal = m_normal["psnr"], m_normal["ssim"]
    return {
        "l2_raw": l2, "normal_output_l2": normal_l2, "normalized_change": normalized_change,
        "rms_change": rms_change, "psnr_change": psnr_swap - psnr_normal, "ssim_change": ssim_swap - ssim_normal,
    }


def main():
    device = "cuda"
    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))[:N_SCENES]

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    print("checkpoint loaded", flush=True)

    rows = []
    t_start = time.time()
    for idx, scene_row in enumerate(scene_rows):
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)

        for point in POINTS:
            for recipient in DEGS:
                for donor in DEGS:
                    if donor == recipient:
                        continue
                    recipient_img = cache[recipient]["input"]
                    donor_tensor = cache[donor][point]
                    normal_out = cache[recipient]["output"]
                    swapped = manual_forward(model, recipient_img, overrides={point: donor_tensor})["output"]
                    m = compute_metrics(swapped, normal_out, clean_t)
                    rows.append({"scene_id": scene_id, "point": point, "recipient": recipient,
                                 "donor": donor, "condition": "same_scene_cross_degradation", **m})

        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{len(scene_rows)}] {scene_id} elapsed={time.time() - t_start:.0f}s", flush=True)

    # cross-scene same-degradation control (subset, for degradation-specificity ratio)
    normals = {sr["scene_id"]: run_normal(model, sr, device) for sr in scene_rows[:20]}
    scene_ids = list(normals.keys())
    skipped = 0
    for point in POINTS:
        for i in range(1, len(scene_ids)):
            recipient_scene, donor_scene = scene_ids[i], scene_ids[i - 1]
            r_clean, r_cache = normals[recipient_scene]
            d_clean, d_cache = normals[donor_scene]
            for deg in DEGS:
                r_tensor = r_cache[deg][point]
                d_tensor = d_cache[deg][point]
                if r_tensor.shape != d_tensor.shape:
                    skipped += 1
                    continue
                recipient_img = r_cache[deg]["input"]
                normal_out = r_cache[deg]["output"]
                swapped = manual_forward(model, recipient_img, overrides={point: d_tensor})["output"]
                m = compute_metrics(swapped, normal_out, r_clean)
                rows.append({"scene_id": recipient_scene, "point": point, "recipient": deg,
                             "donor": f"scene_{donor_scene}", "condition": "cross_scene_same_degradation", **m})
    print(f"skipped {skipped} shape-mismatched cross-scene pairs", flush=True)

    import pandas as pd
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "normalized_intervention.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'normalized_intervention.csv'} ({len(df)} rows)")

    summary = df.groupby(["point", "condition"])[["l2_raw", "normalized_change", "rms_change",
                                                     "psnr_change", "ssim_change"]].mean().reset_index()
    summary.to_csv(OUT_DIR / "normalized_intervention_summary.csv", index=False)
    print(summary.to_string(index=False))

    # degradation-specificity ratio, normalized units
    ratio_rows = []
    for point in POINTS:
        a = df[(df.point == point) & (df.condition == "same_scene_cross_degradation")]
        b = df[(df.point == point) & (df.condition == "cross_scene_same_degradation")]
        for metric in ["l2_raw", "normalized_change", "rms_change"]:
            ratio_rows.append({"point": point, "metric": metric,
                                "effect_A_same_scene_cross_deg": a[metric].mean(),
                                "effect_B_cross_scene_same_deg": b[metric].mean(),
                                "degradation_specificity_ratio": a[metric].mean() / (b[metric].mean() + 1e-12)})
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(OUT_DIR / "degradation_specificity_ratio.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'degradation_specificity_ratio.csv'}")
    print(ratio_df.to_string(index=False))


if __name__ == "__main__":
    main()
