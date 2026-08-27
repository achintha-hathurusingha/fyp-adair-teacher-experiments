"""TEST18 Phase 6: quantitative PSNR/SSIM evaluation of all 5 ablation
variants on real, held-out test data (genuinely disjoint from what each
variant was trained on) -- matching the shape of the paper's own Table 7.

  Dehaze: RESIDE SOTS-outdoor test set (500 images) -- a DIFFERENT RESIDE
          split from OTS (used for training); no overlap.
  Derain: Rain100L's own official test split (100 pairs, rain100L_test/) --
          disjoint from RainTrainL (used for training).
  Denoise: SOTS's clean "target" images (492 real photos, disjoint from
           the DIV2K pool used for training) with synthetic Gaussian noise
           added at test time (sigma=15/25/50), matching AdaIR's own
           DenoiseTestDataset convention. Documented substitution for the
           standard CBSD68 benchmark, which was not found anywhere on
           devon (see TEST18_PLAN.md).

Usage (devon, adair-distill env):
  python eval_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TEST18 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST18 / "scripts"))
from ablatable_model import build_variant, VARIANTS  # noqa: E402

CKPT_DIR = TEST18 / "results" / "checkpoints"
STATS_DIR = TEST18 / "results" / "statistics"
STATS_DIR.mkdir(parents=True, exist_ok=True)

SOTS_INPUT = Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/input")
SOTS_TARGET = Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/target")
RAIN100L_TEST = Path("/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L")
N_EVAL_PER_TASK = 60  # subsample for tractable eval time across 5 variants x 3 tasks x 3 sigmas


def load_img(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def to_numpy_u8(t):
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (arr * 255).round().astype(np.uint8)


def crop16(img):
    h, w = img.shape[:2]
    return img[: h - h % 16, : w - w % 16]


def psnr_ssim(pred, target):
    h = min(pred.shape[0], target.shape[0])
    w = min(pred.shape[1], target.shape[1])
    pred, target = pred[:h, :w], target[:h, :w]
    p = float(peak_signal_noise_ratio(target, pred, data_range=255))
    s = float(structural_similarity(target, pred, data_range=255, channel_axis=2))
    return p, s


def eval_dehaze(model, device, n=N_EVAL_PER_TASK):
    files = sorted(SOTS_INPUT.glob("*.jpg"))[:n]
    rows = []
    for f in files:
        target_name = f.name.split("_")[0] + ".png"
        target_path = SOTS_TARGET / target_name
        if not target_path.exists():
            continue
        inp = crop16(load_img(f))
        tgt = crop16(load_img(target_path))
        with torch.no_grad():
            out = model(to_tensor(inp).to(device))
        pred = to_numpy_u8(out)
        p, s = psnr_ssim(pred, tgt)
        rows.append({"degradation": "dehaze", "image": f.name, "psnr": p, "ssim": s})
    return rows


def eval_derain(model, device, n=N_EVAL_PER_TASK):
    files = sorted((RAIN100L_TEST / "rainy").glob("rain-*.png"))[:n]
    rows = []
    for f in files:
        idx = f.stem.split("-")[-1]
        target_path = RAIN100L_TEST / f"norain-{idx}.png"
        if not target_path.exists():
            continue
        inp = crop16(load_img(f))
        tgt = crop16(load_img(target_path))
        with torch.no_grad():
            out = model(to_tensor(inp).to(device))
        pred = to_numpy_u8(out)
        p, s = psnr_ssim(pred, tgt)
        rows.append({"degradation": "derain", "image": f.name, "psnr": p, "ssim": s})
    return rows


def eval_denoise(model, device, n=N_EVAL_PER_TASK):
    files = sorted(SOTS_TARGET.glob("*.png"))[:n]
    rows = []
    rng = np.random.default_rng(0)
    for sigma in (15, 25, 50):
        for f in files:
            clean = crop16(load_img(f))
            noise = rng.standard_normal(clean.shape) * sigma
            noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            with torch.no_grad():
                out = model(to_tensor(noisy).to(device))
            pred = to_numpy_u8(out)
            p, s = psnr_ssim(pred, clean)
            rows.append({"degradation": f"denoise_sigma{sigma}", "image": f.name, "psnr": p, "ssim": s})
    return rows


def main():
    device = "cuda"
    all_rows = []
    for variant in VARIANTS:
        ckpt_path = CKPT_DIR / f"model_{variant}_final.pt"
        model = build_variant(variant).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        rows = eval_dehaze(model, device) + eval_derain(model, device) + eval_denoise(model, device)
        for r in rows:
            r["variant"] = variant
        all_rows.extend(rows)

        df_v = pd.DataFrame(rows)
        summary = df_v.groupby("degradation")[["psnr", "ssim"]].mean()
        print(f"[{variant}] overall PSNR={df_v.psnr.mean():.3f} SSIM={df_v.ssim.mean():.4f}", flush=True)
        print(summary.to_string(), flush=True)

        del model
        torch.cuda.empty_cache()

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(STATS_DIR / "eval_per_image.csv", index=False)

    summary_df = full_df.groupby(["variant", "degradation"])[["psnr", "ssim"]].mean().reset_index()
    summary_df.to_csv(STATS_DIR / "eval_summary_by_degradation.csv", index=False)

    overall_df = full_df.groupby("variant")[["psnr", "ssim"]].mean().reindex(list(VARIANTS.keys())).reset_index()
    overall_df.to_csv(STATS_DIR / "eval_summary_overall.csv", index=False)
    print("\n=== OVERALL (matches Table 7's shape) ===")
    print(overall_df.to_string(index=False))
    print(f"\nwrote eval_per_image.csv, eval_summary_by_degradation.csv, eval_summary_overall.csv")


if __name__ == "__main__":
    main()
