"""TEST06 Phase 1-2 (06-A): resolution sweep -- measure mask activation and
restoration sanity across input resolutions and aspect ratios, using the
released, UNMODIFIED AdaIR checkpoint (read-only import of test01's
model_variants.load_variant('released'), which instruments -- does not
modify -- the released forward pass).

Two source pools:
  (a) REAL benchmark images at NATIVE resolution (CBSD68, Rain100L test) --
      answers "where does standard published evaluation actually sit
      relative to the activation threshold" (Phase 1B).
  (b) DIV2K validation images (2K resolution, documented provenance,
      downloaded from the official ETH Zurich CVL mirror) as the controlled
      high-resolution mechanism-validation set (Phase 1C), center-cropped
      to the full resolution x aspect-ratio grid (Phase 1A).

For every (image, target H, target W) combination: pad to a multiple of 16
(matches project convention), synthesize ONE degradation (Rain, for cross-
comparability) via degradation_synthesis.py, run through AdaIR, and record
per-AFLB mask/energy statistics plus PSNR/SSIM/NaN-Inf sanity checks.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python resolution_sweep.py
"""
from __future__ import annotations

import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST06 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST06.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
sys.path.insert(0, str(TEST06 / "src"))
from model_variants import load_variant  # noqa: E402 (read-only reuse, test01)
from metrics_utils import psnr_ssim_mse  # noqa: E402 (read-only reuse, test04)
from degradation_synthesis import SYNTH_FUNCS  # noqa: E402

CBSD68_DIR = Path("/home/minura/FYP/Workspace/Himeth/data/CBSD68-dataset/CBSD68/original_png")
RAIN100L_DIR = Path("/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L")
DIV2K_DIR = TEST06 / "data" / "div2k_val" / "DIV2K_valid_HR"
ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
OUT_DIR = TEST06 / "results" / "resolution_sweep"

RESOLUTIONS = [128, 256, 320, 384, 480, 512, 640, 768, 896, 1024, 1280, 1536]
ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9), "2:1": (2, 1)}
AFLBS = ["AFLB1", "AFLB2", "AFLB3"]
N_DIV2K_IMAGES = 8
DEG = "Rain"  # single representative degradation for the main sweep, for cross-comparability


def crop_img(image: np.ndarray, base=16):
    h, w = image.shape[:2]
    crop_h = h - h % base
    crop_w = w - w % base
    return image[:crop_h, :crop_w]


def center_crop_or_pad(img: np.ndarray, target_h: int, target_w: int):
    h, w = img.shape[:2]
    if h < target_h or w < target_w:
        return None  # source too small, skip -- documented as a skip, not silently resized up via interpolation
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[top:top + target_h, left:left + target_w]


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def mask_stats(mask_t: torch.Tensor, raw_low: torch.Tensor, raw_high: torch.Tensor):
    # mask is binary by construction (0/1 box, see source audit) -- avoid torch.unique()
    # (an O(n log n) sort) on tensors that can exceed 50M elements at high resolution;
    # min/max fully characterize a binary tensor's distinct-value count.
    m = mask_t.detach().float()
    m_min, m_max = float(m.min().item()), float(m.max().item())
    active_frac = float((m > 0.5).float().mean().item())
    low_e = float((raw_low.detach().float() ** 2).sum().item())
    high_e = float((raw_high.detach().float() ** 2).sum().item())
    total_e = low_e + high_e + 1e-12
    return {
        "mask_active_fraction": active_frac,
        "mask_unique_values": 1 if m_min == m_max else 2,
        "mask_min": m_min, "mask_max": m_max,
        "mask_mean": float(m.mean().item()), "mask_std": float(m.std().item()),
        "raw_low_energy": low_e, "raw_high_energy": high_e,
        "raw_low_energy_fraction": low_e / total_e, "raw_high_energy_fraction": high_e / total_e,
    }


def run_one(model, recorder, clean_np, seed, source_tag, resolution, aspect_tag, aspect_ratio):
    rng = np.random.RandomState(seed)
    degraded_np = SYNTH_FUNCS[DEG](clean_np, rng)
    device = next(model.parameters()).device
    clean_t = to_tensor(clean_np, device)
    degraded_t = to_tensor(degraded_np, device)

    recorder.start()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model(degraded_t)
    except torch.cuda.OutOfMemoryError:
        del clean_t, degraded_t
        torch.cuda.empty_cache()
        return {"source": source_tag, "resolution": resolution, "aspect": aspect_tag,
                "aspect_ratio": aspect_ratio, "input_h": clean_np.shape[0], "input_w": clean_np.shape[1],
                "oom": True}
    latency_ms = (time.time() - t0) * 1000
    snap = recorder.snapshot_cpu()

    m = psnr_ssim_mse(out, clean_t)
    out_f = out.detach().float()
    row = {
        "source": source_tag, "resolution": resolution, "aspect": aspect_tag,
        "aspect_ratio": aspect_ratio, "input_h": clean_np.shape[0], "input_w": clean_np.shape[1],
        "psnr": m["psnr"], "ssim": m["ssim"], "mse": m["mse"],
        "nan_count": int(torch.isnan(out_f).sum().item()), "inf_count": int(torch.isinf(out_f).sum().item()),
        "output_min": float(out_f.min().item()), "output_max": float(out_f.max().item()),
        "latency_ms": latency_ms, "oom": False,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else float("nan"),
    }
    for aflb in AFLBS:
        d = snap[aflb]
        feat_h, feat_w = d["mask"].shape[-2:]
        stats = mask_stats(d["mask"], d["raw_low"], d["raw_high"])
        alpha_beta = d["threshold_alpha_beta"]
        row[f"{aflb}_feat_h"] = feat_h
        row[f"{aflb}_feat_w"] = feat_w
        row[f"{aflb}_alpha"] = float(alpha_beta[0, 0, 0, 0].item())
        row[f"{aflb}_beta"] = float(alpha_beta[0, 1, 0, 0].item())
        for k, v in stats.items():
            row[f"{aflb}_{k}"] = v
    del out, out_f, clean_t, degraded_t
    torch.cuda.empty_cache()
    return row


def main():
    device = "cuda"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, "released")
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()

    # ---- Phase 1B: real benchmark images at NATIVE resolution ----
    native_sources = (
        [(p, "CBSD68") for p in sorted(glob.glob(str(CBSD68_DIR / "*.png")))[:10]] +
        [(p, "Rain100L") for p in sorted(glob.glob(str(RAIN100L_DIR / "norain-*.png")))[:10]]
    )
    for i, (path, tag) in enumerate(native_sources):
        img = crop_img(np.array(Image.open(path).convert("RGB")))
        row = run_one(model, recorder, img, seed=1000 + i, source_tag=f"native_{tag}",
                      resolution="native", aspect_tag="native", aspect_ratio=img.shape[1] / img.shape[0])
        rows.append(row)
        psnr_str = f"{row['psnr']:.2f}" if not row.get("oom") else "OOM"
        print(f"[native {tag}] {path} {img.shape[:2]} psnr={psnr_str}", flush=True)

    # ---- Phase 1A/1C: controlled resolution x aspect grid on DIV2K ----
    div2k_files = sorted(glob.glob(str(DIV2K_DIR / "*.png")))[:N_DIV2K_IMAGES]
    if not div2k_files:
        print(f"WARNING: no DIV2K images found at {DIV2K_DIR} -- skipping controlled grid", flush=True)
    n_done, n_skipped = 0, 0
    for img_idx, path in enumerate(div2k_files):
        full_img = np.array(Image.open(path).convert("RGB"))
        for res in RESOLUTIONS:
            for aspect_tag, (rw, rh) in ASPECTS.items():
                if rw >= rh:
                    target_h, target_w = res, int(round(res * rw / rh))
                else:
                    target_w, target_h = res, int(round(res * rh / rw))
                target_h -= target_h % 16
                target_w -= target_w % 16
                crop = center_crop_or_pad(full_img, target_h, target_w)
                if crop is None:
                    n_skipped += 1
                    continue
                seed = 2000 + img_idx * 1000 + res
                print(f"  starting img={img_idx} res={res} aspect={aspect_tag} shape={crop.shape[:2]}",
                      flush=True)
                row = run_one(model, recorder, crop, seed=seed, source_tag=f"div2k_{img_idx:03d}",
                              resolution=res, aspect_tag=aspect_tag, aspect_ratio=rw / rh)
                rows.append(row)
                n_done += 1
                if row.get("oom"):
                    print(f"  OOM at res={res} aspect={aspect_tag} img={img_idx} -- skipped, continuing", flush=True)
                if n_done % 20 == 0:
                    print(f"[{n_done} done, {n_skipped} skipped] res={res} aspect={aspect_tag} "
                          f"elapsed={time.time()-t_start:.0f}s", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_DIR / "mask_activation.csv", index=False)  # checkpoint

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mask_activation.csv", index=False)
    n_oom = int(df["oom"].sum()) if "oom" in df.columns else 0
    print(f"\nwrote {OUT_DIR / 'mask_activation.csv'} ({len(df)} rows, {n_skipped} skipped -- source too small, "
          f"{n_oom} OOM)")

    sanity_cols = ["source", "resolution", "aspect", "input_h", "input_w", "psnr", "ssim", "mse",
                   "nan_count", "inf_count", "output_min", "output_max", "latency_ms", "peak_memory_mb", "oom"]
    df[[c for c in sanity_cols if c in df.columns]].to_csv(OUT_DIR / "restoration_sanity.csv", index=False)
    print(f"wrote {OUT_DIR / 'restoration_sanity.csv'}")


if __name__ == "__main__":
    main()
