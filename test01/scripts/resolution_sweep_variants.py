"""Phase 9: resolution sweep across all 3 model variants (released,
modified_mask, no_frequency), at the requested feature-map-adjacent sizes,
including rectangular (non-square) inputs.

Feature-map size at each AFLB is a fixed fraction of the INPUT resolution
(AFLB1=H/8, AFLB2=H/4, AFLB3=H/2). The task asked for feature-map sizes of
approximately 64/128/160/256/320/512 -- we sweep INPUT resolutions chosen so
that AFLB3 (H/2, the shallowest/most-sensitive AFLB) lands close to each of
those targets, and record the actual resulting feature size at every AFLB
(not just AFLB3) since they differ.

  target AFLB3 feature ~= input / 2  =>  input ~= 2 * target
  64 -> 128, 128 -> 256, 160 -> 320, 256 -> 512, 320 -> 640, 512 -> 1024

Square AND rectangular (H != W) variants are both tested.

This does NOT modify the model -- for the `released` and `modified_mask`
variants it only reads out alpha/beta/mask under the (unmodified) trained
rate_conv weights; `no_frequency` has no mask to sweep (recorded as N/A)
but is included for completeness / cost comparison at each resolution.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python resolution_sweep_variants.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST01 = Path(__file__).resolve().parent.parent
REPO = TEST01.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(TEST01 / "scripts"))
from model_variants import load_variant, VARIANTS  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
OUT_CSV = TEST01 / "results" / "resolution_sweep.csv"
SOURCE_IMAGE = "/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L/rainy/rain-001.png"

# (H, W) pairs -- square and rectangular, targeting AFLB3 feature ~= 64/128/160/256/320/512
SIZES = [
    (128, 128), (256, 256), (320, 320), (512, 512), (640, 640), (1024, 1024),  # square
    (256, 128), (512, 256), (640, 320), (1024, 512),                           # rectangular (2:1)
]
AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]


def round_to_multiple(x: int, base: int = 16) -> int:
    return max(base, (x // base) * base)


def main():
    device = "cuda"
    img = Image.open(SOURCE_IMAGE).convert("RGB")
    rows = []

    for variant in VARIANTS:
        print(f"\n=== variant: {variant} ===", flush=True)
        model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, variant)

        for (target_h, target_w) in SIZES:
            H = round_to_multiple(target_h)
            W = round_to_multiple(target_w)
            resized = img.resize((W, H), Image.BICUBIC)
            arr = np.array(resized).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            recorder.start()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                with torch.no_grad():
                    _ = model(t)
            except torch.cuda.OutOfMemoryError:
                print(f"  H={H} W={W}: OOM, skipping")
                torch.cuda.empty_cache()
                continue
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            snap = recorder.snapshot_cpu()

            for aflb in AFLB_NAMES:
                d = snap[aflb]
                conv_feat = d["conv_feat"]
                h, w = conv_feat.shape[-2:]
                n = 128
                row = {
                    "variant": variant, "input_H": H, "input_W": W,
                    "AFLB": aflb, "feature_h": h, "feature_w": w, "n": n,
                    "peak_memory_mb": peak_mem_mb,
                }
                if variant == "no_frequency":
                    row.update({"alpha": "", "beta": "", "h_over_n": "", "w_over_n": "",
                                "mask_area_pct": "", "low_energy_pct": "", "high_energy_pct": ""})
                else:
                    th = d["threshold_alpha_beta"]
                    alpha = th[0, 0, 0, 0].item()
                    beta = th[0, 1, 0, 0].item()
                    mask = d["mask"]
                    mask_pct = mask.float().mean().item() * 100
                    fft = d["fft_shifted"]
                    energy_total = (fft.abs() ** 2).sum().item()
                    energy_low = ((fft * mask).abs() ** 2).sum().item()
                    energy_high = ((fft * (1 - mask)).abs() ** 2).sum().item()
                    row.update({
                        "alpha": alpha, "beta": beta,
                        "h_over_n": h // n, "w_over_n": w // n,
                        "mask_area_pct": mask_pct,
                        "low_energy_pct": 100 * energy_low / max(energy_total, 1e-12),
                        "high_energy_pct": 100 * energy_high / max(energy_total, 1e-12),
                    })
                rows.append(row)
            print(f"  H={H:5d} W={W:5d}  " +
                  "  ".join(f"{a}:feat=({snap[a]['conv_feat'].shape[-2]},{snap[a]['conv_feat'].shape[-1]}) "
                            f"mask%={rows[-3 + i]['mask_area_pct']}"
                            for i, a in enumerate(AFLB_NAMES)), flush=True)
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
