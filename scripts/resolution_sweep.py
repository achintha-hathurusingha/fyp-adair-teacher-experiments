"""Resolution sweep: at what input resolution does FreModule's low-frequency
mask actually turn on?

Feature-map size at each AFLB is a fixed fraction of the input resolution
(AdaIR downsamples by 2 three times before the latent):
  AFLB1 (fre1, on `latent`)          : H/8 x W/8
  AFLB2 (fre2, on decoder_level3)    : H/4 x W/4
  AFLB3 (fre3, on decoder_level2)    : H/2 x W/2

The mask box half-width is h_ = int((h//128) * alpha). For alpha ~= 0.5 (what
we observed at native resolution on R001), h//128 must reach >= 2 before
int() stops truncating to 0. So the predicted activation thresholds are
roughly:
  AFLB3 (h=H/2):  H/2 // 128 >= 2  ->  H >= 512
  AFLB2 (h=H/4):  H/4 // 128 >= 2  ->  H >= 1024
  AFLB1 (h=H/8):  H/8 // 128 >= 2  ->  H >= 2048

This script resizes one real image (R001, rainy) to a sweep of target sizes,
runs it through the full AdaIR forward pass (instrumented), and records
alpha/beta/mask stats per AFLB at each resolution -- confirming or refuting
those predicted thresholds empirically rather than by arithmetic alone.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python resolution_sweep.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import Recorder, attach_instrumentation, load_adair

REPO = Path(__file__).resolve().parent.parent
ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
OUT_CSV = REPO / "csv_export" / "11_Resolution_Sweep.csv"

SOURCE_IMAGE = "/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L/rainy/rain-001.png"
SIZES = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]


def round_to_multiple(x: int, base: int = 16) -> int:
    return max(base, (x // base) * base)


def main():
    device = "cuda"
    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)

    img = Image.open(SOURCE_IMAGE).convert("RGB")
    rows = []

    for size in SIZES:
        target = round_to_multiple(size)
        resized = img.resize((target, target), Image.BICUBIC)
        arr = np.array(resized).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        recorder.start()
        try:
            with torch.no_grad():
                _ = model(t)
        except torch.cuda.OutOfMemoryError:
            print(f"size={target}: OOM, skipping")
            torch.cuda.empty_cache()
            continue
        snap = recorder.snapshot_cpu()

        for aflb in AFLB_NAMES:
            d = snap[aflb]
            conv_feat = d["conv_feat"]
            th = d["threshold_alpha_beta"]
            mask = d["mask"]
            h, w = conv_feat.shape[-2:]
            n = 128
            alpha = th[0, 0, 0, 0].item()
            beta = th[0, 1, 0, 0].item()
            h_ = int(h // n * alpha)
            w_ = int(w // n * beta)
            mask_pct = mask.float().mean().item() * 100

            row = {
                "input_size": target, "AFLB": aflb,
                "feature_h": h, "feature_w": w,
                "h_over_128": h // n, "w_over_128": w // n,
                "alpha": alpha, "beta": beta,
                "h_half_width": h_, "w_half_width": w_,
                "mask_area_pct": mask_pct,
                "mask_active": mask_pct > 0,
            }
            rows.append(row)
            print(f"size={target:5d} {aflb}: feat=({h:4d},{w:4d}) h//128={h//n} alpha={alpha:.4f} "
                  f"h_={h_} mask%={mask_pct:.4f} {'ACTIVE' if mask_pct > 0 else ''}")

        torch.cuda.empty_cache()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")

    print("\nFirst activating resolution per AFLB:")
    for aflb in AFLB_NAMES:
        active = [r for r in rows if r["AFLB"] == aflb and r["mask_active"]]
        if active:
            print(f"  {aflb}: first active at input_size={min(r['input_size'] for r in active)}")
        else:
            print(f"  {aflb}: never activated in sweep range {SIZES}")


if __name__ == "__main__":
    main()
