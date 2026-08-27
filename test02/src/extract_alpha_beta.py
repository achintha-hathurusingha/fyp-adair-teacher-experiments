"""Phase 13: extract the raw MGB alpha/beta scalars (not pooled feature
vectors -- just the 2 numbers per AFLB per image) for all 300 images.
Lightweight, separate pass from extract_features.py (alpha/beta weren't
part of that script's pooled-feature set, which focuses on full feature
tensors).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_alpha_beta.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

TEST02 = Path(__file__).resolve().parent.parent
REPO = TEST02.parent
sys.path.insert(0, str(REPO / "scripts"))
from instrument import Recorder, attach_instrumentation, load_adair  # noqa: E402
from run_inference import crop_img, load_rgb, add_gaussian_noise, to_tensor  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST02 / "results" / "dataset_manifest.csv"
OUT_PATH = TEST02 / "results" / "statistics" / "alpha_beta.csv"
AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]


def main():
    device = "cuda"
    np.random.seed(0)
    torch.manual_seed(0)

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    recorder = Recorder()
    attach_instrumentation(model, recorder)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for idx, row in enumerate(rows):
        image_id, deg = row["image_id"], row["degradation"]
        gt_np = crop_img(load_rgb(row["ground_truth_path"]))
        if deg == "Noise":
            sigma = float(row["noise_sigma"])
            rng = np.random.RandomState(abs(hash(image_id)) % (2 ** 31))
            degraded_np = add_gaussian_noise(gt_np, sigma, rng=rng)
        else:
            degraded_np = crop_img(load_rgb(row["input_path"]))
        degraded_t = to_tensor(degraded_np, device)

        recorder.start()
        with torch.no_grad():
            _ = model(degraded_t)
        snap = recorder.snapshot_cpu()

        for aflb in AFLB_NAMES:
            th = snap[aflb]["threshold_alpha_beta"]
            out_rows.append({
                "image_id": image_id, "degradation": deg, "AFLB": aflb,
                "alpha": th[0, 0, 0, 0].item(), "beta": th[0, 1, 0, 0].item(),
            })

        if (idx + 1) % 50 == 0 or idx == len(rows) - 1:
            print(f"[{idx + 1}/{len(rows)}] {image_id}", flush=True)

    import pandas as pd
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
