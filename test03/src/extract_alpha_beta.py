"""TEST03 Phase 13a: extract raw MGB alpha/beta scalars for all 300
same-scene images (lightweight separate pass, mirrors test02's approach).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_alpha_beta.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST03 = Path(__file__).resolve().parent.parent
REPO = TEST03.parent
sys.path.insert(0, str(REPO / "scripts"))
from instrument import Recorder, attach_instrumentation, load_adair  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST03 / "results" / "manifest" / "scene_manifest.csv"
OUT_PATH = TEST03 / "results" / "statistics" / "alpha_beta.csv"
AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
DEGS = ["Rain", "Haze", "Noise"]


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main():
    device = "cuda"
    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    recorder = Recorder()
    attach_instrumentation(model, recorder)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    out_rows = []
    idx = 0
    for scene_row in scene_rows:
        scene_id = scene_row["scene_id"]
        for deg in DEGS:
            degraded_np = load_rgb(scene_row[f"{deg.lower()}_image_path"])
            degraded_t = to_tensor(degraded_np, device)

            recorder.start()
            with torch.no_grad():
                _ = model(degraded_t)
            snap = recorder.snapshot_cpu()

            for aflb in AFLB_NAMES:
                th = snap[aflb]["threshold_alpha_beta"]
                out_rows.append({
                    "scene_id": scene_id, "degradation": deg, "AFLB": aflb,
                    "alpha": th[0, 0, 0, 0].item(), "beta": th[0, 1, 0, 0].item(),
                })
            idx += 1
        if idx % 60 == 0:
            print(f"[{idx}/{len(scene_rows) * 3}] {scene_id}", flush=True)

    import pandas as pd
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
