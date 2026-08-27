"""TEST06 Phase 5: build the SAME-SCENE Rain/Haze/Noise dataset for 06-E,
at a resolution comfortably above R_first=768px (AFLB3's confirmed
activation threshold from 06-A). Uses 25 NATIVE high-resolution DIV2K
images (indices 8-32, disjoint from the 0-7 used in the resolution sweep,
to avoid any data leakage between 06-A and 06-E), documented provenance
(official ETH Zurich CVL DIV2K validation set).

Resolution choice: 1024x1024 square center crop. Native DIV2K images are
2040x1356 -- no upscaling required (per the task's explicit preference for
native-resolution evidence over upscaled Rain100L). 1024x1024 (1.05M px)
is within the resolution sweep's confirmed-safe range (no OOM observed at
comparable areas).

Usage:
  python build_06e_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST06 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST06 / "src"))
from degradation_synthesis import SYNTH_FUNCS  # noqa: E402

DIV2K_DIR = TEST06 / "data" / "div2k_val" / "DIV2K_valid_HR"
DATA_DIR = TEST06 / "data" / "same_scene_06e"
MANIFEST_DIR = TEST06 / "results" / "frequency_intervention"
DEGS = ["Rain", "Haze", "Noise"]
N_SCENES = 25
CROP_SIZE = 1024
FIRST_IMG_IDX = 8  # disjoint from the 0-7 used in 06-A's resolution sweep


def center_crop(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def main():
    for deg in DEGS:
        (DATA_DIR / deg.lower()).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "clean").mkdir(parents=True, exist_ok=True)

    div2k_files = sorted(DIV2K_DIR.glob("*.png"))[FIRST_IMG_IDX:FIRST_IMG_IDX + N_SCENES]
    assert len(div2k_files) == N_SCENES, f"expected {N_SCENES} DIV2K images, found {len(div2k_files)}"

    manifest_rows = []
    for i, path in enumerate(div2k_files):
        scene_id = f"scene_{i:03d}"
        full_img = np.array(Image.open(path).convert("RGB"))
        clean = center_crop(full_img, CROP_SIZE)
        clean_path = DATA_DIR / "clean" / f"{scene_id}.png"
        Image.fromarray(clean).save(clean_path)

        row = {"scene_id": scene_id, "source_div2k": str(path), "clean_path": str(clean_path),
               "resolution": CROP_SIZE}
        for deg in DEGS:
            seed = abs(hash(f"{scene_id}_{deg}")) % (2 ** 31)
            rng = np.random.RandomState(seed)
            degraded = SYNTH_FUNCS[deg](clean, rng)
            out_path = DATA_DIR / deg.lower() / f"{scene_id}.png"
            Image.fromarray(degraded).save(out_path)
            row[f"{deg.lower()}_path"] = str(out_path)
            row[f"{deg.lower()}_seed"] = seed
        manifest_rows.append(row)
        print(f"[{i+1}/{N_SCENES}] {scene_id} <- {path.name}", flush=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0].keys())
    with open(MANIFEST_DIR / "scene_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote {MANIFEST_DIR / 'scene_manifest.csv'} ({len(manifest_rows)} scenes)")


if __name__ == "__main__":
    main()
