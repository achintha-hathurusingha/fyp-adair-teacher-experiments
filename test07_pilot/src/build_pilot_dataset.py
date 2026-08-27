"""TEST07-Pilot: build a small, manageable train/val set. Reuses the
already-downloaded DIV2K validation images at test06/data/div2k_val
(READ-ONLY reference -- test06/ is not modified, only its data files are
read). Images 40-79 (40 scenes) for training, 90-99 (10 scenes) for
validation -- disjoint from test06's own usage (0-7 resolution sweep,
8-32 same-scene 06-E set), avoiding any cross-experiment data leakage.

Fixed 128x128 crop per scene (deterministic seed), matching AdaIR's own
training patch size. The crop is fixed (not re-randomized per epoch) so
that precomputed teacher embeddings remain valid for the ENTIRE pilot run
-- documented simplification, appropriate for a short pilot, not for a
final training run.

Usage:
  python build_pilot_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST07 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07.parent
sys.path.insert(0, str(TEST07 / "src"))
from degradation_synthesis import SYNTH_FUNCS  # noqa: E402

DIV2K_DIR = TEACHER_EXP / "test06" / "data" / "div2k_val" / "DIV2K_valid_HR"  # READ-ONLY
DATA_DIR = TEST07 / "data"
DEGS = ["Rain", "Haze", "Noise"]
CROP_SIZE = 128
TRAIN_RANGE = (40, 80)  # DIV2K indices 40-79 -> 40 training scenes
VAL_RANGE = (90, 100)   # DIV2K indices 90-99 -> 10 validation scenes


def fixed_crop(img: np.ndarray, size: int, seed: int):
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    top = rng.randint(0, max(1, h - size))
    left = rng.randint(0, max(1, w - size))
    return img[top:top + size, left:left + size]


def build_split(files, split_name, manifest_rows):
    for d in ["clean"] + [x.lower() for x in DEGS]:
        (DATA_DIR / split_name / d).mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(files):
        scene_id = f"{split_name}_{i:03d}"
        full_img = np.array(Image.open(path).convert("RGB"))
        seed = abs(hash(f"{scene_id}_crop")) % (2 ** 31)
        clean = fixed_crop(full_img, CROP_SIZE, seed)
        clean_path = DATA_DIR / split_name / "clean" / f"{scene_id}.png"
        Image.fromarray(clean).save(clean_path)
        row = {"scene_id": scene_id, "split": split_name, "source_div2k": str(path), "clean_path": str(clean_path)}
        for deg in DEGS:
            deg_seed = abs(hash(f"{scene_id}_{deg}")) % (2 ** 31)
            rng = np.random.RandomState(deg_seed)
            degraded = SYNTH_FUNCS[deg](clean, rng)
            out_path = DATA_DIR / split_name / deg.lower() / f"{scene_id}.png"
            Image.fromarray(degraded).save(out_path)
            row[f"{deg.lower()}_path"] = str(out_path)
        manifest_rows.append(row)
        print(f"[{split_name} {i+1}/{len(files)}] {scene_id} <- {path.name}", flush=True)


def main():
    div2k_files = sorted(DIV2K_DIR.glob("*.png"))
    assert len(div2k_files) == 100, f"expected 100 DIV2K images, found {len(div2k_files)}"

    train_files = div2k_files[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
    val_files = div2k_files[VAL_RANGE[0]:VAL_RANGE[1]]

    manifest_rows = []
    build_split(train_files, "train", manifest_rows)
    build_split(val_files, "val", manifest_rows)

    fieldnames = list(manifest_rows[0].keys())
    manifest_path = TEST07 / "results" / "pilot_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nwrote {manifest_path} ({len(manifest_rows)} scenes: "
          f"{len(train_files)} train, {len(val_files)} val)")


if __name__ == "__main__":
    main()
