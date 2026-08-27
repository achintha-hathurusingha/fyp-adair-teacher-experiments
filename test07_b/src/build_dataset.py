"""TEST07-B: build the training/validation dataset. 80 train scenes x 8
random 128x128 crops each (Option 2 from the task spec: precomputed
multi-crop cache, chosen for tonight's run over online AdaIR-during-training
extraction), 20 val scenes x 1 FIXED deterministic crop. Rain/Haze/Noise
synthesized per crop (matching TEST05.5/TEST07-Pilot methodology).

Source: the same DIV2K validation images (100 total) already downloaded for
TEST06 (read-only reuse, TEST06/TEST07-Pilot are NOT modified). All 100
images are used again here (80 train + 20 val, fully disjoint from each
other WITHIN this experiment) -- documented reuse, not a fresh download,
to keep tonight's run tractable. Train/val split is scene-disjoint, which
is the leakage-relevant constraint for THIS experiment.

Usage:
  python build_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST07B = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07B.parent
sys.path.insert(0, str(TEST07B / "src"))
from degradation_synthesis import SYNTH_FUNCS  # noqa: E402

DIV2K_DIR = TEACHER_EXP / "test06" / "data" / "div2k_val" / "DIV2K_valid_HR"  # READ-ONLY
DATA_DIR = TEST07B / "data"
DEGS = ["Rain", "Haze", "Noise"]
CROP_SIZE = 128
N_TRAIN_SCENES = 80
N_VAL_SCENES = 20
N_CROPS_PER_TRAIN_SCENE = 8


def random_crop(img: np.ndarray, size: int, rng: np.random.RandomState):
    h, w = img.shape[:2]
    top = rng.randint(0, max(1, h - size))
    left = rng.randint(0, max(1, w - size))
    return img[top:top + size, left:left + size]


def fixed_crop(img: np.ndarray, size: int, seed: int):
    rng = np.random.RandomState(seed)
    return random_crop(img, size, rng)


def build_train(files, manifest_rows):
    for d in ["clean"] + [x.lower() for x in DEGS]:
        (DATA_DIR / "train" / d).mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(files):
        scene_id = f"train_{i:03d}"
        full_img = np.array(Image.open(path).convert("RGB"))
        crop_seed_base = abs(hash(f"{scene_id}_crops")) % (2 ** 31)
        crop_rng = np.random.RandomState(crop_seed_base)
        for c in range(N_CROPS_PER_TRAIN_SCENE):
            clean = random_crop(full_img, CROP_SIZE, crop_rng)
            crop_id = f"{scene_id}_c{c}"
            clean_path = DATA_DIR / "train" / "clean" / f"{crop_id}.png"
            Image.fromarray(clean).save(clean_path)
            row = {"scene_id": scene_id, "crop_id": crop_id, "crop_idx": c, "split": "train",
                   "source_div2k": str(path), "clean_path": str(clean_path)}
            for deg in DEGS:
                deg_seed = abs(hash(f"{crop_id}_{deg}")) % (2 ** 31)
                rng = np.random.RandomState(deg_seed)
                degraded = SYNTH_FUNCS[deg](clean, rng)
                out_path = DATA_DIR / "train" / deg.lower() / f"{crop_id}.png"
                Image.fromarray(degraded).save(out_path)
                row[f"{deg.lower()}_path"] = str(out_path)
            manifest_rows.append(row)
        print(f"[train {i+1}/{len(files)}] {scene_id} <- {path.name} ({N_CROPS_PER_TRAIN_SCENE} crops)", flush=True)


def build_val(files, manifest_rows):
    for d in ["clean"] + [x.lower() for x in DEGS]:
        (DATA_DIR / "val" / d).mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(files):
        scene_id = f"val_{i:03d}"
        full_img = np.array(Image.open(path).convert("RGB"))
        seed = abs(hash(f"{scene_id}_crop")) % (2 ** 31)
        clean = fixed_crop(full_img, CROP_SIZE, seed)
        crop_id = f"{scene_id}_c0"
        clean_path = DATA_DIR / "val" / "clean" / f"{crop_id}.png"
        Image.fromarray(clean).save(clean_path)
        row = {"scene_id": scene_id, "crop_id": crop_id, "crop_idx": 0, "split": "val",
               "source_div2k": str(path), "clean_path": str(clean_path)}
        for deg in DEGS:
            deg_seed = abs(hash(f"{crop_id}_{deg}")) % (2 ** 31)
            rng = np.random.RandomState(deg_seed)
            degraded = SYNTH_FUNCS[deg](clean, rng)
            out_path = DATA_DIR / "val" / deg.lower() / f"{crop_id}.png"
            Image.fromarray(degraded).save(out_path)
            row[f"{deg.lower()}_path"] = str(out_path)
        manifest_rows.append(row)
        print(f"[val {i+1}/{len(files)}] {scene_id} <- {path.name} (1 fixed crop)", flush=True)


def main():
    div2k_files = sorted(DIV2K_DIR.glob("*.png"))
    assert len(div2k_files) == 100, f"expected 100 DIV2K images, found {len(div2k_files)}"
    assert N_TRAIN_SCENES + N_VAL_SCENES == 100

    train_files = div2k_files[:N_TRAIN_SCENES]
    val_files = div2k_files[N_TRAIN_SCENES:N_TRAIN_SCENES + N_VAL_SCENES]

    manifest_rows = []
    build_train(train_files, manifest_rows)
    build_val(val_files, manifest_rows)

    fieldnames = list(manifest_rows[0].keys())
    manifest_path = TEST07B / "results" / "dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    n_train = sum(1 for r in manifest_rows if r["split"] == "train")
    n_val = sum(1 for r in manifest_rows if r["split"] == "val")
    print(f"\nwrote {manifest_path}: {n_train} train crops ({N_TRAIN_SCENES} scenes x "
          f"{N_CROPS_PER_TRAIN_SCENE} crops), {n_val} val crops ({N_VAL_SCENES} scenes x 1 fixed crop)")

    # explicit scene-disjointness check
    train_scenes = {r["scene_id"] for r in manifest_rows if r["split"] == "train"}
    val_scenes = {r["scene_id"] for r in manifest_rows if r["split"] == "val"}
    assert not (train_scenes & val_scenes), "train/val scene overlap detected!"
    print(f"Scene-disjointness verified: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes, 0 overlap")


if __name__ == "__main__":
    main()
