"""TEST03 Phase 1/2: build the 100-scene clean pool and synthesize
Rain/Haze/Noise for each, deterministically. Source: the 100 Rain100L
GROUND-TRUTH (norain-*.png) images -- natural photos not previously used
as AdaIR *inputs* in test01/test02 (only ever used there as derain
targets), read-only reference, not modified.

Usage:
  python build_scenes.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEST03 = Path(__file__).resolve().parent.parent
REPO = TEST03.parent
sys.path.insert(0, str(TEST03 / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from degradation_synthesis import SYNTHESIS_FUNCS, DEGRADATION_PARAMS  # noqa: E402
from run_inference import crop_img  # noqa: E402

RAIN100L_DIR = Path("/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L")
DATA_DIR = TEST03 / "data"
MANIFEST_DIR = TEST03 / "results" / "manifest"
VIZ_DIR = TEST03 / "results" / "visualizations" / "synthetic_examples"

N_SCENES = 100
DEGS = ["Rain", "Haze", "Noise"]


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main():
    clean_files = sorted(RAIN100L_DIR.glob("norain-*.png"))[:N_SCENES]
    assert len(clean_files) == N_SCENES, f"expected {N_SCENES} clean images, found {len(clean_files)}"

    for deg in DEGS:
        (DATA_DIR / deg.lower()).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "clean").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    param_rows = []

    for i, clean_path in enumerate(clean_files, start=1):
        scene_id = f"scene_{i:03d}"
        clean_np = crop_img(load_rgb(clean_path))
        h, w = clean_np.shape[:2]

        clean_out = DATA_DIR / "clean" / f"{scene_id}.png"
        Image.fromarray(clean_np).save(clean_out)

        row = {"scene_id": scene_id, "clean_image_path": str(clean_out),
               "source_original": str(clean_path), "height": h, "width": w}

        for deg in DEGS:
            seed = abs(hash(scene_id)) % (2 ** 31)
            rng = np.random.RandomState(seed)
            degraded_np = SYNTHESIS_FUNCS[deg](clean_np, rng)
            out_path = DATA_DIR / deg.lower() / f"{scene_id}.png"
            Image.fromarray(degraded_np).save(out_path)
            row[f"{deg.lower()}_image_path"] = str(out_path)

            param_rows.append({"scene_id": scene_id, "degradation": deg, "seed": seed,
                                **DEGRADATION_PARAMS[deg]})

        manifest_rows.append(row)
        if i % 20 == 0 or i == len(clean_files):
            print(f"[{i}/{len(clean_files)}] {scene_id} ({h}x{w})", flush=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / "scene_manifest.csv", "w", newline="") as f:
        fieldnames = ["scene_id", "clean_image_path", "rain_image_path", "haze_image_path",
                      "noise_image_path", "height", "width", "source_original"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {MANIFEST_DIR / 'scene_manifest.csv'} ({len(manifest_rows)} scenes)")

    all_param_keys = sorted({k for r in param_rows for k in r.keys()})
    with open(MANIFEST_DIR / "degradation_parameters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_param_keys)
        writer.writeheader()
        writer.writerows(param_rows)
    print(f"wrote {MANIFEST_DIR / 'degradation_parameters.csv'} ({len(param_rows)} rows)")


if __name__ == "__main__":
    main()
