"""Phase 2: re-export the SAME 300-image manifest used throughout this
project (../manifest.csv, unchanged) into TEST02's required schema. Does not
regenerate or resample -- identical images, identical selection logic as
test01/the original 300-image analysis.

Usage (local or remote, no torch needed):
  python build_manifest.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

TEST02 = Path(__file__).resolve().parent.parent
REPO = TEST02.parent
SRC_MANIFEST = REPO / "manifest.csv"
OUT_MANIFEST = TEST02 / "results" / "dataset_manifest.csv"

LABEL_MAP = {"Rain": 0, "Haze": 1, "Noise": 2}


def crop_to_multiple(h: int, w: int, base: int = 16) -> tuple[int, int]:
    return h - (h % base), w - (w % base)


def main():
    with open(SRC_MANIFEST) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for i, r in enumerate(rows, start=1):
        gt_path = r["gt_path"]
        with Image.open(gt_path) as im:
            w, h = im.size
        crop_h, crop_w = crop_to_multiple(h, w)
        out_rows.append({
            "image_id": r["Image_ID"],
            "filename": Path(r["input_path"]).name,
            "degradation": r["Degradation"],
            "degradation_label": LABEL_MAP[r["Degradation"]],
            "input_path": r["input_path"],
            "ground_truth_path": gt_path,
            "input_height": crop_h,
            "input_width": crop_w,
            "noise_sigma": r["noise_sigma"],
        })

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {OUT_MANIFEST} ({len(out_rows)} rows)")
    for deg, label in LABEL_MAP.items():
        c = sum(1 for r in out_rows if r["degradation"] == deg)
        print(f"  {deg} (label={label}): {c}")


if __name__ == "__main__":
    main()
