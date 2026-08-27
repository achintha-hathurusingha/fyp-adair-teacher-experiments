"""Build the 300-image manifest (100 derain / 100 dehaze / 100 denoise).

Deviations from the literal "100 unique images" wording, and why:

* Derain (Rain100L test split) has exactly 100 paired images -> used as-is.
* Dehaze (SOTS-outdoor) has 492 unique clean scenes, each with one or more
  hazy renders. We de-duplicate by clean-image prefix (keep the first hazy
  render per scene) and take the first 100 sorted scenes -> 100 unique pairs.
* Denoise (BSD68) has only 68 unique clean images -- fewer than 100. AdaIR's
  own protocol does not add real noisy images; it synthesises Gaussian noise
  at sigma in {15, 25, 50} on the fly. To reach 100 instances while staying
  inside that protocol, we take all 68 images at sigma=25 (the canonical
  single-number comparison level) plus 16 extra images at sigma=15 and 16
  more at sigma=50 (deterministic split by sorted filename), giving
  68 + 16 + 16 = 100 (image, sigma) instances. Noise_Sigma is recorded as its
  own column so this is fully transparent in the data, not hidden.

Output: manifest.csv with columns
  Image_ID, Degradation, Dataset, input_path, gt_path, noise_sigma
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # teacher-experiments/
DATA_ROOT_A = Path("/home/minura/fyp-adair-distill/data")
DATA_ROOT_B = Path("/home/minura/FYP/Workspace/Himeth/data")

RAIN100L_DIR = DATA_ROOT_B / "rain100L" / "rain100L_test" / "Rain100L"
SOTS_DIR = DATA_ROOT_A / "dehaze" / "RESIDE" / "SOTS" / "outdoor"
BSD68_DIR = DATA_ROOT_A / "test" / "denoise" / "bsd68"

SEED = 0


def build_derain(n: int = 100) -> list[dict]:
    gt_files = sorted((RAIN100L_DIR).glob("norain-*.png"))
    rows = []
    for i, gt_path in enumerate(gt_files[:n], start=1):
        num = gt_path.stem.split("-")[1]
        rain_path = RAIN100L_DIR / "rainy" / f"rain-{num}.png"
        assert rain_path.exists(), rain_path
        rows.append({
            "Image_ID": f"R{i:03d}",
            "Degradation": "Rain",
            "Dataset": "Rain100L",
            "input_path": str(rain_path),
            "gt_path": str(gt_path),
            "noise_sigma": "",
        })
    assert len(rows) == n, f"derain: expected {n}, got {len(rows)}"
    return rows


def build_dehaze(n: int = 100) -> list[dict]:
    input_files = sorted((SOTS_DIR / "input").iterdir())
    seen_prefix: dict[str, Path] = {}
    for f in input_files:
        prefix = f.name.split("_")[0]
        if prefix not in seen_prefix:
            seen_prefix[prefix] = f
    prefixes = sorted(seen_prefix.keys())[:n]
    rows = []
    for i, prefix in enumerate(prefixes, start=1):
        input_path = seen_prefix[prefix]
        gt_path = SOTS_DIR / "target" / f"{prefix}.png"
        assert gt_path.exists(), gt_path
        rows.append({
            "Image_ID": f"H{i:03d}",
            "Degradation": "Haze",
            "Dataset": "SOTS-outdoor",
            "input_path": str(input_path),
            "gt_path": str(gt_path),
            "noise_sigma": "",
        })
    assert len(rows) == n, f"dehaze: expected {n}, got {len(rows)}"
    return rows


def build_denoise(n: int = 100) -> list[dict]:
    clean_files = sorted(BSD68_DIR.iterdir())
    assert len(clean_files) == 68, f"expected 68 BSD68 images, got {len(clean_files)}"

    instances: list[tuple[Path, int]] = [(f, 25) for f in clean_files]  # baseline: all @ sigma25
    instances += [(f, 15) for f in clean_files[:16]]
    instances += [(f, 50) for f in clean_files[16:32]]
    assert len(instances) == n, f"denoise: expected {n}, got {len(instances)}"

    rows = []
    for i, (clean_path, sigma) in enumerate(instances, start=1):
        rows.append({
            "Image_ID": f"N{i:03d}",
            "Degradation": "Noise",
            "Dataset": "BSD68",
            "input_path": str(clean_path),  # noise synthesised at inference time
            "gt_path": str(clean_path),
            "noise_sigma": sigma,
        })
    return rows


def main() -> None:
    random.seed(SEED)
    rows = build_derain(100) + build_dehaze(100) + build_denoise(100)
    assert len(rows) == 300

    out_path = REPO / "manifest.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image_ID", "Degradation", "Dataset",
                                                 "input_path", "gt_path", "noise_sigma"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")
    for deg in ("Rain", "Haze", "Noise"):
        c = sum(1 for r in rows if r["Degradation"] == deg)
        print(f"  {deg}: {c}")


if __name__ == "__main__":
    main()
