"""TEST05.5 Phase 3-4: parameter-RANDOMIZED degradation synthesis, to test
whether the degradation-representation trajectory is recognizing the
degradation FAMILY or just TEST03's one fixed parameter set per
degradation (loophole L3).

Does NOT replace TEST03 -- writes to test05_5/data/, a separate dataset.
Uses the SAME 100 clean scenes (test03/data/clean/, read-only) but
synthesizes rain/haze/noise with parameters drawn per-image from
documented ranges (not the single fixed TEST03 values), split into two
non-overlapping SEVERITY BANDS (A=weaker average, B=stronger average,
with intentional overlap in the tails so classes are not trivially
separable by magnitude alone -- per the task's explicit instruction).

RAIN: angle in [50,90]deg, length in [10,26]px, width in {1,2}, density
      in [0.0003,0.0010], intensity in [0.35,0.75], blur sigma in [0.3,0.9]
HAZE: beta in [0.6,1.8], A(atmospheric light) in [0.75,0.95],
      depth-gradient strength (top-bottom contrast) in [0.4,0.9]
NOISE: sigma in [12,40]

Band A = lower half of each range (weaker), Band B = upper half
(stronger), each band spans its half PLUS a small overlap margin into the
other half so some "weak-of-B" and "strong-of-A" images overlap in
severity -- avoiding a trivial magnitude-based shortcut.

Usage:
  python robustness_synthesis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from run_inference import crop_img  # noqa: E402 (read-only shared infra)

CLEAN_DIR = TEACHER_EXP / "test03" / "data" / "clean"
DATA_DIR = TEST05_5 / "data"
MANIFEST_DIR = TEST05_5 / "results" / "robustness"
DEGS = ["Rain", "Haze", "Noise"]
N_SCENES = 100

# (low, high) full range; band A = [low, mid+overlap], band B = [mid-overlap, high]
RANGES = {
    "rain": {"angle_deg": (50, 90), "length_px": (10, 26), "width_px": (1, 2),
             "density": (0.0003, 0.0010), "intensity": (0.35, 0.75), "blur_sigma": (0.3, 0.9)},
    "haze": {"beta": (0.6, 1.8), "A": (0.75, 0.95), "depth_strength": (0.4, 0.9)},
    "noise": {"sigma": (12, 40)},
}
OVERLAP_FRAC = 0.15  # fraction of half-range used as deliberate overlap


def band_range(low, high, band, overlap_frac=OVERLAP_FRAC):
    mid = (low + high) / 2
    span = (high - low) / 2
    overlap = span * overlap_frac
    if band == "A":
        return low, mid + overlap
    return mid - overlap, high


def sample(rng, low, high, is_int=False):
    v = rng.uniform(low, high)
    return int(round(v)) if is_int else v


def synth_rain(clean, rng, params):
    h, w = clean.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)
    n_streaks = max(1, int(params["density"] * h * w))
    xs = rng.randint(0, w, size=n_streaks)
    ys = rng.randint(0, h, size=n_streaks)
    angle_rad = np.deg2rad(params["angle_deg"])
    dx = int(round(params["length_px"] * np.cos(angle_rad)))
    dy = int(round(params["length_px"] * np.sin(angle_rad)))
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), color=1.0,
                 thickness=int(params["width_px"]))
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=params["blur_sigma"])
    layer = layer * params["intensity"] * 255.0
    return np.clip(clean.astype(np.float32) + layer[:, :, None], 0, 255).astype(np.uint8)


def synth_haze(clean, params):
    h, w = clean.shape[:2]
    strength = params["depth_strength"]
    top, bottom = 0.5 + strength / 2, 0.5 - strength / 2
    depth = np.tile(np.linspace(top, bottom, h, dtype=np.float32)[:, None], (1, w))
    t = np.exp(-params["beta"] * depth)[:, :, None]
    clean_f = clean.astype(np.float32) / 255.0
    hazy = clean_f * t + params["A"] * (1 - t)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def synth_noise(clean, rng, params):
    noise = rng.randn(*clean.shape) * params["sigma"]
    return np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main():
    clean_files = sorted(CLEAN_DIR.glob("scene_*.png"))[:N_SCENES]
    assert len(clean_files) == N_SCENES

    for deg in DEGS:
        (DATA_DIR / deg.lower()).mkdir(parents=True, exist_ok=True)

    manifest_rows, param_rows = [], []
    for i, clean_path in enumerate(clean_files, start=1):
        scene_id = clean_path.stem
        clean_np = crop_img(np.array(Image.open(clean_path).convert("RGB")))

        row = {"scene_id": scene_id, "clean_image_path": str(clean_path)}
        for deg in DEGS:
            for band in ["A", "B"]:
                seed = abs(hash(f"{scene_id}_{deg}_{band}")) % (2 ** 31)
                rng = np.random.RandomState(seed)

                if deg == "Rain":
                    p = {}
                    for k, (lo, hi) in RANGES["rain"].items():
                        blo, bhi = band_range(lo, hi, band)
                        p[k] = sample(rng, blo, bhi, is_int=(k in ("width_px",)))
                    degraded = synth_rain(clean_np, rng, p)
                elif deg == "Haze":
                    p = {}
                    for k, (lo, hi) in RANGES["haze"].items():
                        blo, bhi = band_range(lo, hi, band)
                        p[k] = sample(rng, blo, bhi)
                    degraded = synth_haze(clean_np, p)
                else:
                    lo, hi = RANGES["noise"]["sigma"]
                    blo, bhi = band_range(lo, hi, band)
                    p = {"sigma": sample(rng, blo, bhi)}
                    degraded = synth_noise(clean_np, rng, p)

                out_path = DATA_DIR / deg.lower() / f"{scene_id}_band{band}.png"
                Image.fromarray(degraded).save(out_path)
                row[f"{deg.lower()}_band{band}_path"] = str(out_path)
                param_rows.append({"scene_id": scene_id, "degradation": deg, "band": band,
                                    "seed": seed, **p})

        manifest_rows.append(row)
        if i % 20 == 0 or i == len(clean_files):
            print(f"[{i}/{len(clean_files)}] {scene_id}", flush=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "clean_image_path"] + [f"{d.lower()}_band{b}_path" for d in DEGS for b in "AB"]
    with open(MANIFEST_DIR / "robustness_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {MANIFEST_DIR / 'robustness_manifest.csv'} ({len(manifest_rows)} scenes)")

    all_keys = sorted({k for r in param_rows for k in r.keys()})
    with open(MANIFEST_DIR / "robustness_parameters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(param_rows)
    print(f"wrote {MANIFEST_DIR / 'robustness_parameters.csv'} ({len(param_rows)} rows)")


if __name__ == "__main__":
    main()
