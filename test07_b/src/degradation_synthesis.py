"""TEST07-B: parameter-randomized Rain/Haze/Noise synthesis, same
methodology as TEST05.5/TEST07-Pilot (fresh, self-contained implementation
per the project's directory-isolation convention)."""
from __future__ import annotations

import numpy as np
import cv2

RANGES = {
    "rain": {"angle_deg": (50, 90), "length_px": (10, 26), "width_px": (1, 2),
             "density": (0.0003, 0.0010), "intensity": (0.35, 0.75), "blur_sigma": (0.3, 0.9)},
    "haze": {"beta": (0.6, 1.8), "A": (0.75, 0.95), "depth_strength": (0.4, 0.9)},
    "noise": {"sigma": (12, 40)},
}


def sample(rng, low, high, is_int=False):
    v = rng.uniform(low, high)
    return int(round(v)) if is_int else v


def synth_rain(clean, rng):
    h, w = clean.shape[:2]
    p = {k: sample(rng, lo, hi, is_int=(k == "width_px")) for k, (lo, hi) in RANGES["rain"].items()}
    layer = np.zeros((h, w), dtype=np.float32)
    n_streaks = max(1, int(p["density"] * h * w))
    xs = rng.randint(0, w, size=n_streaks)
    ys = rng.randint(0, h, size=n_streaks)
    angle_rad = np.deg2rad(p["angle_deg"])
    dx = int(round(p["length_px"] * np.cos(angle_rad)))
    dy = int(round(p["length_px"] * np.sin(angle_rad)))
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)), color=1.0, thickness=int(p["width_px"]))
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=p["blur_sigma"])
    layer = layer * p["intensity"] * 255.0
    return np.clip(clean.astype(np.float32) + layer[:, :, None], 0, 255).astype(np.uint8)


def synth_haze(clean, rng):
    h, w = clean.shape[:2]
    p = {k: sample(rng, lo, hi) for k, (lo, hi) in RANGES["haze"].items()}
    strength = p["depth_strength"]
    top, bottom = 0.5 + strength / 2, 0.5 - strength / 2
    depth = np.tile(np.linspace(top, bottom, h, dtype=np.float32)[:, None], (1, w))
    t = np.exp(-p["beta"] * depth)[:, :, None]
    clean_f = clean.astype(np.float32) / 255.0
    hazy = clean_f * t + p["A"] * (1 - t)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def synth_noise(clean, rng):
    p = {"sigma": sample(rng, *RANGES["noise"]["sigma"])}
    noise = rng.randn(*clean.shape) * p["sigma"]
    return np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)


SYNTH_FUNCS = {"Rain": synth_rain, "Haze": synth_haze, "Noise": synth_noise}
