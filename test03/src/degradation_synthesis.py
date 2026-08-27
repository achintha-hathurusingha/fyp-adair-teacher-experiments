"""TEST03 Phase 2: deterministic, documented degradation synthesis.

All three functions take a clean uint8 RGB (H,W,3) array and a per-scene
RandomState, and return a degraded uint8 RGB array of the SAME shape.
Exact method/parameters documented in test03/report/test03_design.md
section 4 -- summarized in each function's docstring.
"""
from __future__ import annotations

import cv2
import numpy as np

NOISE_SIGMA = 25.0

HAZE_A = 0.85          # atmospheric light
HAZE_BETA = 1.2         # extinction coefficient
HAZE_DEPTH_TOP = 1.0     # synthetic depth proxy at row 0 (image top) -- "far"
HAZE_DEPTH_BOTTOM = 0.3  # synthetic depth proxy at last row -- "near"

RAIN_DENSITY = 0.0006    # fraction of pixels seeded as streak origins
RAIN_LENGTH = 18         # streak length, px
RAIN_ANGLE_DEG = 70      # streak angle from horizontal
RAIN_WIDTH = 1           # streak stroke width, px
RAIN_BLUR_SIGMA = 0.6    # gaussian blur applied to the streak layer
RAIN_INTENSITY = 0.55    # scale factor applied to the streak layer before adding


def synthesize_noise(clean: np.ndarray, rng: np.random.RandomState, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """I_noise = clip(I_clean + n), n ~ N(0, sigma^2). Deterministic given rng."""
    noise = rng.randn(*clean.shape) * sigma
    return np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _synthetic_depth(h: int, w: int) -> np.ndarray:
    """Linear vertical gradient depth PROXY (top=far, bottom=near) -- NOT a
    real depth estimate. Documented simplification, see design doc."""
    col = np.linspace(HAZE_DEPTH_TOP, HAZE_DEPTH_BOTTOM, h, dtype=np.float32)
    return np.tile(col[:, None], (1, w))


def synthesize_haze(clean: np.ndarray, rng: np.random.RandomState,
                     A: float = HAZE_A, beta: float = HAZE_BETA) -> np.ndarray:
    """I_haze(x) = I_clean(x)*t(x) + A*(1-t(x)), t(x) = exp(-beta*d(x)),
    d(x) = synthetic linear-vertical-gradient depth proxy. Deterministic
    (no randomness beyond the fixed gradient) -- rng accepted for interface
    consistency with the other two synthesis functions, unused here."""
    h, w = clean.shape[:2]
    depth = _synthetic_depth(h, w)
    t = np.exp(-beta * depth)[:, :, None]  # (H,W,1)
    clean_f = clean.astype(np.float32) / 255.0
    hazy_f = clean_f * t + A * (1 - t)
    return np.clip(hazy_f * 255.0, 0, 255).astype(np.uint8)


def synthesize_rain(clean: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Deterministic synthetic rain-streak layer: sparse random streak
    origins (density RAIN_DENSITY) rendered as fixed-length/angle/width line
    segments via cv2.line, Gaussian-blurred, scaled, and added. Only streak
    POSITIONS vary with rng; all other parameters are fixed constants."""
    h, w = clean.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)

    n_streaks = max(1, int(RAIN_DENSITY * h * w))
    xs = rng.randint(0, w, size=n_streaks)
    ys = rng.randint(0, h, size=n_streaks)

    angle_rad = np.deg2rad(RAIN_ANGLE_DEG)
    dx = int(round(RAIN_LENGTH * np.cos(angle_rad)))
    dy = int(round(RAIN_LENGTH * np.sin(angle_rad)))

    for x, y in zip(xs, ys):
        x2, y2 = x + dx, y + dy
        cv2.line(layer, (int(x), int(y)), (int(x2), int(y2)), color=1.0, thickness=RAIN_WIDTH)

    layer = cv2.GaussianBlur(layer, ksize=(0, 0), sigmaX=RAIN_BLUR_SIGMA)
    layer = layer * RAIN_INTENSITY * 255.0

    rained = clean.astype(np.float32) + layer[:, :, None]
    return np.clip(rained, 0, 255).astype(np.uint8)


SYNTHESIS_FUNCS = {"Rain": synthesize_rain, "Haze": synthesize_haze, "Noise": synthesize_noise}

DEGRADATION_PARAMS = {
    "Rain": {"density": RAIN_DENSITY, "length_px": RAIN_LENGTH, "angle_deg": RAIN_ANGLE_DEG,
             "width_px": RAIN_WIDTH, "blur_sigma": RAIN_BLUR_SIGMA, "intensity": RAIN_INTENSITY},
    "Haze": {"A": HAZE_A, "beta": HAZE_BETA, "depth_top": HAZE_DEPTH_TOP, "depth_bottom": HAZE_DEPTH_BOTTOM,
             "depth_model": "linear_vertical_gradient_proxy"},
    "Noise": {"sigma": NOISE_SIGMA, "distribution": "gaussian"},
}
