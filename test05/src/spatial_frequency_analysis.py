"""TEST05 Phase 11-13: spatial + frequency-domain analysis of the raw
(unpooled) candidate tensors, for the 15 representative scenes' raw .pt
files saved by extract_features.py. Independent of AdaIR's own (degenerate)
FFT mask -- this analyzes the actual feature tensors' spatial Fourier
spectra, per the task's explicit instruction.

Phase 11 (spatial): per-pixel importance map = variance across the 3
degradations, for the SAME scene, channel-averaged -- shows whether
degradation-discriminative signal is spatially global, localized, or
edge/texture-focused.

Phase 12 (frequency): 2D FFT of each channel-averaged feature map, radial
spectral profile binned into low/mid/high energy, compared across
Rain/Haze/Noise for the same scene.

Phase 13 (frequency-sensitive channel ranking): combine per-channel
degradation_scene_ratio (from channel_rank.csv) with each channel's own
2D FFT radial profile (computed on one representative scene, averaged over
the 15-scene sample) to see whether the most degradation-specific channels
are also spectrally distinctive.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python spatial_frequency_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TEST05 = Path(__file__).resolve().parent.parent
TENSORS_DIR = TEST05 / "results" / "tensors"
SPATIAL_DIR = TEST05 / "results" / "spatial_analysis"
FREQ_DIR = TEST05 / "results" / "frequency_analysis"
VIZ_DIR = TEST05 / "results" / "visualizations"
DEGS = ["rain", "haze", "noise"]

CANDIDATES = ["latent_pre", "AFLB1_aflb_out", "AFLB1_mined_low", "AFLB1_mined_high", "AFLB3_aflb_out"]


def load_tensor(feature, deg, scene_id):
    p = TENSORS_DIR / feature / deg / f"{scene_id}.pt"
    return torch.load(p, weights_only=False).float() if p.exists() else None


def radial_profile(mag2d: np.ndarray, n_bins=3):
    h, w = mag2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / r.max()
    bins = np.linspace(0, 1, n_bins + 1)
    energies = []
    for i in range(n_bins):
        mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        energies.append(float(mag2d[mask].sum()))
    total = sum(energies) + 1e-12
    return [e / total * 100 for e in energies]  # low, mid, high %


def main():
    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    FREQ_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    scene_ids = sorted({p.stem for p in (TENSORS_DIR / "latent_pre" / "rain").glob("*.pt")})
    print(f"{len(scene_ids)} representative scenes available", flush=True)

    spatial_rows, freq_rows = [], []

    for feature in CANDIDATES:
        for scene_id in scene_ids:
            tensors = {}
            for deg in DEGS:
                t = load_tensor(feature, deg, scene_id)
                if t is None:
                    break
                tensors[deg] = t[0].mean(0).numpy()  # channel-averaged (H,W)
            if len(tensors) != 3:
                continue

            # ---- Phase 11: spatial importance = variance across degradations ----
            stacked = np.stack([tensors[d] for d in DEGS])
            importance = stacked.var(axis=0)
            spatial_rows.append({
                "feature": feature, "scene_id": scene_id,
                "importance_mean": float(importance.mean()), "importance_std": float(importance.std()),
                "importance_max": float(importance.max()),
                "importance_center_frac": float(importance[importance.shape[0]//4:3*importance.shape[0]//4,
                                                             importance.shape[1]//4:3*importance.shape[1]//4].mean()
                                                 / (importance.mean() + 1e-12)),
            })

            # ---- Phase 12: 2D FFT radial profile per degradation ----
            for deg in DEGS:
                fft = np.fft.fftshift(np.fft.fft2(tensors[deg]))
                mag2 = np.abs(fft) ** 2
                low, mid, high = radial_profile(mag2)
                freq_rows.append({"feature": feature, "scene_id": scene_id, "degradation": deg,
                                   "low_freq_pct": low, "mid_freq_pct": mid, "high_freq_pct": high})

    spatial_df = pd.DataFrame(spatial_rows)
    freq_df = pd.DataFrame(freq_rows)
    spatial_df.to_csv(SPATIAL_DIR / "spatial_importance.csv", index=False)
    freq_df.to_csv(FREQ_DIR / "frequency_profile.csv", index=False)
    print(f"wrote {SPATIAL_DIR / 'spatial_importance.csv'} ({len(spatial_df)} rows)")
    print(f"wrote {FREQ_DIR / 'frequency_profile.csv'} ({len(freq_df)} rows)")

    print("\nSpatial importance summary (is degradation-signal centered or spread out?):")
    print(spatial_df.groupby("feature")["importance_center_frac"].mean().to_string())
    print("(center_frac ~1.0 = uniform; >1.3 = center-concentrated; <0.7 = edge/border-concentrated)")

    print("\nFrequency profile summary (mean % energy per band, by feature x degradation):")
    freq_summary = freq_df.groupby(["feature", "degradation"])[["low_freq_pct", "mid_freq_pct", "high_freq_pct"]].mean()
    print(freq_summary.to_string())
    freq_summary.reset_index().to_csv(FREQ_DIR / "frequency_summary.csv", index=False)

    # ---- visualization: frequency spectrum bars per degradation, for latent_pre ----
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = freq_summary.loc["latent_pre"] if "latent_pre" in freq_summary.index.get_level_values(0) else None
    if sub is not None:
        x = np.arange(3)
        width = 0.25
        for i, deg in enumerate(DEGS):
            if deg in sub.index:
                vals = sub.loc[deg].values
                ax.bar(x + i * width, vals, width, label=deg.capitalize())
        ax.set_xticks(x + width)
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_ylabel("% spectral energy")
        ax.set_title("latent_pre: spatial-frequency energy by degradation (same scenes averaged)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(VIZ_DIR / "frequency_spectrum_by_degradation.png", dpi=130)
        plt.close(fig)
        print(f"\nwrote {VIZ_DIR / 'frequency_spectrum_by_degradation.png'}")

    # ---- Phase 13: does high-ranked channel = spectrally distinctive channel? ----
    chan_rank_path = TEST05 / "results" / "channel_analysis" / "channel_rank.csv"
    if chan_rank_path.exists():
        chan_rank = pd.read_csv(chan_rank_path)
        rows = []
        for feature in CANDIDATES:
            fr = chan_rank[chan_rank.feature == feature].sort_values("degradation_probe_accuracy", ascending=False)
            top_n = min(10, len(fr))
            top_channels = fr.head(top_n)["channel"].tolist()
            for scene_id in scene_ids[:5]:
                per_deg = {}
                for deg in DEGS:
                    t = load_tensor(feature, deg, scene_id)
                    if t is None:
                        continue
                    per_deg[deg] = t[0]
                if len(per_deg) != 3:
                    continue
                for ch in top_channels:
                    profiles = {}
                    for deg in DEGS:
                        chan_map = per_deg[deg][ch].numpy()
                        fft = np.fft.fftshift(np.fft.fft2(chan_map))
                        mag2 = np.abs(fft) ** 2
                        profiles[deg] = radial_profile(mag2)
                    # spectral distinctiveness = variance of high-freq % across degradations
                    highs = [profiles[d][2] for d in DEGS]
                    rows.append({"feature": feature, "scene_id": scene_id, "channel": ch,
                                 "high_freq_pct_variance_across_deg": float(np.var(highs)),
                                 "rain_high_pct": profiles["rain"][2], "haze_high_pct": profiles["haze"][2],
                                 "noise_high_pct": profiles["noise"][2]})
        fcr = pd.DataFrame(rows)
        fcr.to_csv(FREQ_DIR / "frequency_channel_ranking.csv", index=False)
        print(f"\nwrote {FREQ_DIR / 'frequency_channel_ranking.csv'} ({len(fcr)} rows)")
        if len(fcr):
            print(fcr.groupby("feature")["high_freq_pct_variance_across_deg"].mean().to_string())


if __name__ == "__main__":
    main()
