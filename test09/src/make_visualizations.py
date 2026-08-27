"""TEST09: visualizations for model comparison, per-degradation comparison,
and modulation analysis (per the report's required content areas).

1. A/B/C/D/E/F overall validation PSNR (bar, mean+-std across seeds)
2. Per-degradation PSNR: Rain/Haze/Noise, all 6 models
3. Delta PSNR: D-C, E-C, F-C, C-B, B-A (per-seed bars)
4. Modulation magnitude heatmap: model x stage x degradation

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python make_visualizations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST09 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST09 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "B", "C", "D", "E", "F"]
COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd", "E": "#8c564b", "F": "#e377c2"}


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_seed_df = pd.read_csv(STATS_DIR / "per_seed_deltas.csv")
    mod_df = pd.read_csv(STATS_DIR / "modulation_magnitude.csv")

    # 1: overall PSNR, all 6 models
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [seed_summary[seed_summary.model == m].last5_mean_psnr.mean() for m in MODEL_ORDER]
    stds = [seed_summary[seed_summary.model == m].last5_mean_psnr.std() for m in MODEL_ORDER]
    ax.bar(MODEL_ORDER, means, yerr=stds, capsize=4, color=[COLORS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Val PSNR (dB), last5-window mean")
    ax.set_title("Model Comparison: A/B/C/D/E/F Overall Restoration")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_model_comparison_psnr.png", dpi=150)
    plt.close(fig)

    # 2: per-degradation PSNR, all 6 models
    fig, ax = plt.subplots(figsize=(11, 5))
    degs = ["rain", "haze", "noise"]
    x = np.arange(len(degs))
    width = 0.13
    for i, m in enumerate(MODEL_ORDER):
        means = [seed_summary[seed_summary.model == m][f"last5_mean_{d}_psnr"].mean() for d in degs]
        ax.bar(x + (i - 2.5) * width, means, width, label=f"Model {m}", color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Val PSNR (dB), last5-window mean")
    ax.set_title("Per-Degradation PSNR: All Models")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_per_degradation_psnr_all.png", dpi=150)
    plt.close(fig)

    # 3: delta PSNR comparisons, per seed
    fig, ax = plt.subplots(figsize=(10, 5))
    comparisons = ["D-C", "E-C", "F-C", "C-B", "B-A"]
    xw = np.arange(len(comparisons))
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [per_seed_df[(per_seed_df.comparison == c) & (per_seed_df.seed == seed)].delta_last5_psnr.iloc[0]
                for c in comparisons]
        ax.bar(xw + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xw)
    ax.set_xticklabels(comparisons)
    ax.set_ylabel("Delta PSNR (dB), last5-window mean")
    ax.set_title("Pairwise Restoration Deltas, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_delta_psnr_comparisons.png", dpi=150)
    plt.close(fig)

    # 4: modulation magnitude heatmap
    mod_df["model_stage"] = mod_df.model + " / " + mod_df.stage
    piv = mod_df.groupby(["model_stage", "degradation"])["relative_modulation_magnitude"].mean().unstack()
    piv = piv[["Rain", "Haze", "Noise"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Modulation Magnitude by Model/Stage/Degradation")
    fig.colorbar(im, ax=ax, label="relative modulation magnitude")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_modulation_magnitude_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"wrote 4 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
