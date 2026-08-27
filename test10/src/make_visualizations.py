"""TEST10 Phase 15: 8 required visualizations.
1. A/F/G validation PSNR curves
2. A/F/G validation SSIM curves
3. Rain/Haze/Noise PSNR comparison (all 3 models)
4. G-F delta PSNR by degradation
5. Stage-wise teacher/student cosine similarity (Model G)
6. Stage-wise trajectory loss over training (Model G)
7. Haze restoration comparison (A/F/G, per seed)
8. Parameter/MAC comparison (A/F/G)

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

TEST10 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST10 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "F", "G"]
COLORS = {"A": "#1f77b4", "F": "#2ca02c", "G": "#d62728"}


def plot_metric_bands(epoch_df, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = epoch_df[epoch_df.model == model]
        piv = sub.pivot(index="epoch", columns="seed", values=metric)
        mean = piv.mean(axis=1)
        lo, hi = piv.min(axis=1), piv.max(axis=1)
        ax.plot(piv.index, mean, label=f"Model {model}", color=COLORS[model], linewidth=2)
        ax.fill_between(piv.index, lo, hi, color=COLORS[model], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    epoch_df = pd.read_csv(RESULTS_DIR / "epoch_metrics.csv")
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_deg_df = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    stage_align = pd.read_csv(STATS_DIR / "stage_alignment.csv")

    # 1-2: PSNR/SSIM curves
    plot_metric_bands(epoch_df, "val_psnr", "Validation PSNR (dB)", "Model A vs F vs G: Validation PSNR",
                       "01_val_psnr_AFG.png")
    plot_metric_bands(epoch_df, "val_ssim", "Validation SSIM", "Model A vs F vs G: Validation SSIM",
                       "02_val_ssim_AFG.png")

    # 3: per-degradation PSNR, all 3 models
    fig, ax = plt.subplots(figsize=(9, 5))
    degs = ["rain", "haze", "noise"]
    x = np.arange(len(degs))
    width = 0.25
    for i, m in enumerate(MODEL_ORDER):
        means = [seed_summary[seed_summary.model == m][f"last5_mean_{d}_psnr"].mean() for d in degs]
        stds = [seed_summary[seed_summary.model == m][f"last5_mean_{d}_psnr"].std() for d in degs]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=f"Model {m}", color=COLORS[m], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Val PSNR (dB), last5-window mean")
    ax.set_title("Per-Degradation PSNR: A vs F vs G")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_per_degradation_psnr.png", dpi=150)
    plt.close(fig)

    # 4: G-F delta PSNR by degradation, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    gf = per_deg_df[per_deg_df.comparison == "G-F"]
    degs_cap = ["rain", "haze", "noise"]
    x = np.arange(len(degs_cap))
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [gf[(gf.degradation == d) & (gf.seed == seed)].delta_psnr.iloc[0] for d in degs_cap]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs_cap])
    ax.set_ylabel("Delta PSNR (dB), G - F")
    ax.set_title("G-F Delta PSNR by Degradation, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_G_minus_F_per_degradation.png", dpi=150)
    plt.close(fig)

    # 5: stage-wise cosine similarity (Model G)
    fig, ax = plt.subplots(figsize=(7, 5))
    piv = stage_align.pivot(index="stage", columns="seed", values="cosine_similarity_mean")
    for seed in piv.columns:
        ax.plot(piv.index, piv[seed], marker="o", label=f"seed {seed}")
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Trajectory stage (0=deepest/16x16, 2=shallowest/64x64)")
    ax.set_ylabel("Cosine similarity (student vs teacher)")
    ax.set_title("Model G: Stage-wise Teacher/Student Cosine Similarity\n(NOTE: collapsed -- see report)")
    ax.set_ylim(0.99, 1.001)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_stage_cosine_similarity.png", dpi=150)
    plt.close(fig)

    # 6: stage-wise trajectory loss over training (Model G)
    fig, ax = plt.subplots(figsize=(8, 5))
    g_df = epoch_df[epoch_df.model == "G"]
    for stage, label in [("train_stage0_loss", "stage 0 (deepest)"), ("train_stage1_loss", "stage 1 (mid)"),
                          ("train_stage2_loss", "stage 2 (shallow)")]:
        piv = g_df.pivot(index="epoch", columns="seed", values=stage)
        ax.plot(piv.index, piv.mean(axis=1), label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-stage trajectory loss (normalized MSE)")
    ax.set_title("Model G: Stage-wise Trajectory Loss Over Training\n(collapses to ~0 by epoch ~10 -- representational collapse)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_stage_trajectory_loss.png", dpi=150)
    plt.close(fig)

    # 7: Haze restoration comparison, per seed
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    width = 0.25
    for i, m in enumerate(MODEL_ORDER):
        vals = seed_summary[seed_summary.model == m].sort_values("seed")["last5_mean_haze_psnr"].values
        ax.bar(x + (i - 1) * width, vals, width, label=f"Model {m}", color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["seed 0", "seed 1", "seed 2"])
    ax.set_ylabel("Haze Val PSNR (dB), last5-window mean")
    ax.set_title("Haze Restoration Comparison: A vs F vs G, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_haze_restoration_comparison.png", dpi=150)
    plt.close(fig)

    # 8: parameter/MAC comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    params = [seed_summary[seed_summary.model == m].params.iloc[0] for m in MODEL_ORDER]
    macs = [seed_summary[seed_summary.model == m].macs.iloc[0] for m in MODEL_ORDER]
    axes[0].bar(MODEL_ORDER, params, color=[COLORS[m] for m in MODEL_ORDER])
    axes[0].set_ylabel("Parameters")
    axes[0].set_title("Parameter Count")
    axes[0].ticklabel_format(style="plain", axis="y")
    axes[1].bar(MODEL_ORDER, macs, color=[COLORS[m] for m in MODEL_ORDER])
    axes[1].set_ylabel("MACs @128px")
    axes[1].set_title("MACs (inference, deployable graph)")
    axes[1].ticklabel_format(style="plain", axis="y")
    fig.suptitle("Complexity Comparison: A vs F vs G (deployable/inference graph only)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_complexity_comparison.png", dpi=150)
    plt.close(fig)

    print(f"wrote 8 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
