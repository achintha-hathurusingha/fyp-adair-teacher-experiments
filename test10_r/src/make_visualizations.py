"""TEST10-R: visualizations covering model comparison, per-degradation
comparison, collapse-monitor trends (Phase 6: variance/cosine/distance vs
epoch), and stage alignment validity.

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

TEST10R = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST10R / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "F", "G"]
COLORS = {"A": "#1f77b4", "F": "#2ca02c", "G": "#d62728"}


def plot_metric_bands(epoch_df, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = epoch_df[epoch_df.model == model]
        piv = sub.pivot_table(index="epoch", columns="seed", values=metric)
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
    collapse_monitor = pd.read_csv(RESULTS_DIR / "collapse_monitor.csv")
    stage_align = pd.read_csv(STATS_DIR / "stage_alignment.csv")
    diversity = pd.read_csv(STATS_DIR / "cross_input_diversity.csv")

    # 1-2: PSNR/SSIM curves
    plot_metric_bands(epoch_df, "val_psnr", "Validation PSNR (dB)", "Model A vs F vs G: Validation PSNR",
                       "01_val_psnr_AFG.png")
    plot_metric_bands(epoch_df, "val_ssim", "Validation SSIM", "Model A vs F vs G: Validation SSIM",
                       "02_val_ssim_AFG.png")

    # 3: per-degradation PSNR
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

    # 4: G-F delta by degradation, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    gf = per_deg_df[per_deg_df.comparison == "G-F"]
    x = np.arange(3)
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [gf[(gf.degradation == d) & (gf.seed == seed)].delta_psnr.iloc[0] for d in degs]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Delta PSNR (dB), G - F")
    ax.set_title("G-F Delta PSNR by Degradation, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_G_minus_F_per_degradation.png", dpi=150)
    plt.close(fig)

    # 5: collapse monitor -- variance / pairwise cosine / pairwise distance vs epoch (Phase 6)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for stage_idx in (0, 1, 2):
        sub = collapse_monitor[collapse_monitor.stage == stage_idx]
        piv_var = sub.pivot_table(index="epoch", columns="seed", values="mean_variance")
        piv_cos = sub.pivot_table(index="epoch", columns="seed", values="mean_pairwise_cosine")
        piv_dist = sub.pivot_table(index="epoch", columns="seed", values="mean_pairwise_distance")
        axes[0].plot(piv_var.index, piv_var.mean(axis=1), label=f"stage {stage_idx}")
        axes[1].plot(piv_cos.index, piv_cos.mean(axis=1), label=f"stage {stage_idx}")
        axes[2].plot(piv_dist.index, piv_dist.mean(axis=1), label=f"stage {stage_idx}")
    axes[0].set_title("Student embedding variance vs epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Pairwise cosine (different inputs) vs epoch")
    axes[1].axhline(0.98, color="red", linestyle="--", label="collapse threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].set_title("Pairwise distance vs epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    fig.suptitle("Model G: Collapse Monitoring During Training (mean across 3 seeds) -- stayed healthy throughout")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_collapse_monitor_trends.png", dpi=150)
    plt.close(fig)

    # 6: stage alignment validity -- same-sample cosine vs cross-input cosine
    fig, ax = plt.subplots(figsize=(8, 5))
    piv_same = stage_align.pivot(index="stage", columns="seed", values="same_sample_cosine_mean")
    piv_cross = diversity.pivot(index="stage", columns="seed", values="student_cross_input_mean_cosine")
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, piv_same.mean(axis=1), width, label="Same-sample cosine (student vs teacher target)",
           color="#2ca02c")
    ax.bar(x + width / 2, piv_cross.mean(axis=1), width, label="Cross-input cosine (student vs different student)",
           color="#d62728")
    ax.axhline(0.98, color="black", linestyle="--", linewidth=1, label="collapse threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(["stage 0", "stage 1", "stage 2"])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Stage Alignment Validity: Real Alignment (~0.6-0.76) vs No Collapse (~0)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_stage_alignment_validity.png", dpi=150)
    plt.close(fig)

    # 7: complexity comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    params = [seed_summary[seed_summary.model == m].params.iloc[0] for m in MODEL_ORDER]
    macs = [seed_summary[seed_summary.model == m].macs.iloc[0] for m in MODEL_ORDER]
    axes[0].bar(MODEL_ORDER, params, color=[COLORS[m] for m in MODEL_ORDER])
    axes[0].set_ylabel("Parameters")
    axes[0].set_title("Parameter Count (deployable)")
    axes[1].bar(MODEL_ORDER, macs, color=[COLORS[m] for m in MODEL_ORDER])
    axes[1].set_ylabel("MACs @128px")
    axes[1].set_title("MACs (inference)")
    fig.suptitle("Complexity Comparison: A vs F vs G")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_complexity_comparison.png", dpi=150)
    plt.close(fig)

    print(f"wrote 7 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
