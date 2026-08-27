"""TEST07-B: 7 required training-curve/comparison visualizations.
1. A vs B L1 loss (mean +- range across seeds, per epoch)
2. A vs B val PSNR (mean +- range across seeds, per epoch)
3. A vs B val SSIM (mean +- range across seeds, per epoch)
4. B KD loss (per seed, per epoch)
5. B teacher/student cosine similarity (final-checkpoint, per seed -- bar;
   no per-epoch trace was recorded during training, KD loss curve (#4)
   shows the convergence trajectory instead)
6. Per-degradation PSNR comparison (Rain/Haze/Noise, A vs B, per seed)
7. Per-seed delta PSNR (B-A), primary last5-window metric

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

TEST07B = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST07B / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
SEEDS = [0, 1, 2]
COLORS = {"A": "#1f77b4", "B": "#d62728"}


def plot_metric_bands(epoch_df, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ["A", "B"]:
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
    align_df = pd.read_csv(STATS_DIR / "teacher_student_alignment.csv")
    per_deg_df = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    per_seed_df = pd.read_csv(STATS_DIR / "per_seed_deltas.csv")

    # 1-3: A vs B loss/PSNR/SSIM training curves (mean +- range across seeds)
    plot_metric_bands(epoch_df, "train_l1_loss", "L1 Loss", "Model A vs B: Training L1 Loss", "01_l1_loss_A_vs_B.png")
    plot_metric_bands(epoch_df, "val_psnr", "Validation PSNR (dB)", "Model A vs B: Validation PSNR", "02_val_psnr_A_vs_B.png")
    plot_metric_bands(epoch_df, "val_ssim", "Validation SSIM", "Model A vs B: Validation SSIM", "03_val_ssim_A_vs_B.png")

    # 4: B KD loss, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    b_df = epoch_df[epoch_df.model == "B"]
    for seed in SEEDS:
        sub = b_df[b_df.seed == seed]
        ax.plot(sub.epoch, sub.train_kd_loss, label=f"seed {seed}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KD Loss (MSE, e_S vs e_T)")
    ax.set_title("Model B: KD Loss Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_kd_loss_B.png", dpi=150)
    plt.close(fig)

    # 5: final-checkpoint teacher/student cosine similarity, per seed (bar)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(align_df.seed.astype(str), align_df.mean_cosine_similarity, color="#2ca02c")
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean Cosine Similarity (e_S, e_T)")
    ax.set_title("Model B: Teacher-Student Embedding Alignment\n(final checkpoint, all crops x degradations)")
    for i, v in enumerate(align_df.mean_cosine_similarity):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_teacher_student_cosine_B.png", dpi=150)
    plt.close(fig)

    # 6: per-degradation PSNR comparison, A vs B, mean across seeds
    fig, ax = plt.subplots(figsize=(8, 5))
    degs = ["rain", "haze", "noise"]
    x = np.arange(len(degs))
    width = 0.35
    a_means = [seed_summary[seed_summary.model == "A"][f"last5_mean_{d}_psnr"].mean() for d in degs]
    b_means = [seed_summary[seed_summary.model == "B"][f"last5_mean_{d}_psnr"].mean() for d in degs]
    a_stds = [seed_summary[seed_summary.model == "A"][f"last5_mean_{d}_psnr"].std() for d in degs]
    b_stds = [seed_summary[seed_summary.model == "B"][f"last5_mean_{d}_psnr"].std() for d in degs]
    ax.bar(x - width / 2, a_means, width, yerr=a_stds, label="Model A", color=COLORS["A"], capsize=4)
    ax.bar(x + width / 2, b_means, width, yerr=b_stds, label="Model B", color=COLORS["B"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Val PSNR (dB), last5-window mean")
    ax.set_title("Per-Degradation PSNR: Model A vs B (mean +- std across 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_per_degradation_psnr.png", dpi=150)
    plt.close(fig)

    # 7: per-seed delta PSNR (B - A), primary metric
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in per_seed_df.delta_last5_psnr]
    ax.bar(per_seed_df.seed.astype(str), per_seed_df.delta_last5_psnr, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Delta PSNR (B - A), dB, last5-window mean")
    ax.set_title("Per-Seed Restoration Delta (B - A)")
    for i, v in enumerate(per_seed_df.delta_last5_psnr):
        ax.text(i, v + (0.03 if v >= 0 else -0.08), f"{v:+.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_per_seed_delta_psnr.png", dpi=150)
    plt.close(fig)

    print(f"wrote 7 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
