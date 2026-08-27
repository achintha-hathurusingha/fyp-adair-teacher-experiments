"""TEST12: visualizations covering restoration comparison, per-degradation
breakdown, the causal-control comparison (the key result), and
scene-variance / effective-rank diagnostics.

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

TEST12 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST12 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "F2", "T12"]
COLORS = {"A": "#1f77b4", "F2": "#2ca02c", "T12": "#d62728"}


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    epoch_df = pd.read_csv(RESULTS_DIR / "epoch_metrics.csv")
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_deg_deltas = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    control_df = pd.read_csv(STATS_DIR / "content_shuffle_controls.csv")
    variance_df = pd.read_csv(STATS_DIR / "haze_scene_variance.csv")
    rank_df = pd.read_csv(STATS_DIR / "effective_rank.csv")

    # 1-2: PSNR/SSIM curves
    for metric, ylabel, fname in [("val_psnr", "Validation PSNR (dB)", "01_val_psnr_AF2T12.png"),
                                   ("val_ssim", "Validation SSIM", "02_val_ssim_AF2T12.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in MODEL_ORDER:
            sub = epoch_df[epoch_df.model == model]
            piv = sub.pivot_table(index="epoch", columns="seed", values=metric)
            ax.plot(piv.index, piv.mean(axis=1), label=f"Model {model}", color=COLORS[model], linewidth=2)
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=COLORS[model], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Model A vs F2 vs T12: {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / fname, dpi=150)
        plt.close(fig)

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
    ax.set_title("Per-Degradation PSNR: A vs F2 vs T12")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_per_degradation_psnr.png", dpi=150)
    plt.close(fig)

    # 4: T12-F2 delta by degradation, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    tf = per_deg_deltas[per_deg_deltas.comparison == "T12-F2"]
    x = np.arange(3)
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [tf[(tf.degradation == d) & (tf.seed == seed)].delta_psnr.iloc[0] for d in degs]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Delta PSNR (dB), T12 - F2")
    ax.set_title("T12-F2 Delta PSNR by Degradation, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_T12_minus_F2_per_degradation.png", dpi=150)
    plt.close(fig)

    # 5: causal control comparison (the key result)
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ["psnr_normal", "psnr_degradation_only", "psnr_content_only", "psnr_shuffled_content"]
    labels = ["Normal\n(e_D, phi(F))", "Degradation-only\n(e_D, phi_bar)", "Content-only\n(0, phi(F))",
              "Shuffled content\n(e_D, phi(F_j))"]
    means = [control_df[c].mean() for c in conditions]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    ax.bar(labels, means, color=colors)
    ax.set_ylabel("Mean Val PSNR (dB)")
    ax.set_title("Causal Control Comparison: Does the Operator Use Image Content?")
    for i, v in enumerate(means):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
    ax.set_ylim(min(means) - 1, max(means) + 1)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_causal_control_comparison.png", dpi=150)
    plt.close(fig)

    # 6: scene variance ratio by degradation
    fig, ax = plt.subplots(figsize=(7, 5))
    piv = variance_df.groupby("degradation")["T12_over_F2_variance_ratio_a0"].mean()
    piv = piv.reindex(["Rain", "Haze", "Noise"])
    ax.bar(piv.index, piv.values, color="#9467bd")
    ax.axhline(1.0, color="black", linestyle="--", label="No difference (ratio=1)")
    ax.set_ylabel("T12 / F2 coefficient variance ratio (across scenes)")
    ax.set_title("Does T12's Coefficient Vary More Across Scenes Than F2's?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_scene_variance_ratio.png", dpi=150)
    plt.close(fig)

    # 7: effective rank T12 vs F2
    fig, ax = plt.subplots(figsize=(6, 5))
    means = [rank_df[rank_df.model == m].effective_rank.mean() for m in ["F2", "T12"]]
    ax.bar(["F2", "T12"], means, color=[COLORS["F2"], COLORS["T12"]])
    ax.axhline(2.0, color="gray", linestyle="--", label="configured rank=2")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective Rank: F2 vs T12 (both configured rank=2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_effective_rank_comparison.png", dpi=150)
    plt.close(fig)

    print(f"wrote 7 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
