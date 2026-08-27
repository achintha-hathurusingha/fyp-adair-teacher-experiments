"""TEST11: 10 required visualizations.
1. Overall PSNR vs rank
2. Overall SSIM vs rank
3. Rain PSNR vs rank
4. Haze PSNR vs rank
5. Noise PSNR vs rank
6. Delta PSNR relative to baseline (A), vs rank
7. Parameter overhead vs rank
8. Effective rank vs configured rank
9. Coefficient magnitude by degradation, vs rank
10. Modulation magnitude by degradation, vs rank

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

TEST11 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST11 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
RANK_MODELS = ["F2", "F4", "F8", "F16"]
RANKS = [2, 4, 8, 16]


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_deg_deltas = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    coeff_df = pd.read_csv(STATS_DIR / "coefficient_analysis.csv")
    mod_df = pd.read_csv(STATS_DIR / "modulation_magnitude.csv")
    eff_rank_df = pd.read_csv(STATS_DIR / "effective_rank.csv")

    a_mean = seed_summary[seed_summary.model == "A"].last5_mean_psnr.mean()
    a_ssim_mean = seed_summary[seed_summary.model == "A"].last5_mean_ssim.mean()

    def rank_curve(metric_col, ylabel, title, fname, baseline=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        means = [seed_summary[seed_summary.model == m][metric_col].mean() for m in RANK_MODELS]
        stds = [seed_summary[seed_summary.model == m][metric_col].std() for m in RANK_MODELS]
        ax.errorbar(RANKS, means, yerr=stds, marker="o", capsize=4, color="#2ca02c", linewidth=2)
        if baseline is not None:
            ax.axhline(baseline, color="#1f77b4", linestyle="--", label="Baseline A")
            ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels(RANKS)
        ax.set_xlabel("Rank R")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / fname, dpi=150)
        plt.close(fig)

    # 1-2: overall PSNR/SSIM vs rank
    rank_curve("last5_mean_psnr", "Val PSNR (dB)", "Overall PSNR vs Rank", "01_overall_psnr_vs_rank.png",
               baseline=a_mean)
    rank_curve("last5_mean_ssim", "Val SSIM", "Overall SSIM vs Rank", "02_overall_ssim_vs_rank.png",
               baseline=a_ssim_mean)

    # 3-5: per-degradation PSNR vs rank
    for deg, fname_num in [("rain", "03"), ("haze", "04"), ("noise", "05")]:
        a_deg_mean = seed_summary[seed_summary.model == "A"][f"last5_mean_{deg}_psnr"].mean()
        rank_curve(f"last5_mean_{deg}_psnr", "Val PSNR (dB)", f"{deg.capitalize()} PSNR vs Rank",
                   f"{fname_num}_{deg}_psnr_vs_rank.png", baseline=a_deg_mean)

    # 6: delta PSNR relative to baseline, vs rank
    fig, ax = plt.subplots(figsize=(7, 5))
    deltas = [seed_summary[seed_summary.model == m].last5_mean_psnr.mean() - a_mean for m in RANK_MODELS]
    delta_stds = [seed_summary[seed_summary.model == m].last5_mean_psnr.std() for m in RANK_MODELS]
    ax.errorbar(RANKS, deltas, yerr=delta_stds, marker="o", capsize=4, color="#d62728", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("Rank R")
    ax.set_ylabel("Delta PSNR vs Baseline A (dB)")
    ax.set_title("Delta PSNR Relative to Baseline, vs Rank")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_delta_psnr_vs_baseline.png", dpi=150)
    plt.close(fig)

    # 7: parameter overhead vs rank
    fig, ax = plt.subplots(figsize=(7, 5))
    params = [seed_summary[seed_summary.model == m].params.iloc[0] for m in RANK_MODELS]
    a_params = seed_summary[seed_summary.model == "A"].params.iloc[0]
    overhead = [p - a_params for p in params]
    ax.plot(RANKS, overhead, marker="o", color="#9467bd", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("Rank R")
    ax.set_ylabel("Extra parameters vs Baseline A")
    ax.set_title("Parameter Overhead vs Rank")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_param_overhead_vs_rank.png", dpi=150)
    plt.close(fig)

    # 8: effective rank vs configured rank
    fig, ax = plt.subplots(figsize=(7, 5))
    eff_all = eff_rank_df[eff_rank_df.degradation == "ALL"]
    means = [eff_all[eff_all.model == m].effective_rank.mean() for m in RANK_MODELS]
    stds = [eff_all[eff_all.model == m].effective_rank.std() for m in RANK_MODELS]
    ax.errorbar(RANKS, means, yerr=stds, marker="o", capsize=4, color="#8c564b", linewidth=2, label="Effective rank")
    ax.plot(RANKS, RANKS, linestyle="--", color="gray", label="y=x (full utilization)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("Configured Rank R")
    ax.set_ylabel("Effective Rank (participation ratio)")
    ax.set_title("Effective Rank vs Configured Rank")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_effective_rank_vs_configured.png", dpi=150)
    plt.close(fig)

    # 9: coefficient magnitude by degradation, vs rank
    fig, ax = plt.subplots(figsize=(8, 5))
    for deg in ["Rain", "Haze", "Noise"]:
        means = [coeff_df[(coeff_df.model == m) & (coeff_df.degradation == deg)].coeff_l2_magnitude.mean()
                 for m in RANK_MODELS]
        ax.plot(RANKS, means, marker="o", label=deg, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("Rank R")
    ax.set_ylabel("Coefficient L2 magnitude ||a(e_S)||")
    ax.set_title("Coefficient Magnitude by Degradation, vs Rank")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "09_coefficient_magnitude_by_degradation.png", dpi=150)
    plt.close(fig)

    # 10: modulation magnitude by degradation, vs rank
    fig, ax = plt.subplots(figsize=(8, 5))
    for deg in ["Rain", "Haze", "Noise"]:
        means = [mod_df[(mod_df.model == m) & (mod_df.degradation == deg)].relative_modulation_magnitude.mean()
                 for m in RANK_MODELS]
        ax.plot(RANKS, means, marker="o", label=deg, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels(RANKS)
    ax.set_xlabel("Rank R")
    ax.set_ylabel("Relative modulation magnitude ||F_cond-F||/||F||")
    ax.set_title("Modulation Magnitude by Degradation, vs Rank")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "10_modulation_magnitude_by_degradation.png", dpi=150)
    plt.close(fig)

    print(f"wrote 10 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
