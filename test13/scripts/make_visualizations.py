"""TEST13: 9 required visualizations.
1. PSNR A/F2/T13 curves
2. SSIM A/F2/T13 curves
3. Rain/Haze/Noise PSNR deltas (T13-F2)
4. ||dU|| by degradation
5. ||dV|| by degradation
6. Effective basis rank (dU, dV)
7. Basis singular-value distributions
8. Correct vs shuffled-content causal control (all 5 conditions)
9. Parameter overhead vs PSNR

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

TEST13 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST13 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "F2", "T13"]
COLORS = {"A": "#1f77b4", "F2": "#2ca02c", "T13": "#d62728"}
DEGS = ["Rain", "Haze", "Noise"]


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    epoch_df = pd.read_csv(RESULTS_DIR / "epoch_metrics.csv")
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_deg_deltas = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    basis_df = pd.read_csv(STATS_DIR / "basis_adaptation.csv")
    rank_df = pd.read_csv(STATS_DIR / "basis_effective_rank.csv")
    control_df = pd.read_csv(STATS_DIR / "content_controls.csv")

    # 1-2: PSNR/SSIM curves
    for metric, ylabel, fname in [("val_psnr", "Validation PSNR (dB)", "01_val_psnr_AF2T13.png"),
                                   ("val_ssim", "Validation SSIM", "02_val_ssim_AF2T13.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in MODEL_ORDER:
            sub = epoch_df[epoch_df.model == model]
            piv = sub.pivot_table(index="epoch", columns="seed", values=metric)
            ax.plot(piv.index, piv.mean(axis=1), label=f"Model {model}", color=COLORS[model], linewidth=2)
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=COLORS[model], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Model A vs F2 vs T13: {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / fname, dpi=150)
        plt.close(fig)

    # 3: T13-F2 delta by degradation, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    tf = per_deg_deltas[per_deg_deltas.comparison == "T13-F2"]
    degs_lower = ["rain", "haze", "noise"]
    x = np.arange(3)
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [tf[(tf.degradation == d) & (tf.seed == seed)].delta_psnr.iloc[0] for d in degs_lower]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs_lower])
    ax.set_ylabel("Delta PSNR (dB), T13 - F2")
    ax.set_title("T13-F2 Delta PSNR by Degradation, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_T13_minus_F2_per_degradation.png", dpi=150)
    plt.close(fig)

    # 4-5: ||dU|| / ||dV|| by degradation
    for col, fname, title in [("delta_U_norm", "04_deltaU_by_degradation.png", "||dU|| by Degradation"),
                               ("delta_V_norm", "05_deltaV_by_degradation.png", "||dV|| by Degradation")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot([basis_df[basis_df.degradation == d][col] for d in DEGS], tick_labels=DEGS)
        ax.set_ylabel(col)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / fname, dpi=150)
        plt.close(fig)

    # 6: effective basis rank (dU, dV)
    fig, ax = plt.subplots(figsize=(6, 5))
    means = [rank_df.effective_rank_deltaU.mean(), rank_df.effective_rank_deltaV.mean()]
    ax.bar(["Effective rank dU", "Effective rank dV"], means, color=["#9467bd", "#8c564b"])
    ax.axhline(2.0, color="gray", linestyle="--", label="configured rank=2")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective Rank of Basis Corrections (dU, dV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_effective_basis_rank.png", dpi=150)
    plt.close(fig)

    # 7: singular-value distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([basis_df["sv_U_0"], basis_df["sv_U_1"]], tick_labels=["sv_U_0", "sv_U_1"])
    axes[0].set_title("Singular Values of U(e)")
    axes[1].boxplot([basis_df["sv_V_0"], basis_df["sv_V_1"]], tick_labels=["sv_V_0", "sv_V_1"])
    axes[1].set_title("Singular Values of V(e)")
    fig.suptitle("Basis Singular-Value Distributions Across Validation Samples")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_basis_singular_values.png", dpi=150)
    plt.close(fig)

    # 8: causal control comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = ["psnr_normal", "psnr_zero_eD", "psnr_mean_content", "psnr_shuffled_content",
                  "psnr_shuffled_basis_state"]
    labels = ["Normal", "Zero e_D", "Mean content", "Shuffled\ncontent", "Shuffled\nbasis state"]
    means = [control_df[c].mean() for c in conditions]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    ax.bar(labels, means, color=colors)
    for i, v in enumerate(means):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center")
    ax.set_ylabel("Mean Val PSNR (dB)")
    ax.set_title("T13 Causal Control Comparison")
    ax.set_ylim(min(means) - 0.5, max(means) + 0.5)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_causal_control_comparison.png", dpi=150)
    plt.close(fig)

    # 9: parameter overhead vs PSNR
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODEL_ORDER:
        params = seed_summary[seed_summary.model == m].params.iloc[0]
        psnr = seed_summary[seed_summary.model == m].last5_mean_psnr.mean()
        ax.scatter(params, psnr, s=120, color=COLORS[m], label=f"Model {m}")
        ax.annotate(m, (params, psnr), textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Mean Val PSNR (dB)")
    ax.set_title("Parameter Overhead vs Restoration Quality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "09_param_overhead_vs_psnr.png", dpi=150)
    plt.close(fig)

    print(f"wrote 9 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
