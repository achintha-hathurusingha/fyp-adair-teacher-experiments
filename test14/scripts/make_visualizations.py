"""TEST14: 9 required visualizations.
1. Frequency-band energy profiles
2. Rain/Haze/Noise q_F distributions
3. q_F scene-vs-degradation variance
4. A/F2/T14 PSNR curves
5. A/F2/T14 SSIM curves
6. T14-F2 per-degradation delta
7. Correct/zero/mean/shuffled frequency control
8. e_D vs q_F redundancy (probe comparison)
9. Coefficient changes Delta_a caused by q_F

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

TEST14 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST14 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODEL_ORDER = ["A", "F2", "T14"]
COLORS = {"A": "#1f77b4", "F2": "#2ca02c", "T14": "#d62728"}
DEGS = ["Rain", "Haze", "Noise"]
BAND_COLS = [f"band{i+1}" for i in range(8)]


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    q_df = pd.read_csv(STATS_DIR / "frequency_descriptors_all_crops.csv")
    scene_var_df = pd.read_csv(STATS_DIR / "qF_scene_variance.csv")
    epoch_df = pd.read_csv(RESULTS_DIR / "epoch_metrics.csv")
    seed_summary = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_deg_deltas = pd.read_csv(STATS_DIR / "per_degradation_deltas.csv")
    control_df = pd.read_csv(STATS_DIR / "frequency_controls.csv")
    probe_df = pd.read_csv(STATS_DIR / "freq_redundancy_probe.csv")
    coeff_df = pd.read_csv(STATS_DIR / "coefficient_analysis.csv")

    # 1: frequency-band energy profiles (overall mean)
    fig, ax = plt.subplots(figsize=(8, 5))
    means = q_df[BAND_COLS].mean()
    ax.bar(range(1, 9), means, color="#1f77b4")
    ax.set_xlabel("Frequency band (1=lowest, 8=highest)")
    ax.set_ylabel("Mean normalized spectral energy")
    ax.set_title("Frequency-Band Energy Profile (all crops)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_frequency_band_profile.png", dpi=150)
    plt.close(fig)

    # 2: Rain/Haze/Noise q_F distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(8)
    width = 0.25
    for i, deg in enumerate(DEGS):
        means = q_df[q_df.degradation == deg][BAND_COLS].mean()
        ax.bar(x + (i - 1) * width, means, width, label=deg)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i+1}" for i in range(8)])
    ax.set_ylabel("Mean normalized spectral energy")
    ax.set_title("q_F Distribution by Degradation")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_qF_distribution_by_degradation.png", dpi=150)
    plt.close(fig)

    # 3: q_F scene-vs-degradation variance
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = scene_var_df.pivot(index="band", columns="degradation", values="variance_across_scenes")
    piv = piv.reindex([f"band{i+1}" for i in range(8)])
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Variance across scenes (within one degradation)")
    ax.set_title("q_F Scene Variance by Band and Degradation")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_qF_scene_variance.png", dpi=150)
    plt.close(fig)

    # 4-5: PSNR/SSIM curves
    for metric, ylabel, fname in [("val_psnr", "Validation PSNR (dB)", "04_val_psnr_AF2T14.png"),
                                   ("val_ssim", "Validation SSIM", "05_val_ssim_AF2T14.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in MODEL_ORDER:
            sub = epoch_df[epoch_df.model == model]
            piv = sub.pivot_table(index="epoch", columns="seed", values=metric)
            ax.plot(piv.index, piv.mean(axis=1), label=f"Model {model}", color=COLORS[model], linewidth=2)
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=COLORS[model], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Model A vs F2 vs T14: {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / fname, dpi=150)
        plt.close(fig)

    # 6: T14-F2 per-degradation delta, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    tf = per_deg_deltas[per_deg_deltas.comparison == "T14-F2"]
    degs_lower = ["rain", "haze", "noise"]
    x = np.arange(3)
    width = 0.25
    for i, seed in enumerate([0, 1, 2]):
        vals = [tf[(tf.degradation == d) & (tf.seed == seed)].delta_psnr.iloc[0] for d in degs_lower]
        ax.bar(x + (i - 1) * width, vals, width, label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs_lower])
    ax.set_ylabel("Delta PSNR (dB), T14 - F2")
    ax.set_title("T14-F2 Delta PSNR by Degradation, per Seed")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_T14_minus_F2_per_degradation.png", dpi=150)
    plt.close(fig)

    # 7: causal control comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ["psnr_normal", "psnr_zero_freq", "psnr_mean_freq", "psnr_shuffled_freq"]
    labels = ["Normal\n(correct q_F)", "Zero q_F", "Mean q_F", "Shuffled q_F"]
    means = [control_df[c].mean() for c in conditions]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    ax.bar(labels, means, color=colors)
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylabel("Mean Val PSNR (dB)")
    ax.set_title("T14 Frequency Causal Control Comparison\n(all four conditions statistically identical)")
    ax.set_ylim(min(means) - 0.3, max(means) + 0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_frequency_causal_control.png", dpi=150)
    plt.close(fig)

    # 8: e_D vs q_F redundancy (probe comparison)
    fig, ax = plt.subplots(figsize=(10, 5))
    probe_df_sorted = probe_df.set_index("features").loc[
        ["q_F", "e_D", "phi(F)_pca16", "[e_D,q_F]", "[phi(F)_pca16,q_F]", "[e_D,phi(F)_pca16]",
         "[e_D,phi(F)_pca16,q_F]"]]
    ax.bar(probe_df_sorted.index, probe_df_sorted.accuracy * 100, color="#9467bd")
    ax.set_ylabel("Degradation probe accuracy (%)")
    ax.set_title("Redundancy: Does Adding q_F Improve the Probe?\n([e_D,phi(F)] and [e_D,phi(F),q_F] are equal)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_eD_vs_qF_redundancy.png", dpi=150)
    plt.close(fig)

    # 9: coefficient changes Delta_a by degradation
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = coeff_df.groupby("degradation")[["delta_a0", "delta_a1"]].mean()
    piv = piv.reindex(DEGS)
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, piv.delta_a0, width, label="delta_a0 (T14-F2)")
    ax.bar(x + width / 2, piv.delta_a1, width, label="delta_a1 (T14-F2)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DEGS)
    ax.set_ylabel("Mean coefficient difference (T14 - F2)")
    ax.set_title("Coefficient Changes Caused by Adding q_F to the Generator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "09_coefficient_delta.png", dpi=150)
    plt.close(fig)

    print(f"wrote 9 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
