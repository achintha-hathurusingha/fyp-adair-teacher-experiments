"""TEST08-C: 10 required visualizations.
1. A/B/C validation PSNR (training curves, mean +- range across seeds)
2. A/B/C validation SSIM (training curves)
3. Per-degradation PSNR: Rain/Haze/Noise, A vs B vs C
4. Delta PSNR: C-A, B-A, C-B (per-seed bars)
5. Student degradation probe: A/B/C/Teacher
6. Gamma distribution by degradation (Model C)
7. Beta distribution by degradation (Model C)
8. Bottleneck modulation magnitude by degradation
9. Learned vs random/shuffled/zero conditioning comparison
10. Student embedding intervention: Rain<->Haze, Rain<->Noise, Haze<->Noise

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

TEST08C = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST08C / "results"
STATS_DIR = RESULTS_DIR / "statistics"
VIZ_DIR = RESULTS_DIR / "visualizations"
SEEDS = [0, 1, 2]
COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}


def plot_metric_bands(epoch_df, metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ["A", "B", "C"]:
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
    probe_df = pd.read_csv(STATS_DIR / "representation_probe.csv")
    per_seed_df = pd.read_csv(STATS_DIR / "per_seed_deltas.csv")
    cond_df = pd.read_csv(STATS_DIR / "conditioning_statistics.csv")
    change_df = pd.read_csv(STATS_DIR / "bottleneck_change.csv")
    control_df = pd.read_csv(STATS_DIR / "random_control.csv")
    interv_df = pd.read_csv(STATS_DIR / "embedding_intervention.csv")

    # 1-2: A/B/C val PSNR/SSIM curves
    plot_metric_bands(epoch_df, "val_psnr", "Validation PSNR (dB)", "Model A vs B vs C: Validation PSNR",
                       "01_val_psnr_ABC.png")
    plot_metric_bands(epoch_df, "val_ssim", "Validation SSIM", "Model A vs B vs C: Validation SSIM",
                       "02_val_ssim_ABC.png")

    # 3: per-degradation PSNR, A vs B vs C
    fig, ax = plt.subplots(figsize=(9, 5))
    degs = ["rain", "haze", "noise"]
    x = np.arange(len(degs))
    width = 0.25
    for i, model in enumerate(["A", "B", "C"]):
        means = [seed_summary[seed_summary.model == model][f"last5_mean_{d}_psnr"].mean() for d in degs]
        stds = [seed_summary[seed_summary.model == model][f"last5_mean_{d}_psnr"].std() for d in degs]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=f"Model {model}", color=COLORS[model], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in degs])
    ax.set_ylabel("Val PSNR (dB), last5-window mean")
    ax.set_title("Per-Degradation PSNR: A vs B vs C (mean +- std across 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_per_degradation_psnr_ABC.png", dpi=150)
    plt.close(fig)

    # 4: delta PSNR C-A, B-A, C-B, per seed
    fig, ax = plt.subplots(figsize=(8, 5))
    comparisons = ["C-A", "B-A", "C-B"]
    xw = np.arange(len(comparisons))
    width = 0.25
    for i, seed in enumerate(SEEDS):
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
    fig.savefig(VIZ_DIR / "04_delta_psnr_comparisons.png", dpi=150)
    plt.close(fig)

    # 5: student degradation probe, A/B/C/Teacher
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = probe_df.groupby("representation")["accuracy"].agg(["mean", "std"]).reindex(
        ["teacher_PCA16", "model_A_bottleneck", "model_B_bottleneck", "model_B_eS",
         "model_C_bottleneck", "model_C_eS"])
    ax.bar(agg.index, agg["mean"] * 100, yerr=agg["std"] * 100, capsize=4,
           color=["gray", COLORS["A"], COLORS["B"], COLORS["B"], COLORS["C"], COLORS["C"]])
    ax.set_ylabel("Degradation probe accuracy (%)")
    ax.set_title("Representation Probe: Teacher vs A/B/C")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_representation_probe.png", dpi=150)
    plt.close(fig)

    # 6-7: gamma/beta distribution by degradation (Model C)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["Rain", "Haze", "Noise"]
    ax.boxplot([cond_df[cond_df.degradation == d].gamma_mean for d in order], tick_labels=order)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="identity (gamma=1)")
    ax.set_ylabel("gamma (per-sample mean over 256 channels)")
    ax.set_title("Model C: Gamma Distribution by Degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_gamma_by_degradation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([cond_df[cond_df.degradation == d].beta_mean for d in order], tick_labels=order)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="identity (beta=0)")
    ax.set_ylabel("beta (per-sample mean over 256 channels)")
    ax.set_title("Model C: Beta Distribution by Degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "07_beta_by_degradation.png", dpi=150)
    plt.close(fig)

    # 8: bottleneck modulation magnitude by degradation
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([change_df[change_df.degradation == d].relative_bottleneck_change for d in order], tick_labels=order)
    ax.set_ylabel("||F_cond - F||_2 / ||F||_2")
    ax.set_title("Relative Bottleneck Modulation Magnitude by Degradation")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "08_bottleneck_modulation_magnitude.png", dpi=150)
    plt.close(fig)

    # 9: learned vs random/shuffled/zero conditioning
    fig, ax = plt.subplots(figsize=(7, 5))
    order_cond = ["learned", "zero", "shuffled", "random_matched"]
    means = [control_df[control_df.condition == c].psnr.mean() for c in order_cond]
    stds = [control_df[control_df.condition == c].psnr.std() for c in order_cond]
    ax.bar(order_cond, means, yerr=stds, capsize=4, color=["#2ca02c", "#7f7f7f", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Val PSNR (dB)")
    ax.set_title("Model C: Learned vs Random/Shuffled/Zero Conditioning")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "09_learned_vs_random_conditioning.png", dpi=150)
    plt.close(fig)

    # 10: student embedding intervention, delta PSNR by direction
    fig, ax = plt.subplots(figsize=(9, 5))
    interv_df["direction"] = interv_df.recipient_degradation + " <- " + interv_df.donor_degradation
    summary = interv_df.groupby("direction").apply(
        lambda g: g.psnr_vs_clean.mean() - g.recipient_normal_psnr.mean(), include_groups=False)
    summary = summary.sort_values()
    ax.barh(summary.index, summary.values, color="#9467bd")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta PSNR vs recipient's normal output (dB)")
    ax.set_title("Student-Side Embedding Intervention (donor conditioning swapped in)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "10_embedding_intervention.png", dpi=150)
    plt.close(fig)

    print(f"wrote 10 visualizations to {VIZ_DIR}")
    for p in sorted(VIZ_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
