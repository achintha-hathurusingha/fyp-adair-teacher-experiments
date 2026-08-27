"""TEST05.5: 10 required visualizations. Runs LOCALLY (matplotlib, no GPU)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST05_5 = Path(__file__).resolve().parent.parent
RESULTS = TEST05_5 / "results"
OUT = RESULTS / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


# 1. Simple statistics vs AdaIR: accuracy comparison
def viz1():
    simple = pd.read_csv(RESULTS / "simple_stats" / "simple_statistics_probe.csv")
    fft_only = pd.read_csv(RESULTS / "simple_stats" / "simple_statistics_fft_only_probe.csv")
    fam = pd.read_csv(RESULTS / "robustness" / "family_probe_results.csv")
    labels = ["Simple stats\n(logreg, easy set)", "3 FFT bands\nonly (easy set)", "Raw input\n(hard dataset)",
              "AdaIR latent_pre\n(hard dataset)"]
    vals = [simple[simple.classifier == "logreg"]["accuracy_mean"].values[0] * 100,
            fft_only.iloc[0]["accuracy_mean"] * 100,
            fam[fam.candidate == "input"]["accuracy_mean"].values[0] * 100,
            fam[fam.candidate == "latent_pre"]["accuracy_mean"].values[0] * 100]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#888", "#aaa", "#e07b39", "#2f6690"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(33.3, color="red", linestyle="--", label="chance (33.3%)")
    ax.set_ylabel("Degradation classification accuracy (%)")
    ax.set_title("Simple statistics vs. AdaIR representation\n(easy vs. hard dataset)")
    ax.legend()
    save(fig, "01_simple_stats_vs_adair.png")


# 2. PCA leakage-safe vs original
def viz2():
    df = pd.read_csv(RESULTS / "pca_audit" / "pca_leakage_safe_results.csv")
    pca_rows = df[df.representation.str.startswith("pca_")]
    dims = [int(r.replace("pca_", "").replace("_leakage_safe", "")) for r in pca_rows.representation]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(dims, pca_rows.mean_accuracy * 100, yerr=pca_rows.std_accuracy * 100,
                marker="o", capsize=4, color="#2f6690", label="Leakage-safe (TEST05.5)")
    ax.axhline(99.7, color="gray", linestyle="--", label="TEST05 original PCA-16 (leaky): 99.7%")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Degradation classification accuracy (%)")
    ax.set_title("Leakage-safe PCA re-audit (Phase 2)")
    ax.legend()
    save(fig, "02_pca_leakage_audit.png")


# 3. Severity generalization
def viz3():
    df = pd.read_csv(RESULTS / "robustness" / "severity_generalization_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cands = df.candidate.unique()
    x = np.arange(len(cands))
    ab = [df[(df.candidate == c) & (df.train_band == "A")]["accuracy"].values[0] * 100 for c in cands]
    ba = [df[(df.candidate == c) & (df.train_band == "B")]["accuracy"].values[0] * 100 for c in cands]
    ax.bar(x - 0.2, ab, width=0.4, label="Train A / Test B", color="#2f6690")
    ax.bar(x + 0.2, ba, width=0.4, label="Train B / Test A", color="#e07b39")
    ax.axhline(33.3, color="red", linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(cands, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Severity-band generalization (Phase 3-4)")
    ax.legend()
    save(fig, "03_severity_generalization.png")


# 4. Normalized intervention effect sizes
def viz4():
    df = pd.read_csv(RESULTS / "intervention" / "normalized_intervention_summary.csv")
    points = df.point.unique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(points))
    same = [df[(df.point == p) & (df.condition == "same_scene_cross_degradation")]["normalized_change"].values[0] for p in points]
    cross = [df[(df.point == p) & (df.condition == "cross_scene_same_degradation")]["normalized_change"].values[0] for p in points]
    ax.bar(x - 0.2, same, width=0.4, label="Same-scene, cross-degradation", color="#2f6690")
    ax.bar(x + 0.2, cross, width=0.4, label="Cross-scene, same-degradation", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(points, rotation=20, ha="right")
    ax.set_ylabel("Normalized output change (||ΔY||/||Y||)")
    ax.set_title("Normalized causal intervention (Phase 5-6)\ndegradation-sensitivity now exceeds scene-sensitivity")
    ax.legend()
    save(fig, "04_normalized_intervention.png")


# 5. Degradation-specificity ratio
def viz5():
    df = pd.read_csv(RESULTS / "intervention" / "degradation_specificity_ratio.csv")
    sub = df[df.metric == "normalized_change"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sub.point, sub.degradation_specificity_ratio, color="#2f6690")
    ax.axhline(1.0, color="red", linestyle="--", label="ratio = 1 (no preference)")
    ax.set_ylabel("Degradation-specificity ratio\n(same-scene / cross-scene effect)")
    ax.set_title("Degradation-specificity ratio, normalized (Phase 5-6)")
    ax.set_xticklabels(sub.point, rotation=20, ha="right")
    ax.legend()
    save(fig, "05_degradation_specificity_ratio.png")


# 6. Frequency-path ablation: T0-T3 comparison
def viz6():
    df = pd.read_csv(RESULTS / "frequency" / "variant_representation_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics = ["degradation_probe_accuracy", "pca16_leakage_safe_accuracy", "degradation_scene_ratio"]
    titles = ["Degradation probe accuracy", "PCA-16 accuracy", "Degradation/scene ratio"]
    variants = df.variant.unique()
    points = df.point.unique()
    colors = {"T0_released": "#2f6690", "T1_no_frequency": "#e07b39", "T2_matched_random": "#5b8c5a", "T3_phase_shuffle": "#a05195"}
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(points))
        width = 0.2
        for i, v in enumerate(variants):
            vals = [df[(df.variant == v) & (df.point == p)][metric].values[0] for p in points]
            if metric != "degradation_scene_ratio":
                vals = [val * 100 for val in vals]
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=v.replace("_", " "), color=colors[v])
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_aflb_out", "") for p in points], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Frequency-path ablation: T0 (real) ≈ T1 (disabled) ≈ T2 (random) ≈ T3 (phase-scrambled)\n"
                  "(Phase 7-9 — central test of H_F2S)", fontsize=11)
    save(fig, "06_frequency_ablation_T0_T3.png")


# 7. Restoration quality across variants
def viz7():
    df = pd.read_csv(RESULTS / "frequency" / "variant_restoration_quality.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(df.variant, df.mean_psnr_vs_clean, color="#2f6690")
    ax1.set_ylim(32.5, 32.75)
    ax1.set_ylabel("Mean PSNR vs. clean (dB)")
    ax1.set_xticklabels(df.variant, rotation=30, ha="right", fontsize=8)
    ax2.bar(df.variant, df.mean_ssim_vs_clean, color="#e07b39")
    ax2.set_ylim(0.944, 0.946)
    ax2.set_ylabel("Mean SSIM vs. clean")
    ax2.set_xticklabels(df.variant, rotation=30, ha="right", fontsize=8)
    fig.suptitle("Restoration output quality is flat across T0-T3\n(frequency path is causally inert for output quality)")
    save(fig, "07_restoration_quality_variants.png")


# 8. Frequency randomization control (cross-image swap)
def viz8():
    df = pd.read_csv(RESULTS / "frequency" / "frequency_randomization_control.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df.l2, bins=20, color="#2f6690")
    ax.set_xlabel("L2 output change (swapped vs. normal)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase 10: cross-image frequency-branch swap\nmean L2={df.l2.mean():.4f}, mean SSIM={df.ssim_vs_normal.mean():.5f}")
    save(fig, "08_frequency_randomization_control.png")


# 9. Input vs feature frequency bands
def viz9():
    df = pd.read_csv(RESULTS / "frequency" / "input_vs_feature_frequency_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    points = df.point.unique()
    degs = ["Rain", "Haze", "Noise"]
    x = np.arange(len(points))
    width = 0.25
    colors = {"Rain": "#2f6690", "Haze": "#e07b39", "Noise": "#5b8c5a"}
    for i, d in enumerate(degs):
        vals = [df[(df.point == p) & (df.degradation == d)]["low_frac"].values[0] * 100 for p in points]
        ax.bar(x + (i - 1) * width, vals, width=width, label=d, color=colors[d])
    ax.set_xticks(x)
    ax.set_xticklabels(points, rotation=20, ha="right")
    ax.set_ylabel("Low-frequency energy fraction (%)")
    ax.set_title("Input vs. feature-map frequency (Phase 11, re-verified)\n"
                  "Feature-level bands barely differ by degradation; only INPUT differs as expected")
    ax.legend()
    save(fig, "09_input_vs_feature_frequency.png")


# 10. Negative controls / compactness validation
def viz10():
    df = pd.read_csv(RESULTS / "compact" / "compact_vs_controls.csv")
    d16 = df[df.dim == 16]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = d16.method.tolist()
    vals = (d16.accuracy_mean * 100).tolist()
    colors = ["#2f6690" if "control" not in m and "random" not in m else "#c0392b" for m in methods]
    ax.barh(methods, vals, color=colors)
    ax.axvline(33.3, color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("PCA-16 vs. negative controls (Phase 12-14)\nnegative controls correctly collapse to chance")
    ax.legend()
    save(fig, "10_negative_controls.png")


if __name__ == "__main__":
    for i, fn in enumerate([viz1, viz2, viz3, viz4, viz5, viz6, viz7, viz8, viz9, viz10], 1):
        try:
            fn()
        except Exception as e:
            print(f"FAILED viz{i}: {e}")
