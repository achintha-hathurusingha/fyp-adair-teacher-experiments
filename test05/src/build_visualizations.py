"""TEST05 Phase 23: required visualizations.
  1. Candidate degradation/scene ratio comparison
  2. Channel degradation-specificity ranking
  3. Channel causal-effect ranking (from ablation, if available)
  4. Top-channel overlap (discriminative vs causal vs frequency-sensitive)
  5. Frequency spectrum Rain/Haze/Noise (produced by spatial_frequency_analysis.py)
  6. Compact embedding dimensionality vs performance
  7. Candidate distillation target comparison
  8. Minimum sufficient representation curve (% channels vs retained causal effect)

Usage: python build_visualizations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST05 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST05 / "results" / "statistics"
CHANNEL_DIR = TEST05 / "results" / "channel_analysis"
INTERVENTION_DIR = TEST05 / "results" / "intervention"
VIZ_DIR = TEST05 / "results" / "visualizations"


def fig1_ratio_comparison():
    df = pd.read_csv(STATS_DIR / "scene_sensitivity.csv")
    df = df[df.metric == "euclidean"].sort_values("ratio", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df.feature, df.ratio, color="#3b7dd8")
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="ratio=1 (no degradation preference)")
    ax.set_xlabel("degradation_scene_ratio (euclidean)")
    ax.set_title("Top 15 candidates by degradation/scene ratio")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "1_ratio_comparison.png", dpi=130)
    plt.close(fig)
    print("saved 1_ratio_comparison.png")


def fig2_channel_ranking():
    path = CHANNEL_DIR / "channel_rank.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    latent = df[df.feature == "latent_pre"].sort_values("degradation_probe_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(latent)), latent.degradation_probe_accuracy.values * 100, color="#c0392b")
    ax.axhline(33.3, color="gray", ls="--", lw=1, label="random baseline")
    ax.set_xlabel("channel rank (latent_pre, sorted)")
    ax.set_ylabel("single-channel degradation accuracy (%)")
    ax.set_title("latent_pre: per-channel degradation-specificity ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "2_channel_ranking.png", dpi=130)
    plt.close(fig)
    print("saved 2_channel_ranking.png")


def fig3_channel_causal():
    path = INTERVENTION_DIR / "channel_ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    summary = df.groupby("ablation_type")["l2_vs_normal"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary.index, summary.values, color="#3bb273")
    ax.set_ylabel("mean L2 output change (top-10 latent_pre channels)")
    ax.set_title("Channel ablation: keep vs. zero vs. degradation-group-average")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "3_channel_causal_ablation.png", dpi=130)
    plt.close(fig)
    print("saved 3_channel_causal_ablation.png")


def fig4_top_channel_overlap():
    chan_path = CHANNEL_DIR / "channel_rank.csv"
    freq_path = TEST05 / "results" / "frequency_analysis" / "frequency_channel_ranking.csv"
    if not chan_path.exists():
        return
    chan = pd.read_csv(chan_path)
    latent = chan[chan.feature == "latent_pre"].sort_values("degradation_probe_accuracy", ascending=False)
    top_discriminative = set(latent.head(20).channel)

    freq_overlap = set()
    if freq_path.exists():
        freq = pd.read_csv(freq_path)
        freq_latent = freq[freq.feature == "latent_pre"]
        if len(freq_latent):
            fsummary = freq_latent.groupby("channel")["high_freq_pct_variance_across_deg"].mean().sort_values(ascending=False)
            freq_overlap = set(fsummary.head(20).index)

    fig, ax = plt.subplots(figsize=(6, 6))
    # simple text-based overlap summary instead of a venn diagram (no extra dependency)
    overlap = top_discriminative & freq_overlap
    ax.axis("off")
    ax.text(0.5, 0.7, f"Top-20 discriminative channels: {len(top_discriminative)}", ha="center", fontsize=11)
    ax.text(0.5, 0.55, f"Top-20 frequency-distinctive channels: {len(freq_overlap)}", ha="center", fontsize=11)
    ax.text(0.5, 0.4, f"Overlap: {len(overlap)} channels", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.25, f"Overlapping channel IDs: {sorted(overlap) if overlap else 'none'}", ha="center", fontsize=9, wrap=True)
    ax.set_title("Top-channel overlap: degradation-discriminative vs. frequency-distinctive (latent_pre)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "4_top_channel_overlap.png", dpi=130)
    plt.close(fig)
    print("saved 4_top_channel_overlap.png")


def fig6_compact_embedding():
    path = STATS_DIR / "compact_embedding.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pca_rows = df[df.representation.str.startswith("pca_")].copy()
    pca_rows["dim"] = pca_rows["representation"].str.replace("pca_", "").astype(int)
    pca_rows = pca_rows.sort_values("dim")
    full_acc = df[df.representation == "full_latent_pre"]["accuracy"].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pca_rows.dim, pca_rows.accuracy * 100, marker="o", color="#3b7dd8", label="PCA-compressed latent_pre")
    ax.axhline(full_acc * 100, color="gray", ls="--", label=f"full latent_pre (768-dim) = {full_acc*100:.1f}%")
    ax.axhline(33.3, color="lightgray", ls=":", label="random baseline")
    ax.set_xlabel("compact embedding dimensionality")
    ax.set_ylabel("degradation classification accuracy (%)")
    ax.set_title("Compact embedding: dimensionality vs. degradation-probe accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "6_compact_embedding.png", dpi=130)
    plt.close(fig)
    print("saved 6_compact_embedding.png")


def fig7_candidate_comparison():
    path = STATS_DIR / "distillation_candidate_ranking.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).dropna(subset=["degradation_accuracy_pct"]).sort_values("composite_score", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df.candidate, df.composite_score, color="#8e44ad")
    ax.set_xlabel("composite score (accuracy 0.25 + deg/scene ratio 0.35 + causal 0.20 + compactness 0.20)")
    ax.set_title("Distillation candidate comparison (composite score)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "7_candidate_comparison.png", dpi=130)
    plt.close(fig)
    print("saved 7_candidate_comparison.png")


def fig8_minimum_sufficient():
    path = INTERVENTION_DIR / "channel_group_intervention.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    full_effect = df[df.group_type == "full_tensor"].l2_vs_normal.mean()
    top = df[df.group_type == "top_degradation_specific"].groupby("pct").l2_vs_normal.mean()
    rand = df[df.group_type == "random_same_size"].groupby("pct").l2_vs_normal.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = sorted(top.index)
    ax.plot([p * 100 for p in pcts], [top[p] / full_effect * 100 for p in pcts], marker="o",
            color="#3b7dd8", label="top degradation-specific channels")
    ax.plot([p * 100 for p in pcts], [rand[p] / full_effect * 100 for p in pcts], marker="s",
            color="gray", label="random channels (same size)")
    ax.axhline(100, color="lightgray", ls="--", label="full tensor (100%)")
    ax.set_xlabel("% of channels swapped")
    ax.set_ylabel("% of full-tensor causal effect retained")
    ax.set_title("Minimum sufficient representation: causal effect vs. channel subset size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "8_minimum_sufficient_representation.png", dpi=130)
    plt.close(fig)
    print("saved 8_minimum_sufficient_representation.png")


if __name__ == "__main__":
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig1_ratio_comparison()
    fig2_channel_ranking()
    fig3_channel_causal()
    fig4_top_channel_overlap()
    fig6_compact_embedding()
    fig7_candidate_comparison()
    fig8_minimum_sufficient()
