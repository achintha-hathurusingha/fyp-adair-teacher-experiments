"""TEST07-Pilot: model_summary.csv + all 6 required visualizations. Runs
LOCALLY (matplotlib, no GPU needed -- reads results.csv/representation_probe.csv)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST07 = Path(__file__).resolve().parent.parent
RESULTS = TEST07 / "results"
OUT = RESULTS / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
COLORS = {"A": "#888", "B": "#2f6690", "C": "#e07b39", "D": "#5b8c5a"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def main():
    df = pd.read_csv(RESULTS / "results.csv")
    probe = pd.read_csv(RESULTS / "statistics" / "representation_probe.csv")

    # ---- model_summary.csv ----
    summary_rows = []
    for m in ["A", "B", "C", "D"]:
        sub = df[df.model == m]
        last = sub[sub.epoch == sub.epoch.max()].iloc[0]
        bottleneck_shape = "(B, 256, 8, 8)"
        embedding_dim = 16 if m != "A" else None
        summary_rows.append({
            "model": m, "params": int(last.params), "macs": int(last.macs),
            "bottleneck_shape": bottleneck_shape, "embedding_dim": embedding_dim,
            "final_kd_loss": float(last.train_kd_loss), "extra_parameters": int(last.params) - int(df[df.model=="A"].iloc[-1].params),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "model_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # ---- 1. Validation PSNR vs epoch ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["A", "B", "C", "D"]:
        sub = df[df.model == m].sort_values("epoch")
        ax.plot(sub.epoch, sub.val_psnr, marker="o", label=f"Model {m}", color=COLORS[m])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("1. Validation PSNR vs. epoch"); ax.legend()
    save(fig, "01_val_psnr_vs_epoch.png")

    # ---- 2. Validation SSIM vs epoch ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["A", "B", "C", "D"]:
        sub = df[df.model == m].sort_values("epoch")
        ax.plot(sub.epoch, sub.val_ssim, marker="o", label=f"Model {m}", color=COLORS[m])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation SSIM")
    ax.set_title("2. Validation SSIM vs. epoch"); ax.legend()
    save(fig, "02_val_ssim_vs_epoch.png")

    # ---- 3. PSNR comparison across models (last-5-epoch mean, smoothed) ----
    smoothed = df[df.epoch >= 11].groupby("model")[["val_psnr", "val_ssim"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(smoothed.index, smoothed.val_psnr, color=[COLORS[m] for m in smoothed.index])
    ax.set_ylabel("Mean validation PSNR, epochs 11-15 (dB)")
    ax.set_title("3. PSNR comparison across models (smoothed, last 5 epochs)\n"
                  "differences are within pilot-scale noise")
    save(fig, "03_psnr_comparison_models.png")

    # ---- 4. Degradation-wise PSNR comparison ----
    last_epoch = df[df.epoch == df.epoch.max()]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(4)
    width = 0.25
    for i, deg in enumerate(["rain_psnr", "haze_psnr", "noise_psnr"]):
        vals = [last_epoch[last_epoch.model == m][deg].values[0] for m in ["A", "B", "C", "D"]]
        ax.bar(x + (i - 1) * width, vals, width=width, label=deg.replace("_psnr", "").capitalize())
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_ylabel("PSNR (dB, final epoch)")
    ax.set_title("4. Degradation-wise PSNR comparison"); ax.legend()
    save(fig, "04_degradation_wise_psnr.png")

    # ---- 5. Teacher vs student degradation-probe accuracy ----
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b"] + [COLORS[m] for m in ["A", "B", "C", "D"]]
    ax.bar(probe.representation, probe.accuracy * 100, color=colors)
    ax.axhline(33.3, color="gray", linestyle="--", label="chance")
    ax.set_ylabel("Degradation probe accuracy (%)")
    ax.set_title("5. Teacher vs. student degradation-probe accuracy\n"
                  "student captures a weak signal, far below teacher")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, "05_teacher_vs_student_probe.png")

    # ---- 6. Extra parameter/MAC cost vs PSNR ----
    merged = summary_df.merge(smoothed.reset_index(), on="model")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged.extra_parameters, merged.val_psnr, s=80, c=[COLORS[m] for m in merged.model])
    for _, row in merged.iterrows():
        ax.annotate(row.model, (row.extra_parameters, row.val_psnr), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Extra parameters vs. Model A")
    ax.set_ylabel("Mean validation PSNR, epochs 11-15 (dB)")
    ax.set_title("6. Additional parameter cost vs. PSNR\n(theoretical cost estimate, not NPU latency)")
    save(fig, "06_param_cost_vs_psnr.png")


if __name__ == "__main__":
    main()
