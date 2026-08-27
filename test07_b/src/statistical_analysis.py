"""TEST07-B: primary statistical analysis. Per-seed paired deltas (B-A) for
PSNR/SSIM, mean+-std across the 3 seeds, bootstrap CI (resampling over
seeds -- explicitly exploratory given N=3, NOT a claim of significance),
and per-degradation (Rain/Haze/Noise) deltas so a degradation-specific
failure cannot be averaged away.

Primary comparison metric per the task spec: last5_mean_psnr/ssim (the
smoothed final-window average). Secondary: best_psnr/best_ssim.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python statistical_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST07B = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST07B / "results"
STATS_DIR = RESULTS_DIR / "statistics"
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 10000


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    a = df[df.model == "A"].set_index("seed").sort_index()
    b = df[df.model == "B"].set_index("seed").sort_index()
    assert list(a.index) == SEEDS and list(b.index) == SEEDS, "expected seeds 0,1,2 for both A and B"

    # ---- primary: last5_mean (smoothed final-window) ----
    delta_psnr_primary = (b["last5_mean_psnr"] - a["last5_mean_psnr"]).values
    delta_ssim_primary = (b["last5_mean_ssim"] - a["last5_mean_ssim"]).values

    # ---- secondary: best-epoch ----
    delta_psnr_best = (b["best_psnr"] - a["best_psnr"]).values
    delta_ssim_best = (b["best_ssim"] - a["best_ssim"]).values

    rows = []
    for seed in SEEDS:
        rows.append({
            "seed": seed,
            "A_last5_psnr": float(a.loc[seed, "last5_mean_psnr"]), "B_last5_psnr": float(b.loc[seed, "last5_mean_psnr"]),
            "delta_last5_psnr": float(b.loc[seed, "last5_mean_psnr"] - a.loc[seed, "last5_mean_psnr"]),
            "A_last5_ssim": float(a.loc[seed, "last5_mean_ssim"]), "B_last5_ssim": float(b.loc[seed, "last5_mean_ssim"]),
            "delta_last5_ssim": float(b.loc[seed, "last5_mean_ssim"] - a.loc[seed, "last5_mean_ssim"]),
            "A_best_psnr": float(a.loc[seed, "best_psnr"]), "B_best_psnr": float(b.loc[seed, "best_psnr"]),
            "delta_best_psnr": float(b.loc[seed, "best_psnr"] - a.loc[seed, "best_psnr"]),
            "A_best_ssim": float(a.loc[seed, "best_ssim"]), "B_best_ssim": float(b.loc[seed, "best_ssim"]),
            "delta_best_ssim": float(b.loc[seed, "best_ssim"] - a.loc[seed, "best_ssim"]),
        })
    per_seed_df = pd.DataFrame(rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(STATS_DIR / "per_seed_deltas.csv", index=False)

    ci_psnr = bootstrap_ci(delta_psnr_primary)
    ci_ssim = bootstrap_ci(delta_ssim_primary)
    ci_psnr_best = bootstrap_ci(delta_psnr_best)
    ci_ssim_best = bootstrap_ci(delta_ssim_best)

    summary = {
        "metric": ["delta_last5_psnr", "delta_last5_ssim", "delta_best_psnr", "delta_best_ssim"],
        "mean": [delta_psnr_primary.mean(), delta_ssim_primary.mean(), delta_psnr_best.mean(), delta_ssim_best.mean()],
        "std": [delta_psnr_primary.std(ddof=1), delta_ssim_primary.std(ddof=1),
                delta_psnr_best.std(ddof=1), delta_ssim_best.std(ddof=1)],
        "bootstrap_ci95_lo": [ci_psnr[0], ci_ssim[0], ci_psnr_best[0], ci_ssim_best[0]],
        "bootstrap_ci95_hi": [ci_psnr[1], ci_ssim[1], ci_psnr_best[1], ci_ssim_best[1]],
        "n_seeds": [3, 3, 3, 3],
        "note": ["EXPLORATORY (N=3 seeds) -- not a claim of statistical significance"] * 4,
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(STATS_DIR / "seed_level_summary_stats.csv", index=False)

    # ---- per-degradation (Rain/Haze/Noise), last5-window ----
    deg_rows = []
    for deg in ["rain", "haze", "noise"]:
        psnr_col, ssim_col = f"last5_mean_{deg}_psnr", f"last5_mean_{deg}_ssim"
        d_psnr = (b[psnr_col] - a[psnr_col]).values
        d_ssim = (b[ssim_col] - a[ssim_col]).values
        for seed in SEEDS:
            deg_rows.append({
                "degradation": deg, "seed": seed,
                "A_psnr": float(a.loc[seed, psnr_col]), "B_psnr": float(b.loc[seed, psnr_col]),
                "delta_psnr": float(b.loc[seed, psnr_col] - a.loc[seed, psnr_col]),
                "A_ssim": float(a.loc[seed, ssim_col]), "B_ssim": float(b.loc[seed, ssim_col]),
                "delta_ssim": float(b.loc[seed, ssim_col] - a.loc[seed, ssim_col]),
            })
    per_deg_df = pd.DataFrame(deg_rows)
    per_deg_df.to_csv(STATS_DIR / "per_degradation_deltas.csv", index=False)

    per_deg_summary = per_deg_df.groupby("degradation")[["delta_psnr", "delta_ssim"]].agg(["mean", "std"])
    per_deg_summary.to_csv(STATS_DIR / "per_degradation_summary.csv")

    print("=== Per-seed deltas (B - A), primary metric = last5-window mean ===")
    print(per_seed_df[["seed", "A_last5_psnr", "B_last5_psnr", "delta_last5_psnr",
                        "A_last5_ssim", "B_last5_ssim", "delta_last5_ssim"]].to_string(index=False))
    print("\n=== Seed-level summary (mean +- std across 3 seeds, bootstrap 95% CI) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Per-degradation deltas (B - A), last5-window mean, per seed ===")
    print(per_deg_df.to_string(index=False))
    print("\n=== Per-degradation summary (mean +- std across seeds) ===")
    print(per_deg_summary.to_string())

    all_negative_or_zero = (delta_psnr_primary <= 0).all() and (delta_ssim_primary <= 0).all()
    all_positive = (delta_psnr_primary > 0).all() and (delta_ssim_primary >= 0).all()
    print(f"\nAll 3 seeds delta_psnr<=0 and delta_ssim<=0: {all_negative_or_zero}")
    print(f"All 3 seeds delta_psnr>0 and delta_ssim>=0: {all_positive}")


if __name__ == "__main__":
    main()
