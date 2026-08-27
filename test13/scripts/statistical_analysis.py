"""TEST13: primary statistical analysis. Per-seed paired deltas for T12-F2
(main comparison) and T12-A, F2-A, last5-window PSNR/SSIM primary metric,
mean+-std across 3 seeds, bootstrap CI (exploratory, N=3), same-sign count,
and per-degradation breakdown.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python statistical_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST13 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST13 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 10000
COMPARISONS = [("T13", "F2"), ("T13", "A"), ("F2", "A")]


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    models = ["A", "F2", "T13"]
    per_model = {m: df[df.model == m].set_index("seed").sort_index() for m in models}
    for m in models:
        assert list(per_model[m].index) == SEEDS, f"expected seeds 0,1,2 for model {m}"

    per_seed_rows, summary_rows, per_deg_rows = [], [], []

    for hi_m, lo_m in COMPARISONS:
        hi, lo = per_model[hi_m], per_model[lo_m]
        d_psnr = (hi["last5_mean_psnr"] - lo["last5_mean_psnr"]).values
        d_ssim = (hi["last5_mean_ssim"] - lo["last5_mean_ssim"]).values

        for seed in SEEDS:
            per_seed_rows.append({
                "comparison": f"{hi_m}-{lo_m}", "seed": seed,
                "delta_last5_psnr": float(hi.loc[seed, "last5_mean_psnr"] - lo.loc[seed, "last5_mean_psnr"]),
                "delta_last5_ssim": float(hi.loc[seed, "last5_mean_ssim"] - lo.loc[seed, "last5_mean_ssim"]),
            })

        ci_psnr = bootstrap_ci(d_psnr)
        ci_ssim = bootstrap_ci(d_ssim)
        same_sign_psnr = int(max((d_psnr > 0).sum(), (d_psnr < 0).sum()))
        same_sign_ssim = int(max((d_ssim > 0).sum(), (d_ssim < 0).sum()))
        summary_rows.append({"comparison": f"{hi_m}-{lo_m}", "metric": "delta_last5_psnr", "mean": d_psnr.mean(),
                              "std": d_psnr.std(ddof=1), "bootstrap_ci95_lo": ci_psnr[0],
                              "bootstrap_ci95_hi": ci_psnr[1], "same_sign_count": same_sign_psnr, "n_seeds": 3})
        summary_rows.append({"comparison": f"{hi_m}-{lo_m}", "metric": "delta_last5_ssim", "mean": d_ssim.mean(),
                              "std": d_ssim.std(ddof=1), "bootstrap_ci95_lo": ci_ssim[0],
                              "bootstrap_ci95_hi": ci_ssim[1], "same_sign_count": same_sign_ssim, "n_seeds": 3})

        for deg in ["rain", "haze", "noise"]:
            psnr_col, ssim_col = f"last5_mean_{deg}_psnr", f"last5_mean_{deg}_ssim"
            dp = (hi[psnr_col] - lo[psnr_col]).values
            ds = (hi[ssim_col] - lo[ssim_col]).values
            for seed in SEEDS:
                per_deg_rows.append({
                    "comparison": f"{hi_m}-{lo_m}", "degradation": deg, "seed": seed,
                    "delta_psnr": float(hi.loc[seed, psnr_col] - lo.loc[seed, psnr_col]),
                    "delta_ssim": float(hi.loc[seed, ssim_col] - lo.loc[seed, ssim_col]),
                })

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(STATS_DIR / "per_seed_deltas.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(STATS_DIR / "seed_level_summary_stats.csv", index=False)
    per_deg_df = pd.DataFrame(per_deg_rows)
    per_deg_df.to_csv(STATS_DIR / "per_degradation_deltas.csv", index=False)
    per_deg_summary = per_deg_df.groupby(["comparison", "degradation"])[["delta_psnr", "delta_ssim"]].agg(["mean", "std"])
    per_deg_summary.to_csv(STATS_DIR / "per_degradation_summary.csv")

    print("=== Per-seed deltas, primary metric = last5-window mean ===")
    print(per_seed_df.to_string(index=False))
    print("\n=== Seed-level summary (mean +- std, bootstrap 95% CI, same-sign count) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Per-degradation summary ===")
    print(per_deg_summary.to_string())


if __name__ == "__main__":
    main()
