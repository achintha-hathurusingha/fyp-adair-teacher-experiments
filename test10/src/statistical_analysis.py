"""TEST10: primary statistical analysis. Per-seed paired deltas for
G-F, G-A, F-A (last5-window PSNR/SSIM primary metric, best-epoch secondary),
mean+-std across 3 seeds, bootstrap CI (exploratory, N=3), and
per-degradation (Rain/Haze/Noise) breakdown. G-F is the most important
comparison (does trajectory distillation help beyond TEST09's best
mechanism, especially for Haze).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python statistical_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST10 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST10 / "results"
STATS_DIR = RESULTS_DIR / "statistics"
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 10000
COMPARISONS = [("G", "F"), ("G", "A"), ("F", "A")]


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_csv(RESULTS_DIR / "seed_summary.csv")
    per_model = {m: df[df.model == m].set_index("seed").sort_index() for m in ["A", "F", "G"]}
    for m in ["A", "F", "G"]:
        assert list(per_model[m].index) == SEEDS, f"expected seeds 0,1,2 for model {m}"

    per_seed_rows, summary_rows, per_deg_rows = [], [], []

    for hi_m, lo_m in COMPARISONS:
        hi, lo = per_model[hi_m], per_model[lo_m]
        d_psnr = (hi["last5_mean_psnr"] - lo["last5_mean_psnr"]).values
        d_ssim = (hi["last5_mean_ssim"] - lo["last5_mean_ssim"]).values
        d_psnr_best = (hi["best_psnr"] - lo["best_psnr"]).values
        d_ssim_best = (hi["best_ssim"] - lo["best_ssim"]).values

        for seed in SEEDS:
            per_seed_rows.append({
                "comparison": f"{hi_m}-{lo_m}", "seed": seed,
                f"{hi_m}_last5_psnr": float(hi.loc[seed, "last5_mean_psnr"]),
                f"{lo_m}_last5_psnr": float(lo.loc[seed, "last5_mean_psnr"]),
                "delta_last5_psnr": float(hi.loc[seed, "last5_mean_psnr"] - lo.loc[seed, "last5_mean_psnr"]),
                f"{hi_m}_last5_ssim": float(hi.loc[seed, "last5_mean_ssim"]),
                f"{lo_m}_last5_ssim": float(lo.loc[seed, "last5_mean_ssim"]),
                "delta_last5_ssim": float(hi.loc[seed, "last5_mean_ssim"] - lo.loc[seed, "last5_mean_ssim"]),
                "delta_best_psnr": float(hi.loc[seed, "best_psnr"] - lo.loc[seed, "best_psnr"]),
                "delta_best_ssim": float(hi.loc[seed, "best_ssim"] - lo.loc[seed, "best_ssim"]),
            })

        ci_psnr = bootstrap_ci(d_psnr)
        ci_ssim = bootstrap_ci(d_ssim)
        ci_psnr_best = bootstrap_ci(d_psnr_best)
        ci_ssim_best = bootstrap_ci(d_ssim_best)
        for metric, mean_v, std_v, ci in [
            ("delta_last5_psnr", d_psnr.mean(), d_psnr.std(ddof=1), ci_psnr),
            ("delta_last5_ssim", d_ssim.mean(), d_ssim.std(ddof=1), ci_ssim),
            ("delta_best_psnr", d_psnr_best.mean(), d_psnr_best.std(ddof=1), ci_psnr_best),
            ("delta_best_ssim", d_ssim_best.mean(), d_ssim_best.std(ddof=1), ci_ssim_best),
        ]:
            summary_rows.append({"comparison": f"{hi_m}-{lo_m}", "metric": metric, "mean": mean_v, "std": std_v,
                                  "bootstrap_ci95_lo": ci[0], "bootstrap_ci95_hi": ci[1], "n_seeds": 3,
                                  "note": "EXPLORATORY (N=3 seeds) -- not a claim of statistical significance"})

        for deg in ["rain", "haze", "noise"]:
            psnr_col, ssim_col = f"last5_mean_{deg}_psnr", f"last5_mean_{deg}_ssim"
            dp = (hi[psnr_col] - lo[psnr_col]).values
            ds = (hi[ssim_col] - lo[ssim_col]).values
            for seed in SEEDS:
                per_deg_rows.append({
                    "comparison": f"{hi_m}-{lo_m}", "degradation": deg, "seed": seed,
                    f"{hi_m}_psnr": float(hi.loc[seed, psnr_col]), f"{lo_m}_psnr": float(lo.loc[seed, psnr_col]),
                    "delta_psnr": float(hi.loc[seed, psnr_col] - lo.loc[seed, psnr_col]),
                    f"{hi_m}_ssim": float(hi.loc[seed, ssim_col]), f"{lo_m}_ssim": float(lo.loc[seed, ssim_col]),
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
    print(per_seed_df[["comparison", "seed", "delta_last5_psnr", "delta_last5_ssim"]].to_string(index=False))
    print("\n=== Seed-level summary (mean +- std across 3 seeds, bootstrap 95% CI) ===")
    print(summary_df[["comparison", "metric", "mean", "std", "bootstrap_ci95_lo", "bootstrap_ci95_hi"]].to_string(index=False))
    print("\n=== Per-degradation summary (mean +- std across seeds) ===")
    print(per_deg_summary.to_string())

    for hi_m, lo_m in COMPARISONS:
        sub = per_seed_df[per_seed_df.comparison == f"{hi_m}-{lo_m}"]
        all_pos = (sub.delta_last5_psnr > 0).all()
        all_nonneg_ssim = (sub.delta_last5_ssim >= 0).all()
        all_nonpos = (sub.delta_last5_psnr <= 0).all()
        print(f"\n{hi_m}-{lo_m}: all 3 seeds delta_psnr>0: {all_pos}, all delta_ssim>=0: {all_nonneg_ssim}, "
              f"all delta_psnr<=0: {all_nonpos}")


if __name__ == "__main__":
    main()
