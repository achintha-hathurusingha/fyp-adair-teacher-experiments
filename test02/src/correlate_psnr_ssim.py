"""Phase 14: exploratory correlation between feature statistics and
restoration performance (PSNR/SSIM). Not causal -- just correlation.

For every feature, correlates its scalar stats (mean/std/L2/energy from
feature_statistics.csv) against PSNR and SSIM (psnr_ssim.csv), per image.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python correlate_psnr_ssim.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TEST02 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST02 / "results" / "statistics"

STAT_COLS = ["mean", "std", "L1", "L2", "energy"]


def main():
    feat = pd.read_csv(STATS_DIR / "feature_statistics.csv")
    perf = pd.read_csv(STATS_DIR / "psnr_ssim.csv")
    merged = feat.merge(perf[["image_id", "psnr", "ssim"]], on="image_id")

    rows = []
    for feature_name, g in merged.groupby("feature_name"):
        for stat_col in STAT_COLS:
            x = g[stat_col].to_numpy(dtype=float)
            if np.std(x) == 0 or len(x) < 3:
                continue
            for target in ["psnr", "ssim"]:
                y = g[target].to_numpy(dtype=float)
                pear_r, pear_p = pearsonr(x, y)
                spear_r, spear_p = spearmanr(x, y)
                rows.append({
                    "feature": feature_name, "statistic": stat_col, "target": target,
                    "pearson_r": pear_r, "pearson_p": pear_p,
                    "spearman_r": spear_r, "spearman_p": spear_p, "n": len(x),
                })

    out = pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    out.to_csv(STATS_DIR / "psnr_ssim_correlation.csv", index=False)
    print(f"wrote {STATS_DIR / 'psnr_ssim_correlation.csv'} ({len(out)} rows)")
    print("\nTop 15 |pearson_r| (exploratory, not causal):")
    print(out.head(15)[["feature", "statistic", "target", "pearson_r", "pearson_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
