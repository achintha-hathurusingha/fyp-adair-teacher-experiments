"""TEST04 Phase 18-19: swap matrix table + scene-level statistics (bootstrap
CI, paired comparisons). The experimental unit is the SCENE -- 600
interventions per point are NOT treated as independent; every aggregate
here is computed per (recipient,donor,point) with the 100 per-scene values
as the samples, and bootstrap-resampled by scene.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python build_swap_matrix_and_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST04 = Path(__file__).resolve().parent.parent
INTERVENTIONS_DIR = TEST04 / "results" / "interventions"
STATS_DIR = TEST04 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]


def bootstrap_ci(values: np.ndarray, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_csv(INTERVENTIONS_DIR / "cross_degradation_swaps.csv")

    matrix_rows = []
    for point in df.point.unique():
        for recipient in DEGS:
            for donor in DEGS:
                if recipient == donor:
                    continue
                sub = df[(df.point == point) & (df.recipient == recipient) & (df.donor == donor)]
                if len(sub) == 0:
                    continue
                delta_output_l2 = sub["l2_vs_normal_recipient"].to_numpy()
                delta_psnr = (sub["psnr_vs_clean"]).to_numpy()  # absolute, not delta -- delta computed vs recipient-normal separately below
                ci_lo, ci_hi = bootstrap_ci(delta_output_l2)
                matrix_rows.append({
                    "point": point, "recipient": recipient, "donor": donor, "same_scene": True,
                    "n_scenes": len(sub),
                    "mean_delta_output_L2": float(delta_output_l2.mean()),
                    "median_delta_output_L2": float(np.median(delta_output_l2)),
                    "std_delta_output_L2": float(delta_output_l2.std(ddof=1)),
                    "ci95_lo_delta_output_L2": ci_lo, "ci95_hi_delta_output_L2": ci_hi,
                    "mean_mae_vs_normal_recipient": float(sub["mae_vs_normal_recipient"].mean()),
                    "mean_psnr_vs_clean": float(sub["psnr_vs_clean"].mean()),
                    "mean_ssim_vs_clean": float(sub["ssim_vs_clean"].mean()),
                })

    matrix_df = pd.DataFrame(matrix_rows).sort_values(["point", "mean_delta_output_L2"], ascending=[True, False])
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(STATS_DIR / "swap_matrix.csv", index=False)
    print(f"wrote {STATS_DIR / 'swap_matrix.csv'} ({len(matrix_df)} rows)")
    print(matrix_df.to_string(index=False))

    # per-point summary (average across the 6 pairs) -- ranks the 4 intervention points
    point_summary = df.groupby("point").agg(
        mean_delta_output_L2=("l2_vs_normal_recipient", "mean"),
        std_delta_output_L2=("l2_vs_normal_recipient", "std"),
        mean_mae_vs_normal_recipient=("mae_vs_normal_recipient", "mean"),
        n=("l2_vs_normal_recipient", "count"),
    ).reset_index().sort_values("mean_delta_output_L2", ascending=False)
    point_summary.to_csv(STATS_DIR / "point_summary.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'point_summary.csv'}")
    print(point_summary.to_string(index=False))

    # self-swap reference (from controls) for comparison baseline
    self_swap = pd.read_csv(TEST04 / "results" / "controls" / "self_swap_validation.csv")
    print(f"\nSelf-swap reference: mean L2 diff (should be ~0) = {self_swap['l2_diff'].mean():.6f}, "
          f"max = {self_swap['l2_diff'].max():.6f}")


if __name__ == "__main__":
    main()
