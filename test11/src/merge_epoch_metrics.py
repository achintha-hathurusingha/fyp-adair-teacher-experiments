"""TEST11: merge the 15 per-run epoch_metrics/seed_summary CSVs.

Usage (on devon, adair-distill env, PINNED, run AFTER all 15 runs finish):
  taskset -c 0-7,12-31 python merge_epoch_metrics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST11 = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST11 / "results"
MODELS = ["A", "F2", "F4", "F8", "F16"]
SEEDS = [0, 1, 2]


def main():
    epoch_dfs, summary_dfs = [], []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            epoch_path = RESULTS_DIR / f"epoch_metrics_{model}_seed{seed}.csv"
            summary_path = RESULTS_DIR / f"seed_summary_{model}_seed{seed}.csv"
            if not epoch_path.exists() or not summary_path.exists():
                missing.append((model, seed))
                continue
            epoch_dfs.append(pd.read_csv(epoch_path))
            summary_dfs.append(pd.read_csv(summary_path))

    if missing:
        print(f"WARNING: missing per-run files for {missing}")

    epoch_df = pd.concat(epoch_dfs, ignore_index=True)
    epoch_df.to_csv(RESULTS_DIR / "epoch_metrics.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'epoch_metrics.csv'}: {len(epoch_df)} rows "
          f"({epoch_df.groupby(['model', 'seed']).ngroups} model/seed combos)")

    summary_df = pd.concat(summary_dfs, ignore_index=True)
    summary_df.to_csv(RESULTS_DIR / "seed_summary.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'seed_summary.csv'}: {len(summary_df)} rows")
    print(summary_df[["model", "seed", "final_psnr", "best_psnr", "last5_mean_psnr", "any_nan_or_inf",
                       "params"]].to_string(index=False))
    if summary_df["any_nan_or_inf"].any():
        print("\nWARNING: at least one run reported NaN/Inf during training -- inspect before trusting results.")


if __name__ == "__main__":
    main()
