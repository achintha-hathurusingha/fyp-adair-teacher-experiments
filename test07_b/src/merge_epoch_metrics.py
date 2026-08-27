"""TEST07-B: merge the 6 per-run epoch_metrics_{model}_seed{seed}.csv and
seed_summary_{model}_seed{seed}.csv files (written separately by each
training process to avoid a shared-file write race) into the unified
epoch_metrics.csv and seed_summary.csv the rest of the pipeline expects.

Usage (on devon, adair-distill env, PINNED, run AFTER all 6 training runs
have finished):
  taskset -c 0-7,12-31 python merge_epoch_metrics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST07B = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST07B / "results"
MODELS = ["A", "B"]
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
    print(summary_df[["model", "seed", "final_psnr", "best_psnr", "last5_mean_psnr"]].to_string(index=False))


if __name__ == "__main__":
    main()
