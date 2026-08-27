"""TEST10-R: merge per-run epoch_metrics/seed_summary/collapse_monitor CSVs.

Usage (on devon, adair-distill env, PINNED, run AFTER all 9 runs finish):
  taskset -c 0-7,12-31 python merge_epoch_metrics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST10R = Path(__file__).resolve().parent.parent
RESULTS_DIR = TEST10R / "results"
MODELS = ["A", "F", "G"]
SEEDS = [0, 1, 2]


def main():
    epoch_dfs, summary_dfs, collapse_dfs = [], [], []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            epoch_path = RESULTS_DIR / f"epoch_metrics_{model}_seed{seed}.csv"
            summary_path = RESULTS_DIR / f"seed_summary_{model}_seed{seed}.csv"
            collapse_path = RESULTS_DIR / f"collapse_monitor_{model}_seed{seed}.csv"
            if not epoch_path.exists() or not summary_path.exists():
                missing.append((model, seed))
                continue
            epoch_dfs.append(pd.read_csv(epoch_path))
            summary_dfs.append(pd.read_csv(summary_path))
            if collapse_path.exists():
                collapse_dfs.append(pd.read_csv(collapse_path))

    if missing:
        print(f"WARNING: missing per-run files for {missing}")

    epoch_df = pd.concat(epoch_dfs, ignore_index=True)
    epoch_df.to_csv(RESULTS_DIR / "epoch_metrics.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'epoch_metrics.csv'}: {len(epoch_df)} rows "
          f"({epoch_df.groupby(['model', 'seed']).ngroups} model/seed combos)")

    summary_df = pd.concat(summary_dfs, ignore_index=True)
    summary_df.to_csv(RESULTS_DIR / "seed_summary.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'seed_summary.csv'}: {len(summary_df)} rows")
    print(summary_df[["model", "seed", "aborted", "n_epochs_completed", "final_psnr", "best_psnr",
                       "last5_mean_psnr"]].to_string(index=False))

    if collapse_dfs:
        collapse_df = pd.concat(collapse_dfs, ignore_index=True)
        collapse_df.to_csv(RESULTS_DIR / "collapse_monitor.csv", index=False)
        print(f"wrote {RESULTS_DIR / 'collapse_monitor.csv'}: {len(collapse_df)} rows")
        any_collapsed = collapse_df["collapsed"].any()
        print(f"Any collapse detected during training (any epoch, any stage, any seed): {any_collapsed}")


if __name__ == "__main__":
    main()
