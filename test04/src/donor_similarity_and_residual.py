"""TEST04 Phase 12 (donor-behavior similarity) + Phase 13 (residual
analysis) -- aggregates the per-intervention distances already computed
inline in run_interventions.py (l2_vs_normal_recipient, l2_vs_normal_donor,
l2_vs_normal_<third>, residual_* columns) into summary tables answering:
is the swapped output closer to the recipient's own normal output, the
donor's normal output, or neither?

Usage: python donor_similarity_and_residual.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST04 = Path(__file__).resolve().parent.parent
INTERVENTIONS_DIR = TEST04 / "results" / "interventions"
STATS_DIR = TEST04 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]


def main():
    df = pd.read_csv(INTERVENTIONS_DIR / "cross_degradation_swaps.csv")

    rows = []
    for _, r in df.iterrows():
        third = [d for d in DEGS if d not in (r["recipient"], r["donor"])][0]
        l2_third_col = f"l2_vs_normal_{third.lower()}"
        distances = {"recipient": r["l2_vs_normal_recipient"], "donor": r["l2_vs_normal_donor"],
                     "third": r[l2_third_col] if l2_third_col in r else np.nan}
        closest = min(distances, key=lambda k: distances[k] if not np.isnan(distances[k]) else np.inf)
        rows.append({
            "scene_id": r["scene_id"], "point": r["point"], "recipient": r["recipient"], "donor": r["donor"],
            "l2_to_recipient_normal": distances["recipient"], "l2_to_donor_normal": distances["donor"],
            "l2_to_third_normal": distances["third"], "closest_to": closest,
        })
    sim_df = pd.DataFrame(rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(STATS_DIR / "donor_similarity.csv", index=False)
    print(f"wrote {STATS_DIR / 'donor_similarity.csv'} ({len(sim_df)} rows)")

    summary = sim_df.groupby(["point", "recipient", "donor"])["closest_to"].value_counts(normalize=True).unstack(fill_value=0) * 100
    summary.columns = [f"pct_closest_to_{c}" for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(STATS_DIR / "donor_similarity_summary.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'donor_similarity_summary.csv'}")
    print(summary.to_string(index=False))

    overall = sim_df.groupby("point")["closest_to"].value_counts(normalize=True).unstack(fill_value=0) * 100
    print("\nOverall (%, by intervention point) -- is swapped output closest to recipient-normal, donor-normal, or third-normal?")
    print(overall.to_string())

    # ---- Residual analysis ----
    residual_cols = ["residual_mean", "residual_std", "residual_energy", "residual_mae"]
    res_summary = df.groupby(["point", "recipient", "donor"])[residual_cols].mean().reset_index()
    res_summary.to_csv(STATS_DIR / "residual_summary.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'residual_summary.csv'}")

    # compare against normal (non-swapped) residuals
    normal = pd.read_csv(TEST04 / "results" / "metrics" / "normal_baseline.csv")
    print("\nNormal-inference residual proxy (from PSNR/SSIM/MSE, for reference):")
    print(normal.groupby("degradation")[["psnr", "ssim", "mse"]].mean().to_string())


if __name__ == "__main__":
    main()
