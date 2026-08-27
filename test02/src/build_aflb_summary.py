"""Phase 12: focused AFLB-internal summary -- which of mined_low/mined_high/
H-L/L-H/FMoM-agg/AFLB-output contains the most degradation-discriminative
information? Filters/joins the already-computed linear_probe_results.csv
and degradation_separation.csv rather than re-running anything.

Usage: python build_aflb_summary.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST02 = Path(__file__).resolve().parent.parent
CLASSIFIERS_DIR = TEST02 / "results" / "classifiers"
STATS_DIR = TEST02 / "results" / "statistics"

AFLB_KEYS_OF_INTEREST = ["mined_low", "mined_high", "hl_spatial_weight", "lh_channel_weight",
                          "fmom_agg", "aflb_out", "raw_low", "raw_high", "y_in", "cross_agg_out"]


def main():
    probe = pd.read_csv(CLASSIFIERS_DIR / "linear_probe_results.csv")
    sep = pd.read_csv(STATS_DIR / "degradation_separation.csv")

    probe_aflb = probe[probe.feature.str.startswith("AFLB") & (probe.classifier == "logreg")].copy()
    probe_aflb["AFLB"] = probe_aflb.feature.str.extract(r"(AFLB\d)")
    probe_aflb["sub_feature"] = probe_aflb.feature.str.replace(r"AFLB\d_", "", regex=True)

    sep_aflb = sep[sep.feature.str.startswith("AFLB") & (sep.metric == "euclidean")].copy()
    sep_aflb["AFLB"] = sep_aflb.feature.str.extract(r"(AFLB\d)")
    sep_aflb["sub_feature"] = sep_aflb.feature.str.replace(r"AFLB\d_", "", regex=True)

    merged = probe_aflb.merge(
        sep_aflb[["AFLB", "sub_feature", "separation_ratio"]], on=["AFLB", "sub_feature"], how="left")
    merged = merged[merged.sub_feature.isin(AFLB_KEYS_OF_INTEREST)]
    merged = merged.sort_values("accuracy_mean", ascending=False)
    merged = merged[["AFLB", "sub_feature", "n_dim", "accuracy_mean", "accuracy_std",
                      "balanced_accuracy", "macro_f1", "separation_ratio"]]

    out_path = TEST02 / "results" / "classifiers" / "aflb_analysis.csv"
    merged.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(merged)} rows)")
    print("\nRanked by classification accuracy (which AFLB sub-feature is most discriminative):")
    print(merged.to_string(index=False))

    print("\nBy sub-feature type, averaged across AFLB1/2/3:")
    print(merged.groupby("sub_feature")["accuracy_mean"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
