"""TEST03 Phase 15: focused AFLB-internal summary (mirrors test02's
approach, using the grouped-CV linear_probe_results.csv + degradation_vs_scene.csv).

TEST03 Phase 16: representation-swap PREPARATION (not execution). Confirms
that latent + AFLB1/2/3 outputs are already saved as raw tensors for 5+
representative scenes x 3 degradations (part of the 10-scene tensor set
extract_features.py already wrote), and writes a dedicated index +
documentation note explicitly flagging this as future-intervention
infrastructure, per the task's instruction NOT to perform the actual
swap/intervention in TEST03.

Usage: python build_aflb_and_swapprep.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST03 = Path(__file__).resolve().parent.parent
CLASSIFIERS_DIR = TEST03 / "results" / "classifiers"
STATS_DIR = TEST03 / "results" / "statistics"
TENSORS_DIR = TEST03 / "results" / "tensors"

AFLB_KEYS_OF_INTEREST = ["mined_low", "mined_high", "hl_spatial_weight", "lh_channel_weight",
                          "fmom_agg", "aflb_out", "raw_low", "raw_high", "y_in", "cross_agg_out"]
SWAP_PREP_FEATURES = ["latent", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]
N_SWAP_SCENES = 5


def build_aflb_summary():
    probe = pd.read_csv(CLASSIFIERS_DIR / "linear_probe_results.csv")
    sep = pd.read_csv(STATS_DIR / "degradation_vs_scene.csv")

    probe_aflb = probe[probe.feature.str.startswith("AFLB") & (probe.classifier == "logreg")].copy()
    probe_aflb["AFLB"] = probe_aflb.feature.str.extract(r"(AFLB\d)")
    probe_aflb["sub_feature"] = probe_aflb.feature.str.replace(r"AFLB\d_", "", regex=True)

    sep_aflb = sep[sep.feature.str.startswith("AFLB") & (sep.metric == "euclidean")].copy()
    sep_aflb["AFLB"] = sep_aflb.feature.str.extract(r"(AFLB\d)")
    sep_aflb["sub_feature"] = sep_aflb.feature.str.replace(r"AFLB\d_", "", regex=True)

    merged = probe_aflb.merge(
        sep_aflb[["AFLB", "sub_feature", "degradation_ratio"]], on=["AFLB", "sub_feature"], how="left")
    merged = merged[merged.sub_feature.isin(AFLB_KEYS_OF_INTEREST)]
    merged = merged.sort_values("accuracy_mean", ascending=False)
    merged = merged[["AFLB", "sub_feature", "n_dim", "accuracy_mean", "accuracy_std",
                      "balanced_accuracy", "macro_f1", "degradation_ratio"]]

    out_path = CLASSIFIERS_DIR / "aflb_analysis.csv"
    merged.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(merged)} rows)")
    print(merged.to_string(index=False))
    print("\nBy sub-feature type, averaged across AFLB1/2/3 (grouped-CV accuracy):")
    print(merged.groupby("sub_feature")["accuracy_mean"].mean().sort_values(ascending=False).to_string())


def build_swap_prep_index():
    tensor_index = pd.read_csv(TENSORS_DIR / "tensor_index.csv")
    swap_rows = tensor_index[tensor_index.feature_name.isin(SWAP_PREP_FEATURES)].copy()
    scenes = sorted(swap_rows.scene_id.unique())[:N_SWAP_SCENES]
    swap_rows = swap_rows[swap_rows.scene_id.isin(scenes)]
    out_path = TENSORS_DIR / "representation_swap_prep_index.csv"
    swap_rows.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(swap_rows)} rows) -- "
          f"{len(scenes)} scenes x {len(SWAP_PREP_FEATURES)} features x 3 degradations, "
          f"PREPARED for a future representation-swap intervention (NOT executed in TEST03).")
    print(f"scenes: {scenes}")


if __name__ == "__main__":
    build_aflb_summary()
    build_swap_prep_index()
