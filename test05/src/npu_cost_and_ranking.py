"""TEST05 Phase 18-20: NPU cost estimation + final distillation candidate
ranking, synthesizing every prior phase's results. Reports every metric
independently (per instruction), then an EXPLICIT, documented composite
score with a stated weighting -- never letting classification accuracy
alone decide the ranking.

Usage: python npu_cost_and_ranking.py   (reads already-computed CSVs; run
either locally after pulling results, or on devon -- no GPU needed)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST05 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST05 / "results" / "statistics"
CHANNEL_DIR = TEST05 / "results" / "channel_analysis"
INTERVENTION_DIR = TEST05 / "results" / "intervention"
FREQ_DIR = TEST05 / "results" / "frequency_analysis"

# (feature, n_channels, spatial_h, spatial_w) for an example 480x320 input,
# per the established shape table (test01-04 audits)
CANDIDATES = {
    "latent_pre": (384, 40, 60),
    "AFLB1_aflb_out": (384, 40, 60),
    "AFLB2_aflb_out": (192, 80, 120),
    "AFLB3_aflb_out": (96, 160, 240),
    "AFLB1_mined_low": (384, 40, 60),
    "AFLB1_mined_high": (384, 40, 60),
    "AFLB1_fmom_agg": (384, 40, 60),
    "alpha_beta_all": (6, 1, 1),
    "top_10pct_latent_channels": (38, 40, 60),  # ~10% of 384
}


def npu_cost_row(name, c, h, w):
    n_elements = c * h * w
    return {
        "candidate": name, "channels": c, "spatial_h": h, "spatial_w": w,
        "n_elements": n_elements,
        "fp32_kb": n_elements * 4 / 1024, "fp16_kb": n_elements * 2 / 1024, "int8_kb": n_elements * 1 / 1024,
    }


def main():
    npu_rows = [npu_cost_row(name, *dims) for name, dims in CANDIDATES.items()]
    npu_df = pd.DataFrame(npu_rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    npu_df.to_csv(STATS_DIR / "npu_cost.csv", index=False)
    print(f"wrote {STATS_DIR / 'npu_cost.csv'}")
    print(npu_df.to_string(index=False))

    # ---- gather evidence from every phase for the final ranking ----
    probe = pd.read_csv(STATS_DIR / "linear_probe.csv").set_index("feature")
    sens = pd.read_csv(STATS_DIR / "scene_sensitivity.csv")
    sens_e = sens[sens.metric == "euclidean"].set_index("feature")
    chan_rank = pd.read_csv(CHANNEL_DIR / "channel_rank.csv") if (CHANNEL_DIR / "channel_rank.csv").exists() else pd.DataFrame()
    group_interv = pd.read_csv(INTERVENTION_DIR / "channel_group_intervention.csv") if (INTERVENTION_DIR / "channel_group_intervention.csv").exists() else pd.DataFrame()
    freq_summary = pd.read_csv(FREQ_DIR / "frequency_summary.csv") if (FREQ_DIR / "frequency_summary.csv").exists() else pd.DataFrame()
    compact = pd.read_csv(STATS_DIR / "compact_embedding.csv") if (STATS_DIR / "compact_embedding.csv").exists() else pd.DataFrame()

    candidates = ["latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out",
                  "AFLB1_mined_low", "AFLB1_mined_high", "AFLB1_fmom_agg", "AFLB1_raw_high",
                  "AFLB1_lh_channel_weight"]
    rows = []
    for c in candidates:
        acc = probe.loc[c, "accuracy_mean"] if c in probe.index else np.nan
        ratio = sens_e.loc[c, "ratio"] if c in sens_e.index else np.nan
        n_dim = probe.loc[c, "n_dim"] if c in probe.index else np.nan

        # causal effect: from TEST04 (full tensor, latent_pre / aflb_out only) as reference
        causal_ref = {"latent_pre": 14.17, "AFLB1_aflb_out": 14.18, "AFLB2_aflb_out": 32.35,
                       "AFLB3_aflb_out": 53.94}.get(c, np.nan)

        freq_note = ""
        if not freq_summary.empty and c in freq_summary.feature.values:
            sub = freq_summary[freq_summary.feature == c]
            freq_note = "measured (see Frequency_Analysis sheet)"

        rows.append({
            "candidate": c, "degradation_accuracy_pct": acc * 100 if pd.notna(acc) else np.nan,
            "degradation_scene_ratio": ratio, "causal_effect_L2_full_tensor": causal_ref,
            "n_dim_pooled": n_dim, "frequency_analyzed": freq_note != "",
        })

    # top channel subset summary (from group intervention, if available)
    if not group_interv.empty:
        for pct in sorted(group_interv[group_interv.group_type == "top_degradation_specific"].pct.unique()):
            sub = group_interv[(group_interv.group_type == "top_degradation_specific") & (group_interv.pct == pct)]
            full = group_interv[group_interv.group_type == "full_tensor"]
            rand = group_interv[(group_interv.group_type == "random_same_size") & (group_interv.pct == pct)]
            frac_of_full = sub.l2_vs_normal.mean() / full.l2_vs_normal.mean() if len(full) else np.nan
            rows.append({
                "candidate": f"top_{int(pct*100)}pct_latent_channels",
                "degradation_accuracy_pct": np.nan, "degradation_scene_ratio": np.nan,
                "causal_effect_L2_full_tensor": sub.l2_vs_normal.mean(),
                "n_dim_pooled": int(round(384 * pct)) * 2,
                "frequency_analyzed": False,
                "pct_of_full_tensor_effect": frac_of_full * 100,
                "vs_random_same_size_L2": rand.l2_vs_normal.mean() if len(rand) else np.nan,
            })

    if not compact.empty:
        for _, r in compact.iterrows():
            if r["representation"].startswith("pca_"):
                rows.append({
                    "candidate": r["representation"], "degradation_accuracy_pct": r["accuracy"] * 100,
                    "degradation_scene_ratio": np.nan, "causal_effect_L2_full_tensor": np.nan,
                    "n_dim_pooled": r["dim"], "frequency_analyzed": False,
                })
            elif "alpha_beta" in r["representation"]:
                rows.append({
                    "candidate": r["representation"], "degradation_accuracy_pct": r["accuracy"] * 100,
                    "degradation_scene_ratio": np.nan, "causal_effect_L2_full_tensor": np.nan,
                    "n_dim_pooled": r["dim"], "frequency_analyzed": False,
                })

    df = pd.DataFrame(rows)

    # ---- explicit composite score, documented, with sensitivity analysis ----
    # normalize each metric to [0,1] (min-max over the candidates that have it), then weight.
    def norm(col):
        s = df[col]
        if s.notna().sum() < 2:
            return pd.Series(np.nan, index=df.index)
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    weights = {"degradation_accuracy_pct": 0.25, "degradation_scene_ratio": 0.35,
               "causal_effect_L2_full_tensor": 0.20}
    # compactness bonus: smaller n_dim_pooled is better -> invert
    n_dim_norm = 1 - norm("n_dim_pooled")
    weights_compact = 0.20

    composite = (
        weights["degradation_accuracy_pct"] * norm("degradation_accuracy_pct").fillna(0) +
        weights["degradation_scene_ratio"] * norm("degradation_scene_ratio").fillna(0) +
        weights["causal_effect_L2_full_tensor"] * norm("causal_effect_L2_full_tensor").fillna(0) +
        weights_compact * n_dim_norm.fillna(0)
    )
    df["composite_score"] = composite
    df = df.sort_values("composite_score", ascending=False)
    df.to_csv(STATS_DIR / "distillation_candidate_ranking.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'distillation_candidate_ranking.csv'}")
    print(f"Composite weights: accuracy=0.25, degradation_scene_ratio=0.35, causal_effect=0.20, compactness=0.20")
    print(df.to_string(index=False))

    # sensitivity check: rank with degradation_scene_ratio weight doubled
    composite_alt = (
        0.15 * norm("degradation_accuracy_pct").fillna(0) +
        0.55 * norm("degradation_scene_ratio").fillna(0) +
        0.15 * norm("causal_effect_L2_full_tensor").fillna(0) +
        0.15 * n_dim_norm.fillna(0)
    )
    df["composite_score_ratio_weighted"] = composite_alt
    top5_default = df.sort_values("composite_score", ascending=False).head(5)["candidate"].tolist()
    top5_alt = df.sort_values("composite_score_ratio_weighted", ascending=False).head(5)["candidate"].tolist()
    print(f"\nSensitivity check -- top 5 default weighting: {top5_default}")
    print(f"Sensitivity check -- top 5 with degradation_scene_ratio weight doubled: {top5_alt}")
    print(f"Stable candidates in both top-5s: {set(top5_default) & set(top5_alt)}")
    df.to_csv(STATS_DIR / "distillation_candidate_ranking.csv", index=False)


if __name__ == "__main__":
    main()
