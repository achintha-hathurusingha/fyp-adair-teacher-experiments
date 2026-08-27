"""TEST06-R Phases 6-8: paired statistical testing, practical equivalence,
and per-degradation/per-donor-direction breakdowns. CPU-only, reads the
CSVs produced by balanced_intervention.py.

Phase 6 -- paired statistics: unit of pairing is (scene_id, recipient,
donor) -- the exact row index shared by primary_swap.csv and (expanded)
balanced_controls.csv. Bootstrap resamples at the SCENE level (24 scenes,
scene_021 excluded -- see report/rerun_audit.md -- each contributing up to
6 rows), 10,000 resamples, seed=0 (documented).

Phase 7 -- practical equivalence: epsilon is defined PURELY from
control-vs-control pairwise differences (6 pairs x N matched rows),
computed BEFORE looking at any primary-vs-control comparison, per the
task's explicit instruction not to manufacture epsilon after seeing the
result.

Phase 8 -- per-degradation (recipient) and per-donor-direction breakdowns.

Usage (local machine or devon, CPU-only):
  python paired_statistics.py
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

TEST06_R = Path(__file__).resolve().parent.parent
BC_DIR = TEST06_R / "results" / "balanced_controls"
PROP_DIR = TEST06_R / "results" / "internal_propagation"
STAT_DIR = TEST06_R / "results" / "statistics"
STAT_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_TYPES = ["cross_scene_same_degradation", "random_matched", "zero", "global_mean"]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0
METRIC = "normalized_l2"


def load_matched():
    primary = pd.read_csv(BC_DIR / "primary_swap.csv")
    controls = pd.read_csv(BC_DIR / "balanced_controls.csv")
    primary["key"] = primary.scene_id + "_" + primary.recipient + "_" + primary.donor
    controls["key"] = controls.scene_id + "_" + controls.recipient + "_" + controls.donor
    return primary, controls


def bootstrap_ci_scene_level(df_matched, value_col, scene_col="scene_id", n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    scenes = df_matched[scene_col].unique()
    n_scenes = len(scenes)
    boot_means = np.empty(n_boot)
    grouped = {s: df_matched[df_matched[scene_col] == s][value_col].values for s in scenes}
    for b in range(n_boot):
        resampled_scenes = rng.choice(scenes, size=n_scenes, replace=True)
        vals = np.concatenate([grouped[s] for s in resampled_scenes])
        boot_means[b] = vals.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_lo), float(ci_hi), boot_means


def paired_permutation_test(diffs, n_perm=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    observed = diffs.mean()
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    perm_means = (signs * diffs.values).mean(axis=1)
    p = float((np.abs(perm_means) >= np.abs(observed)).mean())
    return p


def phase6_paired_statistics(primary, controls):
    rows = []
    diff_records = {}
    for ct in CONTROL_TYPES:
        c = controls[controls.control == ct][["key", "scene_id", METRIC]].rename(columns={METRIC: "control_val"})
        m = primary[["key", "scene_id", METRIC]].rename(columns={METRIC: "primary_val"}).merge(
            c[["key", "control_val"]], on="key")
        m["D"] = m.primary_val - m.control_val
        diff_records[ct] = m

        ci_lo, ci_hi, _ = bootstrap_ci_scene_level(m, "D")
        p_perm = paired_permutation_test(m["D"])
        try:
            wilcoxon_stat, p_wilcoxon = stats.wilcoxon(m.primary_val, m.control_val)
        except ValueError:
            wilcoxon_stat, p_wilcoxon = float("nan"), float("nan")

        rows.append({
            "control": ct, "n": len(m),
            "mean_diff": float(m.D.mean()), "median_diff": float(m.D.median()), "std_diff": float(m.D.std()),
            "bootstrap_ci95_lo": ci_lo, "bootstrap_ci95_hi": ci_hi,
            "paired_permutation_p": p_perm, "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(p_wilcoxon),
        })
    df = pd.DataFrame(rows)
    df.to_csv(STAT_DIR / "paired_statistics.csv", index=False)
    print("Phase 6 -- paired statistics (primary - control, normalized_l2):")
    print(df.to_string(index=False))
    return df, diff_records


def phase7_practical_equivalence(controls):
    pooled_abs_diffs = []
    controls_wide = controls.pivot_table(index="key", columns="control", values=METRIC)
    for a, b in combinations(CONTROL_TYPES, 2):
        diffs = (controls_wide[a] - controls_wide[b]).abs().dropna()
        pooled_abs_diffs.extend(diffs.tolist())
    epsilon = float(np.percentile(pooled_abs_diffs, 95))
    print(f"\nPhase 7 -- epsilon (95th percentile of |control-vs-control| differences, "
          f"n={len(pooled_abs_diffs)} pooled pairwise diffs, computed BEFORE any primary comparison): "
          f"{epsilon:.8f}")
    with open(STAT_DIR / "epsilon_definition.txt", "w") as f:
        f.write(f"epsilon = 95th percentile of |control_A - control_B| across all "
                f"{len(list(combinations(CONTROL_TYPES, 2)))} control-type pairs x 150 matched rows "
                f"({len(pooled_abs_diffs)} pooled values) = {epsilon:.8f}\n"
                f"Computed from control-vs-control comparisons ONLY, before any primary-vs-control "
                f"comparison was examined, per the task's explicit instruction.\n")
    return epsilon


def classify_equivalence(mean_diff, ci_lo, epsilon):
    if abs(mean_diff) <= epsilon:
        return "practically equivalent"
    if mean_diff > epsilon and ci_lo > 0:
        return "materially larger"
    return "inconclusive"


def phase8_breakdowns(primary):
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    per_recipient = primary.groupby("recipient")[METRIC].agg(["mean", "median", "std", "count"])
    per_recipient.to_csv(STAT_DIR / "per_degradation.csv")
    print("\nPhase 8a -- per-recipient-degradation breakdown:")
    print(per_recipient.to_string())

    primary["direction"] = primary.recipient + "<-" + primary.donor
    per_direction = primary.groupby("direction")[METRIC].agg(["mean", "median", "std", "count"])
    per_direction.to_csv(STAT_DIR / "per_donor_direction.csv")
    print("\nPhase 8b -- per-donor-direction breakdown:")
    print(per_direction.to_string())
    return per_recipient, per_direction


def phase9_donor_behavior():
    prop = pd.read_csv(PROP_DIR / "propagation_compact_stats.csv")
    final = prop[prop.stage == "final_output"].copy()
    final["moved_toward_donor"] = final.normalized_dist_to_donor < final.normalized_dist_to_recipient
    final["closeness_gap"] = final.normalized_dist_to_recipient - final.normalized_dist_to_donor  # >0 means moved toward donor

    pct = float(final.moved_toward_donor.mean() * 100)
    mean_gap = float(final.closeness_gap.mean())
    ci_lo, ci_hi, _ = bootstrap_ci_scene_level(final, "closeness_gap")
    print(f"\nPhase 9 -- donor-behavior: {pct:.1f}% of {len(final)} swaps moved toward donor; "
          f"mean(d_recipient - d_donor) = {mean_gap:.8f}, scene-level bootstrap 95% CI = [{ci_lo:.8f}, {ci_hi:.8f}]")
    final[["scene_id", "recipient", "donor", "normalized_dist_to_recipient", "normalized_dist_to_donor",
           "moved_toward_donor", "closeness_gap"]].to_csv(STAT_DIR / "donor_behavior_stats.csv", index=False)
    with open(STAT_DIR / "donor_behavior_summary.txt", "w") as f:
        f.write(f"percent_moved_toward_donor: {pct:.4f}\n")
        f.write(f"mean_closeness_gap: {mean_gap:.8f}\n")
        f.write(f"bootstrap_ci95: [{ci_lo:.8f}, {ci_hi:.8f}]\n")
    return final


def main():
    primary, controls = load_matched()
    stats_df, _ = phase6_paired_statistics(primary, controls)
    epsilon = phase7_practical_equivalence(controls)

    stats_df["equivalence_classification"] = stats_df.apply(
        lambda row: classify_equivalence(row.mean_diff, row.bootstrap_ci95_lo, epsilon),
        axis=1)
    stats_df.to_csv(STAT_DIR / "paired_statistics.csv", index=False)
    print("\nPhase 7 classification (mean_diff vs epsilon, CI-based):")
    print(stats_df[["control", "mean_diff", "bootstrap_ci95_lo", "bootstrap_ci95_hi",
                     "equivalence_classification"]].to_string(index=False))

    phase8_breakdowns(primary)
    phase9_donor_behavior()
    print(f"\nwrote statistics to {STAT_DIR}")


if __name__ == "__main__":
    main()
