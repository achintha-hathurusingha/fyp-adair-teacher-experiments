"""TEST06 Phase 3-4 (06-A conclusion): determine R_first per AFLB -- the
first input resolution at which the frequency mask becomes GENUINELY
non-zero, using an explicit, documented threshold (not "one accidental
non-zero pixel").

Threshold definition (documented, not assumed, and corrected after
inspecting real data): the mask half-width h_/w_ are computed as
`int(floor(feat/128) * threshold)` -- an INTEGER, not a floating-point
quantity subject to rounding noise. A mask box is therefore either
genuinely absent (h_=w_=0, mask_active_fraction EXACTLY 0.0) or genuinely
present (h_>=1 or w_>=1, mask_active_fraction > 0.0), with no ambiguous
near-zero regime to threshold away. The correct criterion is simply
`mask_active_fraction > 0` (strict), which the AFLB3 data confirms is
sufficient: even a 1-pixel half-width box carries a large share (>80%) of
the total raw_low+raw_high energy (natural images concentrate most energy
near DC), which is reported alongside as corroborating evidence, not as an
additional gating threshold.

Also applies Phase 4's gate decision: if NO resolution in the tested range
shows meaningful activation at ANY AFLB, report that plainly and recommend
stopping rather than continuing to 06-E.

Usage (local or devon, CPU-only, reads mask_activation.csv):
  python first_activation_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEST06 = Path(__file__).resolve().parent.parent
RESULTS = TEST06 / "results" / "resolution_sweep"
AFLBS = ["AFLB1", "AFLB2", "AFLB3"]


def main():
    df = pd.read_csv(RESULTS / "mask_activation.csv")
    grid = df[df.source.str.startswith("div2k_")].copy()  # controlled grid only, not native single-points
    grid["resolution"] = pd.to_numeric(grid["resolution"], errors="coerce")  # was read as str (mixed w/ "native")

    rows = []
    for aflb in AFLBS:
        active_col = f"{aflb}_mask_active_fraction"
        energy_col = f"{aflb}_raw_low_energy_fraction"
        feat_h_col = f"{aflb}_feat_h"
        is_active = grid[active_col] > 0  # strict: h_/w_ are integers, no floating-point ambiguity (see docstring)
        active_rows = grid[is_active]
        if len(active_rows) == 0:
            rows.append({"aflb": aflb, "first_activation_input_resolution": None,
                         "first_activation_feat_resolution": None,
                         "mask_active_fraction_at_first": None, "n_activating_configs": 0,
                         "n_total_configs": len(grid), "status": "NEVER ACTIVATES IN TESTED RANGE"})
            continue
        first_res = active_rows["resolution"].min()
        at_first = active_rows[active_rows.resolution == first_res].iloc[0]
        rows.append({
            "aflb": aflb, "first_activation_input_resolution": int(first_res),
            "first_activation_feat_resolution": int(at_first[feat_h_col]),
            "mask_active_fraction_at_first": float(at_first[active_col]),
            "raw_low_energy_fraction_at_first": float(at_first[energy_col]),
            "n_activating_configs": int(is_active.sum()), "n_total_configs": len(grid),
            "status": "ACTIVATES",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "first_activation_summary.csv", index=False)
    print(f"wrote {RESULTS / 'first_activation_summary.csv'}")
    print(out_df.to_string(index=False))

    any_activates = (out_df.status == "ACTIVATES").any()
    print("\n" + "=" * 70)
    if not any_activates:
        print("GATE DECISION: NO meaningful frequency activation at ANY AFLB in the")
        print("tested resolution range. Per Phase 4, TEST06 should STOP here and NOT")
        print("proceed to 06-E/B/C/D.")
    else:
        activating = out_df[out_df.status == "ACTIVATES"]
        print("GATE DECISION: meaningful activation found. Proceeding to 06-E.")
        print(f"Activating AFLBs: {activating.aflb.tolist()}")
        print(f"Recommended R_first for 06-E dataset: "
              f"{int(activating.first_activation_input_resolution.max())}px "
              f"(max across activating AFLBs, to guarantee ALL activating AFLBs are non-degenerate)")
    print("=" * 70)


if __name__ == "__main__":
    main()
