"""TEST05 Phase 22: render AdaIR_Degradation_Specificity_Analysis.xlsx from
CSVs. Runs LOCALLY (CSV-first, Excel-rendered-off-host policy, same as
test01-04, given devon's flaky logical CPUs 8-11).

Usage (local machine):
  python build_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST05 = Path(__file__).resolve().parent.parent
RESULTS = TEST05 / "results"
EXCEL_PATH = RESULTS / "AdaIR_Degradation_Specificity_Analysis.xlsx"

SHEETS = [
    ("README", RESULTS, "README.csv"),
    ("Candidate_Features", RESULTS / "statistics", "linear_probe.csv"),
    ("Linear_Probe", RESULTS / "statistics", "linear_probe.csv"),
    ("Scene_Sensitivity", RESULTS / "statistics", "scene_sensitivity.csv"),
    ("Channel_Ranking", RESULTS / "channel_analysis", "channel_rank.csv"),
    ("Channel_Ablation", RESULTS / "intervention", "channel_ablation.csv"),
    ("Channel_Group_Intervention", RESULTS / "intervention", "channel_group_intervention.csv"),
    ("Content_Control", RESULTS / "intervention", "content_control.csv"),
    ("Spatial_Analysis", RESULTS / "spatial_analysis", "spatial_importance.csv"),
    ("Frequency_Analysis", RESULTS / "frequency_analysis", "frequency_profile.csv"),
    ("Frequency_Summary", RESULTS / "frequency_analysis", "frequency_summary.csv"),
    ("Frequency_Channel_Ranking", RESULTS / "frequency_analysis", "frequency_channel_ranking.csv"),
    ("Compact_Embedding", RESULTS / "statistics", "compact_embedding.csv"),
    ("Alpha_Beta", RESULTS / "statistics", "alpha_beta.csv"),
    ("NPU_Cost", RESULTS / "statistics", "npu_cost.csv"),
    ("Distillation_Ranking", RESULTS / "statistics", "distillation_candidate_ranking.csv"),
    ("Environment", RESULTS, "environment.txt"),
    ("Tensor_Index", RESULTS / "tensors", "tensor_index.csv"),
]


def main():
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet, base_dir, fname in SHEETS:
            path = Path(base_dir) / fname
            if not path.exists():
                print(f"SKIP {sheet}: {path} not found")
                continue
            if path.suffix == ".txt":
                text = path.read_text(encoding="utf-8")
                df = pd.DataFrame({sheet: text.split("\n")})
            elif sheet == "README":
                text = path.read_text(encoding="utf-8")
                if text.split("\n", 1)[0].strip() != "README":
                    df = pd.DataFrame({"README": text.split("\n")})
                else:
                    df = pd.read_csv(path)
            else:
                df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"{sheet}: {df.shape} <- {path}")

    print(f"\nwrote {EXCEL_PATH}")


if __name__ == "__main__":
    main()
