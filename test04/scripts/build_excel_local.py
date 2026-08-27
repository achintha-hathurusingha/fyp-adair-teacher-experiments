"""TEST04 Phase 23: render AdaIR_Causal_Representation_Intervention.xlsx
from CSVs. Runs LOCALLY (CSV-first, Excel-rendered-off-host policy, same
as test01/test02/test03, given devon's flaky logical CPUs 8-11).

Usage (local machine):
  python build_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST04 = Path(__file__).resolve().parent.parent
RESULTS = TEST04 / "results"
EXCEL_PATH = RESULTS / "AdaIR_Causal_Representation_Intervention.xlsx"

SHEETS = [
    ("README", RESULTS, "README.csv"),
    ("Forward_Graph", RESULTS / "statistics", "forward_graph_table.csv"),
    ("Environment", RESULTS, "environment.txt"),
    ("Normal_Baseline", RESULTS / "metrics", "normal_baseline.csv"),
    ("Self_Swap_Control", RESULTS / "controls", "self_swap_validation.csv"),
    ("Cross_Degradation_Swaps", RESULTS / "interventions", "cross_degradation_swaps.csv"),
    ("Skip_Connection_Progressive", RESULTS / "interventions", "skip_connection_progressive.csv"),
    ("Cross_Scene_Control", RESULTS / "controls", "cross_scene_control.csv"),
    ("Random_Control", RESULTS / "controls", "random_control.csv"),
    ("Zero_Mean_Control", RESULTS / "controls", "zero_mean_control.csv"),
    ("Donor_Similarity", RESULTS / "statistics", "donor_similarity.csv"),
    ("Donor_Similarity_Summary", RESULTS / "statistics", "donor_similarity_summary.csv"),
    ("Residual_Analysis", RESULTS / "statistics", "residual_summary.csv"),
    ("Output_Probe", RESULTS / "statistics", "output_degradation_probe.csv"),
    ("Swap_Matrix", RESULTS / "statistics", "swap_matrix.csv"),
    ("Point_Summary", RESULTS / "statistics", "point_summary.csv"),
    ("Distillation_Ranking", RESULTS / "statistics", "distillation_target_ranking.csv"),
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
