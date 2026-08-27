"""TEST03 Phase 20: render AdaIR_Controlled_Degradation_Analysis.xlsx from
CSVs exported off devon. Runs LOCALLY (CSV-first, Excel-rendered-off-host
policy, same as test01/test02, given devon's flaky logical CPUs 8-11).

Usage (local machine):
  python build_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST03 = Path(__file__).resolve().parent.parent
RESULTS = TEST03 / "results"
EXCEL_PATH = RESULTS / "AdaIR_Controlled_Degradation_Analysis.xlsx"

SHEETS = [
    ("README", RESULTS, "README.csv"),
    ("Scene_Manifest", RESULTS / "manifest", "scene_manifest.csv"),
    ("Degradation_Parameters", RESULTS / "manifest", "degradation_parameters.csv"),
    ("Data_Validation", RESULTS / "manifest", "validation_checks.csv"),
    ("Feature_Index", RESULTS / "tensors", "tensor_index.csv"),
    ("Feature_Statistics", RESULTS / "statistics", "feature_statistics.csv"),
    ("Linear_Probe", RESULTS / "classifiers", "linear_probe_results.csv"),
    ("Feature_Trajectory", RESULTS / "classifiers", "feature_trajectory.csv"),
    ("Paired_Distances", RESULTS / "statistics", "paired_distance_analysis.csv"),
    ("Scene_vs_Degradation", RESULTS / "statistics", "degradation_vs_scene.csv"),
    ("Alpha_Beta", RESULTS / "statistics", "alpha_beta.csv"),
    ("Alpha_Beta_Probe", RESULTS / "classifiers", "alpha_beta_probe_results.csv"),
    ("AFLB_Analysis", RESULTS / "classifiers", "aflb_analysis.csv"),
    ("Raw_Low_Check", RESULTS / "statistics", "raw_low_check.csv"),
    ("PSNR_SSIM", RESULTS / "statistics", "restoration_metrics.csv"),
    ("TEST02_vs_TEST03", RESULTS / "statistics", "test02_vs_test03.csv"),
    ("Bootstrap_CI", RESULTS / "statistics", "scene_aware_bootstrap_ci.csv"),
    ("Confusion_Matrices", RESULTS / "classifiers", "confusion_matrices.csv"),
    ("Tensor_Index", RESULTS / "tensors", "tensor_index.csv"),
    ("Swap_Prep_Index", RESULTS / "tensors", "representation_swap_prep_index.csv"),
    ("Environment", RESULTS, "environment.txt"),
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
