"""Phase 16: render AdaIR_Degradation_Representation_Analysis.xlsx from the
CSVs exported off devon. Runs LOCALLY -- devon has flaky logical CPUs (8-11)
observed to corrupt in-process data, so CSV is the source of truth and the
.xlsx is assembled on a separate, trusted machine (same policy as test01).

Usage (local machine):
  python build_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST02 = Path(__file__).resolve().parent.parent
RESULTS = TEST02 / "results"
EXCEL_PATH = RESULTS / "AdaIR_Degradation_Representation_Analysis.xlsx"

SHEETS = [
    ("README", RESULTS, "README.csv"),
    ("Dataset", RESULTS, "dataset_manifest.csv"),
    ("Feature_Index", RESULTS / "tensors", "tensor_index.csv"),
    ("Feature_Statistics", RESULTS / "statistics", "feature_statistics.csv"),
    ("Linear_Probe", RESULTS / "classifiers", "linear_probe_results.csv"),
    ("Distance_Analysis", RESULTS / "statistics", "degradation_separation.csv"),
    ("Alpha_Beta", RESULTS / "statistics", "alpha_beta.csv"),
    ("Alpha_Beta_Probe", RESULTS / "classifiers", "alpha_beta_probe_results.csv"),
    ("AFLB_Analysis", RESULTS / "classifiers", "aflb_analysis.csv"),
    ("PSNR_SSIM", RESULTS / "statistics", "psnr_ssim.csv"),
    ("PSNR_SSIM_Correlation", RESULTS / "statistics", "psnr_ssim_correlation.csv"),
    ("Feature_Trajectory", RESULTS / "classifiers", "feature_trajectory.csv"),
    ("Confusion_Matrices", RESULTS / "classifiers", "confusion_matrices.csv"),
    ("Tensor_Index", RESULTS / "tensors", "tensor_index.csv"),
    ("Environment", RESULTS, "environment.txt"),
]


def main():
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet, base_dir, fname in SHEETS:
            path = Path(base_dir) / fname
            if not path.exists():
                print(f"SKIP {sheet}: {path} not found")
                continue
            if sheet in ("README", "Environment") and path.suffix == ".txt":
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
