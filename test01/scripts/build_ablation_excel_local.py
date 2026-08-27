"""Phase 11: render results/AdaIR_Ablation_Study.xlsx from the CSVs exported
off devon. Runs LOCALLY -- devon has flaky logical CPUs (8-11) that have been
observed to corrupt in-process data, so CSV is the source of truth and the
.xlsx is assembled on a separate, trusted machine (same policy as the
original 300-image analysis).

Usage (local machine):
  python build_ablation_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST01 = Path(__file__).resolve().parent.parent
CSV_DIR = TEST01 / "csv_export"
RESULTS_DIR = TEST01 / "results"
EXCEL_PATH = RESULTS_DIR / "AdaIR_Ablation_Study.xlsx"

# sheet name -> csv glob pattern (relative to CSV_DIR unless noted RESULTS_DIR)
SHEET_SOURCES = [
    ("README", CSV_DIR, "*README.csv"),
    ("Baseline_Image_Results", RESULTS_DIR / "baseline", "results.csv"),
    ("Baseline_Summary", RESULTS_DIR, "comparison.csv"),
    ("MGB_Values", CSV_DIR, "*21_Ablation_MGB_Values.csv"),
    ("Frequency_Statistics", CSV_DIR, "*22_Ablation_Frequency_Statistics.csv"),
    ("FMiM_Statistics", CSV_DIR, "*23_Ablation_FMiM_Statistics.csv"),
    ("FMoM_Statistics", CSV_DIR, "*24_Ablation_FMoM_Statistics.csv"),
    ("Per_Image_All_Variants", CSV_DIR, "*20_Per_Image_All_Variants.csv"),
    ("Resolution_Sweep", RESULTS_DIR, "resolution_sweep.csv"),
    ("Released_vs_Modified", CSV_DIR, "*26_Released_vs_Modified.csv"),
    ("Released_vs_NoFrequency", CSV_DIR, "*27_Released_vs_NoFrequency.csv"),
    ("Statistical_Analysis", CSV_DIR, "*25_Statistical_Analysis.csv"),
    ("Mechanism_Audit", CSV_DIR, "*28_Mechanism_Audit.csv"),
    ("Tensor_File_Index", CSV_DIR, "*29_Tensor_File_Index.csv"),
]


def main():
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sheet, base_dir, pattern in SHEET_SOURCES:
            matches = list(Path(base_dir).glob(pattern))
            if not matches:
                print(f"SKIP {sheet}: no file matching {pattern} in {base_dir}")
                continue
            path = matches[0]
            if sheet == "README":
                text = path.read_text(encoding="utf-8")
                if text.split("\n", 1)[0].strip() != "README":
                    df = pd.DataFrame({"README": text.split("\n")})
                else:
                    df = pd.read_csv(path)
            else:
                df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"{sheet}: {df.shape} <- {path.name}")

    print(f"\nwrote {EXCEL_PATH}")


if __name__ == "__main__":
    main()
