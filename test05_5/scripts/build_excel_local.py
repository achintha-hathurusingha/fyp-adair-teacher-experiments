"""TEST05.5: render AdaIR_F2S_Scientific_Audit.xlsx from CSVs. Runs LOCALLY
(CSV-first, Excel-rendered-off-host policy, same as test01-05, given
devon's flaky logical CPUs 8-11 and history of connectivity/reboot issues).

Usage (local machine):
  python build_excel_local.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST05_5 = Path(__file__).resolve().parent.parent
RESULTS = TEST05_5 / "results"
EXCEL_PATH = RESULTS / "AdaIR_F2S_Scientific_Audit.xlsx"

SHEETS = [
    ("README", RESULTS, "README.csv"),
    ("Claim_Audit", RESULTS / "statistics", "claim_audit.csv"),
    ("Simple_Statistics", RESULTS / "simple_stats", "simple_statistics_probe.csv"),
    ("PCA_Leakage_Audit", RESULTS / "pca_audit", "pca_leakage_safe_results.csv"),
    ("PCA_Dimensionality", RESULTS / "compact", "compact_vs_controls.csv"),
    ("Severity_Generalization", RESULTS / "robustness", "severity_generalization_results.csv"),
    ("Parameter_Robustness", RESULTS / "robustness", "family_probe_results.csv"),
    ("Normalized_Intervention", RESULTS / "intervention", "normalized_intervention_summary.csv"),
    ("Content_vs_Degradation", RESULTS / "intervention", "degradation_specificity_ratio.csv"),
    ("Frequency_Path_Audit", RESULTS / "frequency", "variant_representation_summary.csv"),
    ("Frequency_Ablation", RESULTS / "frequency", "variant_distance_from_T0.csv"),
    ("Frequency_Randomization", RESULTS / "frequency", "frequency_randomization_control.csv"),
    ("Input_vs_Feature_Frequency", RESULTS / "frequency", "input_vs_feature_frequency_summary.csv"),
    ("Compact_Representation", RESULTS / "compact", "restoration_relevance.csv"),
    ("Negative_Controls", RESULTS / "compact", "compact_vs_controls.csv"),
    ("Mathematical_Model_Audit", RESULTS / "statistics", "mathematical_model_audit.csv"),
    ("Alternative_Models", RESULTS / "statistics", "alternative_models.csv"),
    ("GO_NO_GO", RESULTS / "statistics", "go_no_go_summary.csv"),
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
