"""Writes results/README.csv for the TEST05.5 workbook. Run LOCALLY."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST05_5 = Path(__file__).resolve().parent.parent

readme_text = (TEST05_5 / "report" / "readme_text_source.txt").read_text(encoding="utf-8")

out_path = TEST05_5 / "results" / "README.csv"
pd.DataFrame({"README": readme_text.split("\n")}).to_csv(out_path, index=False)
print(f"wrote {out_path}")
