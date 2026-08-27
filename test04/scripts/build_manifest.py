"""TEST04: reference TEST03's existing scene manifest (read-only) --
does NOT regenerate or copy the underlying images, per the task's
instruction to use the exact TEST03 generated data.

Usage: python build_manifest.py
"""
from __future__ import annotations

import csv
from pathlib import Path

TEST04 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST04.parent
TEST03_MANIFEST = TEACHER_EXP / "test03" / "results" / "manifest" / "scene_manifest.csv"
OUT_PATH = TEST04 / "results" / "manifest" / "scene_manifest.csv"


def main():
    with open(TEST03_MANIFEST) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100, f"expected 100 scenes from test03, got {len(rows)}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH} ({len(rows)} scenes, referencing test03/data/ images read-only)")


if __name__ == "__main__":
    main()
