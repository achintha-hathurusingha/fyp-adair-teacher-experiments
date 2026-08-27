"""Phase 7: paired statistical analysis of ModifiedMask-Released and
NoFrequency-Released, per degradation and overall. Paired because the same
300 images are evaluated under all 3 variants (see run_ablation.py's
per-image seeding fix, which guarantees identical noisy inputs across
variants for the Noise degradation).

For each (contrast, degradation) computes: mean diff, std diff, median diff,
a 95% CI on the mean (paired t-interval), and BOTH a paired t-test and a
Wilcoxon signed-rank test (non-parametric, robust to the many exact-zero
differences we expect given the mask-degeneracy finding -- a t-test alone
would be misleading when most differences are exactly 0.0).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python analyze_results.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TEST01 = Path(__file__).resolve().parent.parent
CSV_DIR = TEST01 / "csv_export"
IN_PATH = CSV_DIR / "20_Per_Image_All_Variants.csv"
OUT_PATH = CSV_DIR / "25_Statistical_Analysis.csv"

CONTRASTS = [("modified_mask", "released"), ("no_frequency", "released")]
METRICS = ["psnr", "ssim"]


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    n = len(diff)
    mean_d, std_d, median_d = diff.mean(), diff.std(ddof=1), np.median(diff)
    se = std_d / np.sqrt(n) if n > 1 else float("nan")
    ci_lo, ci_hi = (mean_d - 1.96 * se, mean_d + 1.96 * se) if n > 1 else (float("nan"), float("nan"))

    all_zero = np.allclose(diff, 0)
    if all_zero:
        t_p, w_p = float("nan"), float("nan")
        note = "all differences are exactly 0 (variant produced bit-identical output on every image) -- no test applicable"
    else:
        try:
            t_stat, t_p = stats.ttest_rel(a, b)
        except Exception:
            t_p = float("nan")
        try:
            nonzero = diff[diff != 0]
            if len(nonzero) >= 1:
                w_stat, w_p = stats.wilcoxon(a[diff != 0], b[diff != 0])
            else:
                w_p = float("nan")
        except Exception:
            w_p = float("nan")
        note = ""

    return {
        "n": n, "mean_diff": mean_d, "std_diff": std_d, "median_diff": median_d,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "n_exactly_zero_diff": int((diff == 0).sum()),
        "paired_ttest_p": t_p, "wilcoxon_p": w_p, "note": note,
    }


def main():
    df = pd.read_csv(IN_PATH)
    rows = []
    for variant_a, variant_b in CONTRASTS:
        for metric in METRICS:
            piv = df[df.model.isin([variant_a, variant_b])].pivot(
                index="Image_ID", columns="model", values=metric)
            piv = piv.dropna()
            for deg in ["Rain", "Haze", "Noise", "ALL"]:
                if deg == "ALL":
                    ids = piv.index
                else:
                    ids = df[df.Degradation == deg].Image_ID.unique()
                sub = piv.loc[piv.index.intersection(ids)]
                if len(sub) == 0:
                    continue
                s = paired_stats(sub[variant_a].to_numpy(), sub[variant_b].to_numpy())
                rows.append({"contrast": f"{variant_a} - {variant_b}", "metric": metric, "degradation": deg, **s})

    out = pd.DataFrame(rows)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
