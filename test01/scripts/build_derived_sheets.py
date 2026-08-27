"""Derives the remaining Excel sheets from run_ablation.py's raw CSVs:
  Released_vs_Modified / Released_vs_NoFrequency  -- per-image PSNR/SSIM diff tables
  Mechanism_Audit                                  -- side-by-side internal-stat comparison
                                                       across variants, for the 9 representative images
  Tensor_File_Index                                -- index of every .pt bundle under results/tensors/

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python build_derived_sheets.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST01 = Path(__file__).resolve().parent.parent
CSV_DIR = TEST01 / "csv_export"
RESULTS_DIR = TEST01 / "results"
TENSORS_DIR = RESULTS_DIR / "tensors"


def build_diff_tables():
    df = pd.read_csv(CSV_DIR / "20_Per_Image_All_Variants.csv")
    for variant, out_name in [("modified_mask", "26_Released_vs_Modified"),
                               ("no_frequency", "27_Released_vs_NoFrequency")]:
        piv_psnr = df[df.model.isin([variant, "released"])].pivot(
            index=["Image_ID", "Degradation"], columns="model", values="psnr").reset_index()
        piv_ssim = df[df.model.isin([variant, "released"])].pivot(
            index=["Image_ID", "Degradation"], columns="model", values="ssim").reset_index()
        merged = piv_psnr.merge(piv_ssim, on=["Image_ID", "Degradation"], suffixes=("_psnr", "_ssim"))
        merged["psnr_diff"] = merged[f"{variant}_psnr"] - merged["released_psnr"]
        merged["ssim_diff"] = merged[f"{variant}_ssim"] - merged["released_ssim"]
        merged = merged.rename(columns={f"{variant}_psnr": f"{variant}_psnr", "released_psnr": "released_psnr",
                                         f"{variant}_ssim": f"{variant}_ssim", "released_ssim": "released_ssim"})
        merged.to_csv(CSV_DIR / f"{out_name}.csv", index=False)
        print(f"wrote {out_name}.csv ({len(merged)} rows), "
              f"psnr_diff: mean={merged.psnr_diff.mean():.6f} nonzero={int((merged.psnr_diff != 0).sum())}/{len(merged)}")


def build_mechanism_audit():
    mgb = pd.read_csv(CSV_DIR / "21_Ablation_MGB_Values.csv")
    freq = pd.read_csv(CSV_DIR / "22_Ablation_Frequency_Statistics.csv") \
        if (CSV_DIR / "22_Ablation_Frequency_Statistics.csv").exists() else pd.DataFrame()
    fmom = pd.read_csv(CSV_DIR / "24_Ablation_FMoM_Statistics.csv")

    rows = []
    for (image_id, deg, aflb), g in mgb.groupby(["Image_ID", "Degradation", "AFLB"]):
        row = {"Image_ID": image_id, "Degradation": deg, "AFLB": aflb}
        for _, r in g.iterrows():
            row[f"{r['model']}_alpha"] = r["alpha"]
            row[f"{r['model']}_beta"] = r["beta"]
            row[f"{r['model']}_mask_area_pct"] = r["mask_area_pct"]
        if not freq.empty:
            fsub = freq[(freq.Image_ID == image_id) & (freq.Degradation == deg) & (freq.AFLB == aflb)]
            for _, r in fsub.iterrows():
                row[f"{r['model']}_low_pct"] = r["low_pct"]
                row[f"{r['model']}_high_pct"] = r["high_pct"]
        fsub2 = fmom[(fmom.Image_ID == image_id) & (fmom.Degradation == deg) & (fmom.AFLB == aflb)]
        for _, r in fsub2.iterrows():
            row[f"{r['model']}_hl_mean"] = r["hl_mean"]
            row[f"{r['model']}_lh_mean"] = r["lh_mean"]
            row[f"{r['model']}_aflb_out_mean"] = r["aflb_out_mean"]
            row[f"{r['model']}_aflb_out_energy"] = r["aflb_out_energy"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(CSV_DIR / "28_Mechanism_Audit.csv", index=False)
    print(f"wrote 28_Mechanism_Audit.csv ({len(out)} rows)")


def build_tensor_index():
    rows = []
    if not TENSORS_DIR.exists():
        print("no tensors dir found, skipping index")
        return
    for pt_path in sorted(TENSORS_DIR.rglob("*.pt")):
        rel = pt_path.relative_to(TEST01)
        parts = pt_path.relative_to(TENSORS_DIR).parts  # variant/degradation/image_id/aflbN.pt
        if len(parts) != 4:
            continue
        variant, degradation, image_id, fname = parts
        rows.append({
            "variant": variant, "degradation": degradation, "Image_ID": image_id,
            "AFLB": fname.replace(".pt", "").upper(), "pt_path": str(rel),
            "size_bytes": pt_path.stat().st_size,
        })
    out = pd.DataFrame(rows)
    out.to_csv(CSV_DIR / "29_Tensor_File_Index.csv", index=False)
    print(f"wrote 29_Tensor_File_Index.csv ({len(out)} rows, "
          f"{out.size_bytes.sum() / 1e6:.1f} MB total)")


if __name__ == "__main__":
    build_diff_tables()
    build_mechanism_audit()
    build_tensor_index()
