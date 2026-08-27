"""Writes csv_export/README.csv for the ablation-study workbook, summarizing
the 3-condition experiment and its findings. Run LOCALLY (mirrors the policy
established for the original 300-image analysis: devon's flaky cores 8-11
mean CSV is the source of truth and .xlsx is rendered off-host).

Usage:
  python patch_readme.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST01 = Path(__file__).resolve().parent.parent
CSV_DIR = TEST01 / "csv_export"

GIT_SHA = "ccb8b98e49614e07badd0641e5163fa7635c2f02"
CKPT_NAME = "adair3d.ckpt"
N_PARAMS = 28_784_824

readme_text = f"""AdaIR Ablation Study -- Released vs. ModifiedMask vs. NoFrequency
Checkpoint: {CKPT_NAME} (3-degradation all-in-one teacher, {N_PARAMS:,} params)
AdaIR source: github.com/c-yn/AdaIR @ {GIT_SHA}
Images: 300 (100 Rain100L / 100 SOTS-outdoor / 100 BSD68), same manifest as
        the original 300-image analysis (../manifest.csv)
Full source-code audit: docs/adair_source_audit.md
Full research report:   docs/adair_ablation_report.md

BACKGROUND
The original 300-image analysis found the released AdaIR checkpoint's
low-frequency mask (Ml) is identically zero at every AFLB, for every image,
at native benchmark resolution -- independently verified via a from-scratch
forward-pass trace (Ml+Mh==1 exact, energy split exact, manual ifft
reconstruction exact match, mask.unique()==[0.0]). This workbook answers the
follow-up question that finding raises: if the mask mechanism is inert here,
does it matter to restoration quality, and where is the paper's stated
frequency-adaptivity actually coming from?

THREE CONDITIONS (all inference-only, same checkpoint, see Phase-4 note below)
  released       Exactly the vendored AdaIR/net/model.py forward pass.
  modified_mask  Same checkpoint, same alpha/beta values -- ONLY the mask
                 half-width formula is re-ordered to match the paper's
                 Eq.(1) (p.6): "alpha*H/k" (multiply then divide), not the
                 released code's "(H//k)*alpha" (floor-divide then multiply).
                 These are NOT algebraically equivalent in general (proof and
                 counterexample in docs/adair_source_audit.md section 3.1).
  no_frequency   AFLB / FMiM cross-attention / FMoM all remain fully active
                 (not removed, per task instructions). ONLY the FFT/mask
                 computation is disabled: high_feature <- conv_feat directly
                 (identity), low_feature <- zeros. This is mathematically
                 what the released mask ALREADY evaluates to at these
                 resolutions (mask==0 everywhere) -- so it isolates "does the
                 fft2/ifft2 round trip itself contribute anything" from "does
                 the AFLB/FMoM machinery downstream contribute."

CHECKPOINT COMPATIBILITY (Phase 4)
All three variants load the IDENTICAL adair3d.ckpt state_dict via the same
strict loader (0 missing / 0 unexpected keys, 28,784,824 params) -- neither
variant adds, removes, or resizes any parameter, so NO RETRAINING was
required or performed for either variant. Caveat: modified_mask's alpha/beta
values were learned under gradients that flowed through the ORIGINAL
(floor-then-multiply) formula during training, so this variant tests "the
same learned alpha/beta, decoded via the paper's formula" rather than an
independently-trained model. This is documented, not hidden.

HEADLINE RESULTS (300 images, mean +/- std; full breakdown in Baseline_Summary)
  released:       PSNR 33.428 +/- 5.560 dB   SSIM 0.9500 +/- 0.0538
  modified_mask:  PSNR 33.428 +/- 5.560 dB   SSIM 0.9500 +/- 0.0538   (bit-identical to released on 264/300 images)
  no_frequency:   PSNR 33.425 +/- 5.556 dB   SSIM 0.9499 +/- 0.0539

STATISTICAL ANALYSIS (paired, same 300 images across conditions -- Statistical_Analysis sheet)
  modified_mask - released:  NOT statistically significant, ANY degradation.
    Rain/Noise: 100/100 images EXACTLY zero difference (both formulas floor
    to the same zero mask at these resolutions). Haze: 64/100 exactly zero,
    36/100 differ by ~1e-5 dB (float noise), overall p=0.395 (paired t-test).
  no_frequency - released:  mixed. Overall (ALL, n=300) NOT significant for
    PSNR (p=0.46) but IS significant for Noise specifically (p=1.5e-18,
    paired t-test; p=6.7e-18 Wilcoxon) and for Haze SSIM (p=0.012) -- BUT the
    effect size is negligible in every case: -0.0035 dB mean for Noise PSNR,
    with a 95% CI of [-0.0042, -0.0029] dB. This is a statistically-detectable
    but practically-meaningless systematic offset, consistent with float32
    FFT/IFFT round-trip error (removing the FFT round trip is numerically
    CLEANER, not worse, in the sense that it skips an identity operation that
    picks up rounding error). Do not read "significant" as "matters."

RESOLUTION SWEEP (Resolution_Sweep sheet -- does NOT modify the model, only
sweeps input size; released vs modified_mask compared, no_frequency has no
mask to sweep)
  AFLB3 (shallowest AFLB) mask first turns on:
    released:       1024x1024 input  (feature 512x512) -- within the tested
                     grid {{128,256,320,512,640,1024}} + rectangular 2:1 variants
    modified_mask:   640x640 input   (feature 320x320) -- turns on ~40%
                     EARLIER than released. This is direct empirical proof
                     the two formulas are not equivalent once resolution is
                     large enough -- they only happen to coincide (both zero)
                     at the resolutions AdaIR is actually benchmarked at.
  AFLB1/AFLB2: did not activate for either variant within the tested grid
  (deeper AFLBs need proportionally larger inputs; see original
  11_Resolution_Sweep.csv in ../csv_export for the single-variant sweep that
  also hit GPU OOM above 1024px on this 24GB card).

COMPUTATIONAL COST
  no_frequency mean latency 169.9ms vs released 173.5ms (ALL, n=300) -- a
  modest ~2% speedup from skipping fft2/mask/ifft2 x3 AFLBs. Peak-memory
  figures in Baseline_Summary are the MAX observed within each group (not
  mean) and no_frequency's reported max is HIGHER than released's despite
  doing less work -- this is very likely a CUDA allocator/fragmentation
  artifact of running all 3 variants sequentially in one process (not
  reset between variants beyond torch.cuda.empty_cache()) rather than a
  genuine cost of the no_frequency computation; flagged here as a
  measurement caveat, not asserted as a real effect. A clean per-process
  remeasurement would be needed to trust the memory numbers quantitatively.

SHEETS
  README                        this sheet
  Baseline_Image_Results        Phase-1 baseline, released only, per-image (image_id/degradation/filename/psnr/ssim/inference_time_ms)
  Baseline_Summary              Phase-6 comparison.csv: all 3 variants x degradation, mean/std/median PSNR+SSIM, latency, peak mem, params
  MGB_Values                    alpha/beta/mask_area_pct, 9 representative images (3/degradation) x 3 AFLB x 3 variants
  Frequency_Statistics          FFT energy split (low/high %), same 9 images x 3 AFLB, released+modified_mask only (no_frequency has no FFT)
  FMiM_Statistics               conv_feat/fft_magnitude/raw_low/raw_high/mined_low/mined_high stats, same 9 images x 3 AFLB x 3 variants
  FMoM_Statistics                H-L/L-H attention + AFLB-output stats, same 9 images x 3 AFLB x 3 variants
  Per_Image_All_Variants        all 300 images x 3 variants, psnr/ssim/inference_time_ms/peak_memory_mb (raw data behind every summary above)
  Resolution_Sweep               Phase 9: alpha/beta/mask/energy vs. input resolution (square+rectangular), 3 variants
  Released_vs_Modified          per-image paired PSNR/SSIM diff, modified_mask - released
  Released_vs_NoFrequency       per-image paired PSNR/SSIM diff, no_frequency - released
  Statistical_Analysis          Phase 7: paired t-test + Wilcoxon signed-rank, both contrasts, per degradation + overall
  Mechanism_Audit                side-by-side alpha/beta/mask/energy/H-L/L-H/AFLB-output stats across all 3 variants, 9 representative images
  Tensor_File_Index             index of all .pt bundles under results/tensors/ (81 files, ~4.2GB, 9 images x 3 AFLB x 3 variants)

FULL TENSORS
  results/tensors/<variant>/<degradation>/<Image_ID>/aflb{{1,2,3}}.pt
  (float16, one dict per AFLB per image per variant; complex FFT tensor
  stored as its magnitude only, `fft_shifted_abs`, to keep files small)

REPRODUCE
  cd test01/scripts && conda activate adair-distill
  ./run_all_experiments.sh          (or: taskset -c 0-7,12-31 python run_ablation.py, etc. -- see configs/*.yaml)
"""

df = pd.DataFrame({"README": readme_text.split("\n")})
out_path = CSV_DIR / "README.csv"
df.to_csv(out_path, index=False)
print(f"wrote {out_path} ({len(df)} lines)")
