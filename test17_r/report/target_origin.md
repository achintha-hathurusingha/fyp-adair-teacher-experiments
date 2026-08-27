# Target Origin (Phase 1)

## The claimed target values

```
sigma=15 : PSNR 34.10 dB / SSIM 0.9356
sigma=25 : PSNR 31.45 dB / SSIM 0.8931
sigma=50 : PSNR 28.22 dB / SSIM 0.8064
```

## Provenance, fully traced

1. **Model**: R2R's officially released **3D** (three-degradation: denoise+derain+dehaze)
   all-in-one checkpoint, `train_ckpt_3D_f/last.ckpt`, paired with prompt bank
   `save_prompts_3D/last/{denoise,derain,dehaze}_{key,value}.pt`. Downloaded from the
   paper's own Google Drive link (`github.com/cscxwang/R2R` README → Weights link),
   folder `ckpt/train_ckpt_3D_f` + `prompt/save_prompts_3D`.
2. **Checkpoint status**: this is the *finetune*-stage final checkpoint — i.e. the
   full 240+30 epoch schedule has already run; it is not a partial/epoch-N snapshot.
3. **Dataset**: CBSD68 (68 images), symlinked from an already-downloaded copy at
   `~/fyp-adair-distill/data/test/denoise/bsd68` (verified: `ls | wc -l` = 68 files,
   matches the standard CBSD68 benchmark size).
4. **Degradation**: synthetic AWGN added on the fly by R2R's own dataloader at
   σ=15/25/50 (`utils/dataset_utils.py:533-536`), applied fresh to the clean CBSD68
   images at test time — not a pre-degraded dataset.
5. **Eval code**: R2R's own `test_3D.py --mode 0`, which calls `test_Denoise()` three
   times (once per σ) against the *same* loaded model and *same* 68 images, using
   `utils/val_utils.py:compute_psnr_ssim` (skimage, RGB, float `[0,1]`, no shave —
   see `R2R_exact_recipe.md`).
6. **Command run**: `python test_3D.py --mode 0 --ckpt_name weights/ckpt/train_ckpt_3D_f/
   --prompt_dir weights/prompt/save_prompts_3D/ --output_path output/3D/`
   on `devon` (RTX 4090, conda env `r2r`), 2026-08-17.
7. **Result obtained**: 34.10/0.9356, 31.45/0.8931, 28.22/0.8064 — matches the paper's
   Table 1 row for R2R (3-degradation setting) to the hundredth of a dB on PSNR and to
   3 decimals on SSIM.

## What these numbers are NOT

- **Not three separate training runs.** One model, evaluated three times at three
  noise levels.
- **Not the TEST12/TEST07-B dataset.** CBSD68 is 68 real natural images at their native
  resolution (typically 321×481 or 481×321); TEST12's val set is 20 DIV2K scenes,
  each reduced to a single fixed 128×128 crop (see `dataset_comparison.csv`).
- **Not the same degradation family as the project's "Noise" condition.** Both use
  AWGN, but TEST12/TEST07-B's `degradation_synthesis.py` "Noise" function has not
  been verified to use the same σ range, distribution, or clipping as R2R's
  `_add_gaussian_noise` (see `dataset_comparison.csv`, row "noise synthesis exact
  formula" — marked UNKNOWN pending a direct diff of the two functions).
- **Not an average across degradations.** Each of the three numbers is denoising-only,
  at one fixed σ. The paper's *combined* 3-degradation average (denoise+derain+dehaze)
  is 32.53 dB / 0.918 (paper Table 1, "Average" column) — a different, lower-variance
  number than any single σ result, and itself not directly comparable to a Rain/Haze/
  Noise blended average unless the three tasks are weighted the same way.

## Direct implication for TEST17-R's comparison

Comparing "our current student ~27.0-27.3 dB" against "34.10/31.45/28.22" as if they
were commensurate numbers conflates at least three independent variables at once:
dataset, degradation-type mix, and evaluation quantization (see
`evaluation_crosscheck.md`). The comparison is not yet valid — see
`dataset_comparison.csv` and the Phase 8/9 recommendation in the main report.
