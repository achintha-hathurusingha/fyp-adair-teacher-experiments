# Evaluation Crosscheck (Phase 3)

## Code compared

- **R2R**: `~/FYP/R2R/utils/val_utils.py:compute_psnr_ssim` (official repo, read directly)
- **This project**: `~/teacher-experiments/test17/scripts/train.py:90` `psnr_ssim()`
  (identical function used across TEST12/TEST17 per grep; TEST12 does not define its
  own separate metric function — `test12/scripts/models.py` and `train.py` were not
  found to redefine PSNR/SSIM, so TEST17's version is treated as representative)

## Side-by-side

| Aspect | R2R (`utils/val_utils.py`) | This project (`test17/scripts/train.py:90-95`) | Match? |
|---|---|---|---|
| PSNR function | `skimage.metrics.peak_signal_noise_ratio` | same | YES |
| SSIM function | `skimage.metrics.structural_similarity` | same | YES |
| `channel_axis` | `-1` (explicit, correct for HWC) | `2` (explicit, correct for HWC) | YES (equivalent) |
| Value range fed to metric | float `[0,1]`, `data_range=1` | **uint8 `[0,255]`**, `data_range=255` | **NO** |
| Quantization step | none — metric computed directly on float tensor (after `np.clip(...,0,1)`) | **`tgt_u8, pred_u8` — explicit round-to-uint8 conversion before metric call** | **NO** |
| Color space | RGB, all 3 channels | RGB, all 3 channels (`channel_axis=2` implies HWC RGB) | YES |
| Border/shave | none in either | none in either | YES |
| Averaging | per-image, then arithmetic mean (`AverageMeter`) | per-crop, then `.mean()` over a dataframe | YES (equivalent) |

## Finding

Two of seven metric-pipeline properties differ: **this project quantizes to uint8
before computing PSNR/SSIM; R2R does not.** Converting a float restoration to uint8
introduces rounding of up to ±0.5/255 per pixel before the metric is computed, which
is a real, well-understood source of measured PSNR being *slightly lower* (typically a
few hundredths to ~0.1-0.15 dB, occasionally more depending on how close-to-boundary
values are) than the equivalent float-domain computation on the same restoration.
`data_range=255` vs `data_range=1` is mathematically equivalent *given inputs scaled
consistently* — it is not, by itself, a source of error; the actual mismatch is the
uint8 rounding step, not the data_range constant.

**This is real but almost certainly not the dominant cause of a ~1+ dB gap.** It is a
confound worth controlling for in any head-to-head number (Phase 4/5, if run), but it
does not explain results that differ by whole dB, only fractions of a dB.

## What was NOT cross-checked (scope limit of this pass)

- No numeric crosscheck was run yet (i.e., taking one actual restored/clean image pair
  and computing PSNR/SSIM through *both* code paths to measure the real-world delta
  empirically, rather than reasoning about it from the code alone). This is a cheap,
  fast (~seconds) diagnostic that should be run before Phase 4/5 — flagged as the
  first recommended action in the main report.
- `test12/scripts/models.py` and other per-experiment eval variants were not
  individually diffed against `test17/scripts/train.py` — if any earlier test (05, 06,
  07-pilot, etc.) used a different metric implementation than TEST17, that has not
  been verified here. Marked UNKNOWN.
