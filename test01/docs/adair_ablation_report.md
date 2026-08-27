# AdaIR Frequency-Mechanism Ablation: Released vs. ModifiedMask vs. NoFrequency

## 1. Objective

The prior 300-image white-box analysis found that AdaIR's released
low-frequency mask (`Ml`) is identically zero at every AFLB, for every
tested image, at standard benchmark resolution — independently verified
with a from-scratch forward-pass trace. That finding raised a sharper
question than "how does the mask differ across degradations": **if the
mask mechanism is inert, does it matter to restoration quality, and where
is the paper's claimed frequency-adaptivity actually coming from?** This
report answers that question with a controlled 3-condition ablation.

## 2. Original AdaIR implementation

AdaIR (Cui et al., ICLR 2025, arXiv:2403.14614) is a 4-level
Transformer encoder-decoder for all-in-one image restoration. Between
decoder levels it inserts an **Adaptive Frequency Learning Block (AFLB)**
(`FreModule` in code), which (1) mines low/high-frequency components from
the input image via FFT + a learned adaptive mask (**FMiM**), then (2)
cross-modulates them with the decoder feature via channel/spatial
attention (**FMoM**). Full class/equation map in
`docs/adair_source_audit.md`.

## 3. Source-code audit

See `docs/adair_source_audit.md` for the complete audit. Headline finding:
the paper's Eq. (1) (p.6) specifies the mask box half-height as `alpha*H/k`
(multiply `alpha` by `H`, then divide by `k=128`); the released code
computes `int((H//k) * alpha)` — **floor-divides `H` by `k` first, then
multiplies by `alpha`**. These are not algebraically equivalent in
general (counterexample: `H=250, alpha=0.9, k=128` gives `1` under the
paper's order and `0` under the code's order). At the resolutions
AdaIR's own benchmarks use, both orders happen to floor to zero (verified
below), but they diverge once resolution grows — this ablation's
`modified_mask` condition tests exactly that divergence.

## 4. Experimental setup

- Checkpoint: `adair3d.ckpt`, the 3-degradation all-in-one teacher,
  28,784,824 parameters, loaded via a strict loader (0 missing / 0
  unexpected keys) for all three conditions.
- Host: devon (RTX 4090, 24GB). **Known issue**: logical CPUs 8-11 on this
  host intermittently and silently corrupt in-process data; every script
  invocation is pinned with `taskset -c 0-7,12-31`. Excel is never written
  on this host (CSV is the source of truth; `.xlsx` is rendered on a
  separate machine).
- Determinism: global seed 0 (`np.random.seed`, `torch.manual_seed`) plus,
  critically, a **per-image** noise seed (`np.random.RandomState(hash(image_id))`)
  for the synthetic denoising noise — a global-seed-only approach was tried
  first and found (via a smoke test) to let the RNG stream drift between
  the three sequential variant runs, silently giving each variant a
  *different* noisy input for the same `Image_ID`. This is documented as a
  bug found and fixed during this study, not swept under the rug.

## 5. Dataset

Identical 300-image manifest to the original analysis: 100 Rain100L
(derain, full test split), 100 SOTS-outdoor (dehaze, de-duplicated by
scene), 100 BSD68 (denoise, 68 unique images at sigma=25 + 16 extra at
sigma=15 + 16 extra at sigma=50 = 100 instances). Sampling rationale
unchanged from the original analysis; see that report for detail.

## 6. Baseline reproduction

`released` condition, 300 images: **PSNR 33.428 ± 5.560 dB, SSIM 0.9500 ±
0.0538** overall (Rain 38.62 dB / Haze 30.39 dB / Noise 31.28 dB — see
`results/comparison.csv`, `Baseline_Summary` sheet). These numbers match
the original 300-image analysis exactly (same manifest, same checkpoint,
same crop/noise protocol) — confirms this rerun reproduces that baseline
correctly rather than silently drifting from it.

## 7. Mask degeneracy finding (recap)

Independently re-confirmed in this ablation's `released` condition on the
9 representative images: `mask_area_pct == 0` at every AFLB (see
`MGB_Values` sheet). Not re-litigated in depth here — see the original
analysis notebook's Appendix A for the full verification (`Ml+Mh==1`
exact, energy split exact, manual IFFT match, `mask.unique()==[0.0]`).

## 8. Resolution sweep

`results/resolution_sweep.csv` / `Resolution_Sweep` sheet, square inputs
{128,256,320,512,640,1024}px and four 2:1 rectangular variants, `released`
vs `modified_mask` (no_frequency has no mask to sweep):

| | first mask-active input (AFLB3) |
|---|---|
| `released` | 1024×1024 (feature 512×512) |
| `modified_mask` | **640×640** (feature 320×320) |

`modified_mask`'s boundary switches on at a **~40% smaller** input
resolution than `released`'s — direct empirical confirmation that the two
formulas are not equivalent, and a first quantification of the gap. AFLB1
and AFLB2 (deeper, smaller feature maps) did not activate for either
variant within the tested grid (consistent with needing proportionally
larger inputs — 4x and 8x more than AFLB3's threshold respectively, per
the downsample factors in the audit doc).

## 9. Modified-mask experiment

**Design**: single-line change per axis in `FreModule.fft()`
(`scripts/model_variants.py::_fft_modified_mask`) — mask half-width
computed as `alpha*h/n` instead of `(h//n)*alpha`. Same checkpoint, same
learned `alpha`/`beta`, no retraining (Phase 4: verified inference-only
compatible, 0 missing/0 unexpected keys, no parameter shape changes — see
`configs/modified_mask.yaml`).

**Result on the 300-image benchmark set**: statistically and practically
indistinguishable from `released`.
- Rain: 100/100 images **exactly bit-identical** PSNR/SSIM.
- Noise: 100/100 images exactly bit-identical.
- Haze: 64/100 exactly identical, 36/100 differ by ~1e-5 dB (floating-point
  noise level, not a meaningful restoration difference).
- Paired t-test, PSNR, overall (n=300): p=0.395 (not significant).

**Interpretation**: at native benchmark resolution, correcting the
order-of-operations deviation from the paper's Eq.(1) does **not** change
model behavior, because both formulas still floor to zero given the actual
`alpha`/`beta` (~0.5) and feature sizes (<256px) present in this
checkpoint's AFLBs. The resolution sweep (section 8) shows the two
formulas *do* diverge at larger, still-realistic input sizes (640-1024px)
— so this is a genuine implementation-level deviation from the published
equation, it simply does not happen to matter for the specific benchmark
resolutions AdaIR reports numbers on.

## 10. No-frequency experiment

**Design**: `FreModule.fft()` replaced with `high_feature = conv_feat`
(identity), `low_feature = zeros_like(conv_feat)` — no `torch.fft.fft2` /
`ifft2` call at all (`scripts/model_variants.py::_fft_no_frequency`). AFLB,
FMiM cross-attention, and FMoM (H-L/L-H) stay fully active and unmodified,
per the task's explicit instruction not to remove the whole AFLB. Same
checkpoint, no retraining (`configs/no_frequency.yaml`); this reproduces
exactly what the released mask already evaluates to at these resolutions
(mask≡0), so it isolates "does the FFT/IFFT computation itself contribute
anything" from "does the AFLB/FMoM machinery downstream contribute" (the
latter is identical in both conditions here by construction).

**Result**: overall (n=300) mean PSNR difference vs. `released` is
**-0.0031 dB** (not significant, paired t p=0.464). Per degradation:
Rain not significant (p=0.798); Haze not significant for PSNR (p=0.218)
but SSIM is (p=0.012, mean diff -0.0000455, negligible); **Noise is
statistically significant** for both PSNR (p=1.5e-18) and SSIM (p<1e-15),
but the effect size is **-0.0035 dB mean PSNR**, 95% CI
[-0.0042, -0.0029] dB — a highly consistent but minuscule systematic
offset, three orders of magnitude smaller than the 5.56 dB std-dev of PSNR
across the dataset.

**Interpretation**: this is the expected float32 `fft2`→`ifft2` round-trip
numerical error (an identity operation under exact arithmetic, `high =
ifft(fft(x))`, picks up rounding error under floating point; skipping the
round trip entirely, as `no_frequency` does, is if anything numerically
*cleaner*). It is statistically detectable because it's systematic across
100 paired images, not because it reflects a real quality difference.
**Do not read "p<0.05" as "the frequency computation matters" here** —
the confidence interval excludes zero but sits entirely inside a range no
human observer or downstream application would notice.

## 11. PSNR/SSIM comparison summary

| model | PSNR (dB) | SSIM | mean latency (ms) |
|---|---:|---:|---:|
| released | 33.428 ± 5.560 | 0.9500 ± 0.0538 | 173.5 |
| modified_mask | 33.428 ± 5.560 | 0.9500 ± 0.0538 | 171.7 |
| no_frequency | 33.425 ± 5.556 | 0.9499 ± 0.0539 | 169.9 |

(full per-degradation breakdown: `results/comparison.csv`,
`Baseline_Summary` sheet)

## 12. Internal representation comparison

`Mechanism_Audit` sheet (9 representative images x 3 AFLB x 3 variants):
`released`/`modified_mask` alpha/beta/mask/energy/H-L/L-H/AFLB-output
statistics are identical to the precision shown (5-6 significant figures)
on Rain/Noise, near-identical on Haze. `no_frequency`'s AFLB-output
statistics (mean/std/energy) are close to but not identical to
`released`'s — consistent with the FMiM cross-attention producing a
slightly different result when fed an exact-zero `low_feature` versus a
numerically-near-zero-but-not-exact `raw_low` from the FFT round trip
(both give the *same* qualitative behavior: cross-attention over an
all/near-zero query degenerates to a uniform/global-context attention
pattern — this was observed directly in `FMiM_Statistics`, where
`mined_low`'s mean/std are non-zero despite `raw_low` being exactly zero
in `released`).

## 13. Statistical analysis

Full paired analysis (paired t-test + Wilcoxon signed-rank, both
contrasts, 4 degradation splits x 2 metrics = 16 rows) in
`Statistical_Analysis` sheet / `csv_export/25_Statistical_Analysis.csv`.
Summarized in sections 9-10 above. Method note: Wilcoxon was computed
alongside the t-test specifically because many differences are exactly
zero (bit-identical outputs) — a t-test alone on a mostly-zero-with-some-
noise distribution can be misleading; both tests agree in every row here.

## 14. Computational cost

Parameters: 28,784,824 for all three conditions (identical architecture
and weights). Mean per-image latency: `released` 173.5ms, `modified_mask`
171.7ms, `no_frequency` 169.9ms — a modest ~2% speedup from skipping
`fft2`/mask/`ifft2` across all 3 AFLBs, consistent with the FFT ops being
a small fraction of total compute in a Transformer-dominated architecture.
**Peak-memory figures should not be over-interpreted**: `Baseline_Summary`
reports the group **maximum**, not mean, and `no_frequency`'s reported
peak (5240MB) is *higher* than `released`'s (4836MB) despite doing
strictly less computation — almost certainly a CUDA allocator/fragmentation
artifact of running all three variants sequentially in one process
(`torch.cuda.empty_cache()` between variants, no full process restart)
rather than a genuine property of the `no_frequency` computation. A clean
per-process remeasurement would be needed before drawing any conclusion
from the memory numbers specifically.

## 15. Interpretation

- **Observed** (direct measurement, no inference required): the released
  mask is exactly zero at every AFLB on all 300 benchmark images; the
  paper's corrected formula would *also* be zero on these same 300 images
  but diverges from the released formula at resolutions ≥640px; removing
  the FFT/mask computation entirely changes PSNR by less than 0.004 dB on
  average, with the only "statistically significant" effect being a
  numerical-precision artifact three orders of magnitude smaller than the
  dataset's natural PSNR variance.
- **Inferred** (reasonable conclusion from the observations, not directly
  measured): the specific binary FFT-based frequency-boundary mechanism
  the paper describes is not where AdaIR's restoration quality on these
  benchmarks comes from. Whatever discriminative, degradation-specific
  behavior the original 300-image analysis found in `FMiM_Statistics` /
  `FMoM_Statistics` (Rain vs. Haze vs. Noise producing measurably different
  internal statistics) must be produced by the **cross-attention and
  gating stages** (FMiM's `channel_cross_l/h`, FMoM's H-L/L-H units, the
  final `channel_cross_agg`) operating on the *un-split* signal, or by the
  Transformer backbone itself — not by any genuine low/high frequency
  separation.
- **Hypothesized** (plausible but not tested here): if AdaIR were
  evaluated at higher resolution (≥640-1024px, e.g. full-resolution photos
  rather than cropped benchmark patches), the mask mechanism would
  activate and might contribute more meaningfully to quality — this
  ablation does not test whether an *activated* mask actually helps
  (that would require rerunning PSNR/SSIM at those resolutions with paired
  ground truth at that resolution, which the current benchmark datasets
  don't provide at ≥640px reliably). Flagged as the natural next
  experiment, not claimed as established.

## 16. Limitations

- `modified_mask`'s `alpha`/`beta` were learned under gradients that flowed
  through the *original* mask formula; this ablation tests "same learned
  gate values, different discretization," not an independently retrained
  model with the corrected formula from scratch. A full retrain was
  explicitly out of scope per the task's Phase-4 instruction (no silent
  retraining) and was not needed since inference-only compatibility held.
- Deep internal-tensor traces (`MGB_Values`, `FMiM_Statistics`,
  `FMoM_Statistics`, `Mechanism_Audit`, raw `.pt` tensors) cover only 9
  representative images (3/degradation), not all 300, to keep runtime and
  storage bounded (4.2GB across 81 tensor bundles); scalar PSNR/SSIM/
  latency/memory cover the full 300 x 3 = 900 runs.
- Peak-memory measurements are confounded by running all three variants in
  one long-lived process (see section 14) and should be treated as
  indicative, not precise.
- The resolution sweep uses one source image (`rain-001.png`) resized to
  each target — activation thresholds could differ modestly for other
  images/content (though the underlying arithmetic — `h//n` vs `alpha*h/n`
  — depends only on feature-map size and the learned `alpha`/`beta`, which
  cluster tightly around 0.5 across the whole 300-image set per the
  original analysis, so this is expected to generalize).
- AFLB1/AFLB2 activation thresholds remain untested empirically (GPU OOM
  above 1024px on this 24GB card blocked reaching the ~1536-3072px inputs
  those AFLBs would need); only algebraically extrapolated.
- No `Transformer_Features`-equivalent sheet (per-stage encoder/decoder
  energy comparison across all 3 variants) was produced for this ablation,
  unlike the original 300-image workbook — scoped out given the
  FMiM/FMoM/AFLB-output statistics already captured the relevant
  downstream signal for the representative images.

## 17. Implications for knowledge distillation

- The FFT-based frequency split is a **near-zero-value teacher signal** at
  the resolutions this project's student will actually be evaluated at
  (mobile/NPU inference on similar crop sizes). Distilling "what frequency
  band does the teacher attend to" would be distilling noise.
- The FMiM/FMoM cross-attention *machinery* (not the frequency split
  feeding it) is a more plausible target — it remains active and produces
  measurably degradation-discriminative statistics (per the original
  300-image PCA/t-SNE) regardless of whether its low/high inputs are
  genuinely split or degenerate.
- `no_frequency`'s AFLB-output statistics differ only slightly from
  `released`'s (section 12) — suggesting a student that skips explicit
  frequency decomposition entirely, but keeps an analogous cross-attention
  gating structure fed by *spatial* (not frequency-domain) high/low-pass
  approximations, could plausibly match teacher behavior without needing
  FFT at all.

## 18. Implications for NPU deployment

- `torch.fft.fft2`/`ifft2` on complex tensors was already identified (in
  the original feasibility study) as the specific op that fails ONNX
  export for the teacher. This ablation adds a quantitative reason to drop
  it rather than work around it: **removing it changes PSNR by <0.004 dB
  on the actual benchmark images**, and even provides a small (~2%)
  latency win. There is no accuracy argument for preserving the FFT path
  in a distilled/exported student.
- The dead-code submodules noted in the audit (`self.conv`, `self.score_gen`
  inside every `FreModule`) are additional evidence that not everything the
  released checkpoint carries weights for is functionally load-bearing —
  worth a similar dead/inert-code check on any other component before
  committing student-architecture effort to replicating it.

## 19. Recommended next experiment

Two candidates, in priority order:

1. **Activated-mask quality test**: rerun `released` vs. `modified_mask`
   (or a further-corrected mask formula) on a small set of genuinely
   high-resolution images (≥1024px, with matched high-res ground truth —
   e.g. a curated subset of DIV2K or similar, not the current cropped
   benchmark sets) to directly test whether an *activated* frequency mask
   improves quality when it actually engages, closing the gap left by
   section 15's "hypothesized" item.
2. **FMoM-without-frequency ablation**: a fourth condition that keeps FMoM
   (H-L/L-H) but feeds it a cheap spatial-domain high/low-pass
   approximation (e.g. a Gaussian-blur split) instead of either the FFT
   split or the current identity/zero split, to test whether the
   cross-attention machinery benefits from *any* meaningful frequency-like
   signal, or is equally indifferent to genuinely spatial substitutes —
   directly informing whether a student needs a frequency-domain
   analog at all (Phase 14, Q7-Q8 territory).
