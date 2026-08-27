# TEST18 — AdaIR Component Ablation (Paper-Style Retraining) + Frequency-Domain Diagnostics

## 1. Objective

TEST01-TEST06R established, with increasing statistical rigor, that
AdaIR's frequency mechanism is not causally load-bearing for restoration
quality on the **frozen, released checkpoint, at inference time**. This
was challenged on the grounds that the frequency modules (FMiM/FMoM) sit
in the main forward path and are the mechanism the paper is named after —
a claim about the architecture, not about one frozen checkpoint's
behavior. TEST18 tests that stronger claim directly, the way the paper's
own ablation (§4.4, Table 7) does: by **retraining AdaIR from scratch**,
once per component-ablation variant, and measuring both restoration
quality and — for the first time in this project — what the frequency
mechanism is actually doing internally, visualized directly, during a
model that was trained (not frozen) with that mechanism present.

## 2. Models

Five variants, matching Table 7's structure exactly (`test18/scripts/ablatable_model.py`):

| Variant | mask_mode | use_lh | use_hl | Params |
|---|---|---|---|---:|
| A_baseline | none | — | — | 26,126,644 |
| B_fixed_mask | fixed (10×10) | False | False | 28,717,240 |
| C_learned_mask | learned (MGB) | False | False | 28,741,600 |
| D_plus_lh | learned | True | False | 28,765,792 |
| E_full (= released AdaIR) | learned | True | True | 28,766,086 |

Validation of the reimplementation: baseline→full adds 2,639,442
parameters — matching the paper's own stated "2.64M parameter overhead"
almost exactly.

## 3. Training

3-in-1 degradation setting (dehaze+derain+denoise), real data: dehaze on
a real, seeded 10,000-image subsample of the actual RESIDE OTS training
set (72,135 total confirmed present and extracted, matching the paper's
own reported count); derain on the real Rain100L/RainTrainL 200 pairs;
denoise on 100 real DIV2K clean images with online Gaussian noise
(BSD400/WED, the paper's actual denoise corpus, were not found anywhere
on devon — documented substitution). AdamW, lr=2e-4,
`LinearWarmupCosineAnnealingLR` (15-epoch warmup), L1 loss, batch=8,
AMP — all matching the released `train.py`. **8 epochs per variant**
(rescoped down from an initial 30-epoch target after a timing
calibration showed 30 epochs on the full corpus would take ~21
hours/variant even with AMP — see `TEST18_PLAN.md` §4 for the full
calibration). Single seed per variant. All 5 variants trained
sequentially (AdaIR is far larger than this project's usual NAFNet-M
student; concurrent multi-variant training doesn't fit one GPU),
completing in 8.05 hours total (19:04 → 03:07).

Final training-loss trajectory:

| Variant | Final epoch mean L1 |
|---|---:|
| A_baseline | 0.0217 |
| B_fixed_mask | 0.0214 |
| C_learned_mask | 0.0212 |
| D_plus_lh | 0.0219 |
| E_full | 0.0216 |

Not monotonic — flagged live during training (D breaks the improving
trend), and this holds up in the real quality eval below, not just as
training-loss noise.

## 4. Restoration Results — the central, surprising finding

Real held-out test data throughout (genuinely disjoint from training):
dehaze on RESIDE SOTS-outdoor test (a different RESIDE split from the
OTS training subsample); derain on Rain100L's own official test split
(100 pairs, disjoint from RainTrainL); denoise on SOTS's clean images
with synthetic noise (σ=15/25/50) as a documented CBSD68 substitute.

| Variant | Overall PSNR | Overall SSIM |
|---|---:|---:|
| A_baseline | 28.572 | 0.8048 |
| **B_fixed_mask** | **28.738** | **0.8051** |
| C_learned_mask | 27.953 | 0.7833 |
| D_plus_lh | 28.542 | 0.7951 |
| E_full | 28.674 | 0.7975 |

**This does NOT reproduce the paper's monotonic a→e improvement.**
Reported exactly as measured, not smoothed over:

- **B_fixed_mask is the best-performing variant overall** — better than
  the full released architecture (E_full) on both PSNR and SSIM.
- **C_learned_mask is the WORST of all 5** — worse than the no-AFLB
  baseline on both metrics. Adding the "adaptive" mask actively hurt
  restoration quality relative to both the baseline and the fixed-mask
  variant, in this reproduction.
- D_plus_lh and E_full partially recover from C but neither reaches
  B_fixed_mask's level.

## 5. Per-Degradation Breakdown

| Variant | Dehaze PSNR | Derain PSNR | Denoise σ15 PSNR | Denoise σ25 PSNR | Denoise σ50 PSNR |
|---|---:|---:|---:|---:|---:|
| A_baseline | 26.11 | 32.19 | 31.01 | 28.50 | 25.06 |
| B_fixed_mask | 26.32 | 32.38 | 31.31 | 28.71 | 24.97 |
| C_learned_mask | 26.42 | **31.13** | **29.20** | 28.33 | 24.69 |
| D_plus_lh | **26.72** | 32.15 | 30.67 | 28.26 | 24.89 |
| E_full | 26.19 | 32.38 | 31.32 | 28.80 | 24.67 |

**Dehaze actually shows a partial upward trend** (A<B<C<D, dropping only
at E) — the one degradation where the paper's expected direction holds
partway. **Derain and denoise are where C_learned_mask's regression is
concentrated** (derain: -1.25dB vs. B; denoise σ15: -2.11dB vs. B) — the
degenerate mask (see §6) appears to hurt non-dehaze tasks more, plausibly
because dehaze's own training data is what the mask's threshold-predictor
sees most often (72,135 dehaze samples in the full corpus vs. 200
derain/100 denoise sources), so whatever the MGB head is learning is most
dehaze-shaped.

## 6. Frequency-Domain Diagnostics — the root-cause finding

`test18/scripts/frequency_diagrams.py` extracted the actual mask, mined
high/low features, FMoM output, and AFLB output spectrum at all 3 AFLB
positions, for one representative image per degradation (Rain/Haze/
Noise), for every variant — 36 composite figures in
`results/frequency_diagrams/`.

**Central finding**: for every variant with `mask_mode="learned"`
(C, D, E), **the mask is uniformly zero (degenerate) at every one of the
3 AFLB positions**, confirmed visually (flat gray panels, no spatial
structure) for both AFLB1 (deepest, smallest spatial resolution) and
AFLB3 (shallowest, largest resolution) — checked directly, not assumed.

This is not a new finding in isolation — it is an **exact, independent
reproduction of TEST01 and TEST06-R's original finding on the frozen
released checkpoint**, this time observed directly in freshly-trained
models via a completely different method (live diagnostic hooks during a
forward pass on a just-trained checkpoint, not intervention on a frozen
one). The root cause is architectural, not weight-dependent: `FreModule.fft()`
computes the mask half-width as `h_ = (h // 128 * threshold).int()`;
at the 256px diagnostic resolution used here (matching the 128px training
crop scale), `h // 128` floors to 0 or 1 for every AFLB's actual feature
resolution, making the mask's size effectively fixed at (near-)zero
**regardless of what the threshold-predicting network learns** — the
learned component has no way to express a non-degenerate mask at this
scale, trained or not.

**This plausibly explains §4's central finding**: C_learned_mask's mask
never becomes spatially selective, so its "FMiM" mining collapses to
(effectively) an all-high-frequency split — a real difference from
B_fixed_mask's honest, non-trivial 10×10 low-frequency box, which stays
non-degenerate by construction, not by learning. A degenerate-but-still-
parameterized mechanism (C) apparently gives the network something to
optimize around that hurts more than either not having the mechanism at
all (A) or having a crude-but-real one (B).

## 7. Reconciliation with TEST01-06R

This is not a contradiction of TEST01-06R's frozen-checkpoint null
finding — it is an **extension of it to a new setting**. TEST01-06R
showed the frequency-adaptive mechanism doesn't behave adaptively for the
*frozen released checkpoint* at inference time. TEST18 shows the same
degeneracy is present *during and after fresh training* at practical
resolutions, and — for the first time — connects that degeneracy
causally to a **measurable quality regression** (C_learned_mask
underperforming both A and B), not just a null causal-intervention
effect. The frequency modules do execute (confirmed, matching Himeth's
original architectural point) — but the specific mask-adaptivity
mechanism inside them is not doing what its name promises, in either the
frozen or the freshly-trained setting, at the resolutions actually used.

## 8. Limitations

- **Single seed per variant** — no statistical confidence on whether
  B>E or C<A would replicate across seeds. This is the most important
  caveat: the central finding here is a real, measured result, not yet a
  statistically-confirmed one.
- **8 epochs, not the paper's 20 (ablation) or 150 (main)** — training
  loss was still decreasing at epoch 8 for most variants; the ranking
  could shift with more training. However, this doesn't change §6's
  finding (mask degeneracy is architectural, not a training-time
  convergence issue — it holds by construction at this resolution).
- **Dehaze subsampled to 10k of 72,135 real images** — a real but
  reduced-diversity training set.
- **Denoise trained on DIV2K (100 images), not BSD400+WED** — genuinely
  smaller and less diverse than the paper's corpus.
- **Frequency diagrams run at 256px** — below the resolution TEST06
  showed CAN produce a non-degenerate mask at AFLB3 (≥768px input). A
  natural, cheap follow-up: re-run `frequency_diagrams.py` at 1024px+ on
  these SAME trained checkpoints to see whether the learned mask
  *would* become adaptive at higher resolution, even though it never saw
  that regime during training.

## 9. Small Plan After Results

Per `TEST18_PLAN.md` §8's decision framework, filled in now that real
data exists (not left as a template):

The framework's first branch predicted: *"if E_full reproduces the
paper's trend AND diagrams show structured mask behavior → reconcile
'matters at training time' vs. 'doesn't matter for frozen inference.'"*
That branch does not apply — E_full does **not** reproduce the paper's
trend, and the diagrams show the **opposite** of structured mask
behavior (uniform degeneracy). The framework's third branch is the one
that actually matches what was found: *"if the frequency diagrams show
the learned mask converging to degenerate behavior even though PSNR
still improves a→e, that suggests the ablated components help via a
different mechanism than genuine frequency-adaptivity."* Refined based
on what was actually measured: **PSNR does NOT improve monotonically
a→e, and the mask IS degenerate** — a more clear-cut result than the
framework anticipated, not a partial match requiring further disambiguation.

**Concrete next steps, in priority order**:

1. **3-seed replication of just A, B, C** (the three variants that
   produced the surprising ranking) at the same 8-epoch/10k-image scale,
   to establish whether B>E and C<A are real effects or single-seed
   noise. Cheapest, highest-value follow-up — reuses everything built
   here.
2. **Re-run `frequency_diagrams.py` at 1024px+ input** on the existing
   trained checkpoints (no retraining needed) to test whether the mask
   becomes non-degenerate outside the training resolution regime, and
   whether that changes anything about C's internal behavior.
3. **Do NOT prioritize scaling up epochs/data before (1)** — more
   training compute is expensive and the single-seed result already
   raises a specific, cheap-to-test question (does the ranking replicate)
   that should be answered first.

## 10. GO / NO-GO

**Reconciliation verdict**: TEST01-06R's frozen-checkpoint null finding
is **not overturned** — it is independently corroborated from a new
angle (fresh training, direct internal diagnostics) and now has a
plausible causal explanation for *why* it holds (the mask's mathematical
degeneracy at practical resolutions is architectural, not an artifact of
using a frozen checkpoint). The counter-argument that "the modules
execute at inference, so they must matter" is correct about execution and
incorrect about the adaptivity claim specifically — confirmed here in a
setting (fresh training) that the counter-argument's own logic would have
predicted should show adaptive behavior if it were architecturally
present.

**This project's own reproduction does not currently support building a
student mechanism around AdaIR's specific learned-mask design** — a
crude fixed mask outperformed it here. This is consistent with, and now
mechanistically explains, the earlier finding (TEST05.5 onward) that this
project's actual distillation signal (`e_D`, a compact PCA-16 projection
of `latent_pre`) never depended on the frequency path being real.
