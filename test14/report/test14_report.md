# TEST14 — Frequency-Augmented Conditional Operator

## 1. Motivation

TEST12 established the validated conditional operator: `a = G(e_D, φ(F))`,
fixed rank-2 basis. TEST13 showed that adapting the basis itself is
harmful and should not be extended. TEST14 asks an orthogonal question:
does a compact frequency descriptor, extracted directly from the degraded
input, provide information beyond `e_D` and `φ(F)` — without reproducing
AdaIR's own (previously shown irrelevant) frequency branch?

## 2. Relationship to TEST06-R and Previous Frequency Findings

TEST06-R established that AdaIR's own internal frequency-path tensors
(`raw_low`, `raw_high`, the FMiM/FMoM stages) are practically irrelevant to
its final restoration output for the tested checkpoint. TEST14 does **not**
reopen that finding or claim it resurrects AdaIR's frequency mechanism —
it asks a deliberately independent question using a from-scratch
descriptor, computed without any AdaIR internals.

## 3. Frequency Descriptor Definition

`q_F ∈ R^8`: luminance `Y = 0.299R+0.587G+0.114B` → `|FFT2(Y)|²` →
`fftshift` → normalize by total spectral energy → sum into 8 radial bands
(edges at 0, 1/16, 2/16, 3/16, 4/16, 6/16, 8/16, 12/16, and 1.0 = Nyquist,
where radius is normalized so the axis edge equals 1.0). Corner
frequencies beyond the axis-Nyquist radius (up to `√2`) are excluded from
all 8 bands by design, so `sum(q_F)` is close to but not exactly 1
(observed mean 0.996, min 0.884 across all crops).

## 4. Descriptor Validity

Audited **before** any student training (mandatory gate): per-band std
non-trivial (band1 std 0.0995, other bands 0.012-0.025), pairwise cosine
similarity across *different* images = 0.993 (high, but expected — natural
image spectra are dominated by low-frequency/DC content; this did not
trigger the collapse threshold since band-level variance was clearly
non-zero), degradation-classification probe accuracy = **78.4%** (well
above the 33% chance level for 3 classes). Haze's spectrum is markedly
more concentrated at the lowest band (98.7% of energy in band1) than
Rain/Noise (89.0%/89.4%) — a sensible, physically-plausible signature
(haze is a smooth, low-frequency degradation; rain/noise introduce
higher-frequency texture). **Audit passed; training proceeded.**

## 5. Models

- **A**: baseline locked NAFNet.
- **F2**: TEST12's validated operator (fixed basis, `a=G([e_D;φ(F)])`) —
  the reference.
- **T14**: F2 + `q_F` concatenated into the coefficient generator's input
  only: `a=G([e_D;φ(F);q_F])`. Rank, basis, NAFNet, teacher embedding,
  and loss all unchanged from F2.

## 6. Training

Exact TEST07-B/12 dataset/split, reused read-only. Adam, LR=2e-4, batch=8,
50 epochs, seeds {0,1,2}, `λKD=0.1`, no new losses. All 9 runs completed
cleanly, zero NaN/Inf. Models A/F2 reproduce TEST12/13's own numbers
exactly (deterministic).

## 7. Restoration Results

| Model | Mean PSNR (dB) | Mean SSIM |
|---|---|---|
| A | 27.315 | 0.815 |
| F2 | 27.116 | 0.830 |
| T14 | 27.044 | 0.827 |

| Comparison | Mean ΔPSNR (dB) | Same-sign (of 3) | Mean ΔSSIM | Same-sign |
|---|---|---|---|---|
| T14 − F2 | -0.072 | 2/3 | -0.0034 | **3/3** |
| T14 − A | -0.270 | 2/3 | +0.0116 | 2/3 |
| F2 − A | -0.199 | 2/3 | +0.0150 | 2/3 |

T14-F2 is small and directionally inconsistent for PSNR (one seed +0.13,
two seeds -0.20/-0.14) — this is a materially different, much more
ambiguous signal than TEST13's clean, consistent regression, but it is not
a positive result either. SSIM is consistently slightly negative (3/3) but
the magnitude (-0.003) is negligible in practical terms.

## 8. Frequency Causal Controls

The four mandatory evaluation-time controls, no retraining:

| Condition | Mean PSNR (dB) | Δ from Normal |
|---|---|---|
| Normal (correct `q_F`) | 27.010 | — |
| Zero `q_F` | 27.010 | **~0.000** |
| Mean `q_F` (training-set average) | 27.010 | **~0.000** |
| Shuffled `q_F` (wrong image) | 27.009 | **~0.001** |

**All four conditions are statistically indistinguishable**, both overall
and per-degradation (Haze: 23.338/23.339/23.336/23.333; Noise:
28.555/28.555/28.555/28.555; Rain: 29.138/29.137/29.139/29.139 — every
difference is in the third decimal place). Per the task's pre-specified
interpretation: *"if correct ≈ zero ≈ mean ≈ shuffled, then frequency adds
no useful information."* This is exactly what was observed — the trained
coefficient generator learned to produce essentially the same output
regardless of what `q_F` actually contains.

## 9. Degradation-Specific Frequency Effects

T14-F2 per degradation: Haze -0.440dB, Noise +0.019dB, Rain +0.205dB — none
represent a meaningful, causally-attributable improvement given Section 8's
finding that `q_F`'s content has no measurable causal effect. Any
degradation-level differences between T14 and F2 are attributable to
different training trajectories from the larger coefficient-head
architecture, not to `q_F`'s information content.

## 10. Frequency vs. Degradation Embedding

The central redundancy question. Grouped degradation-classification probe
accuracy using different feature sets:

| Features | Dim | Accuracy |
|---|---|---|
| `q_F` alone | 8 | 78.4% |
| `e_D` alone | 16 | 96.4% |
| `φ(F)` (PCA-16) alone | 16 | 97.9% |
| `[e_D, q_F]` | 24 | 97.0% |
| `[φ(F), q_F]` | 24 | 97.8% |
| `[e_D, φ(F)]` | 32 | **98.2%** |
| `[e_D, φ(F), q_F]` | 40 | **98.2%** |

**Adding `q_F` to `[e_D, φ(F)]` produces exactly zero additional probe
accuracy** (98.2% either way). Canonical correlation analysis between
`e_D` and `q_F` gives a top correlation of **0.867** — high. Maximum
pairwise linear correlation between individual `e_D` dimensions and `q_F`
bands is 0.575 (mean 0.137); between `φ(F)`-PCA dimensions and `q_F` bands,
0.658 (mean 0.147). `q_F` is a real signal on its own (78.4% probe
accuracy, well above chance) — but its information is a near-complete
subset of what `e_D` (the teacher-distilled degradation embedding) already
encodes.

## 11. Frequency vs. Spatial Content

Between-degradation `q_F` distance (mean 0.0720) is nearly identical to
within-scene, cross-degradation `q_F` distance (mean 0.0745) — `q_F`
varies almost as much when only the degradation changes (same scene) as
when comparing different degradations across different scenes. This
indicates `q_F` is driven predominantly by **degradation identity**, not by
scene-specific spatial content — consistent with Section 10's finding that
it is redundant with `e_D` specifically (not primarily with `φ(F)`,
though `φ(F)` alone already outperforms `q_F` alone: 97.9% vs 78.4%).

## 12. Operator Coefficient Effects

Adding `q_F` to the generator input does change the learned coefficients
noticeably (`coeff_magnitude_T14` vs `coeff_magnitude_F2`: Haze 2.02 vs
1.82, Noise 5.47 vs 5.86, Rain 6.11 vs 6.30 — different, not dramatically
so). This confirms the network's coefficient-generation *did* adapt to the
larger input space during training — but Section 8 shows this adaptation
is **not** causally driven by `q_F`'s actual per-sample value (shuffling it
changes nothing), so the coefficient differences reflect a different
training trajectory/local optimum for the larger head, not genuine
frequency-responsive behavior.

## 13. Complexity

| Model | Params | Extra vs. F2 | MACs @128px |
|---|---|---|---|
| A | 7,371,923 | — | 1,033,040,896 |
| F2 | 7,398,149 | — | 1,033,131,584 |
| T14 | 7,398,405 | +256 | 1,033,262,912 |

NN parameter overhead is trivial (+256, just the expanded first
coefficient-head layer). The FFT descriptor computation itself is a
**separate** cost, not captured by parameter count, and is explicitly
**not** claimed to be NPU-friendly — FFT is a poor fit for Snapdragon
Hexagon INT8 execution. Since the signal proved unhelpful, no NPU-friendly
replacement (DCT, fixed cosine filters, learned filter bank) needs to be
pursued as a follow-up to this specific result.

## 14. Limitations

- N=3 seeds; T14-F2's PSNR delta is small and only 2/3 consistent in sign
  — genuinely ambiguous at the seed level, though the causal-control and
  redundancy evidence (Sections 8, 10) are unambiguous regardless of the
  restoration-delta's own statistical uncertainty.
- Only one specific descriptor (8-band radial FFT-magnitude profile) and
  one specific injection point (coefficient generator only) were tested.
  A richer descriptor or a different injection point (e.g., into `φ(F)`'s
  own summary, or conditioning the basis rather than only coefficients)
  might behave differently — but per the task's explicit scope, this
  experiment intentionally isolated the simplest, most direct test.
- The log-energy alternative descriptor (specified as an analysis-only
  fallback in case the primary descriptor failed its validity check) was
  not needed, since the primary descriptor passed.

## 15. GO / NO-GO

Per the task's decision rule:

- **Strong GO** requires `T14>F2`, `correct>shuffled`, `correct>zero`, and
  `q_F` beyond `e_D+φ(F)`. **None of these are met.**
- **Partial GO** requires `q_F` causally useful for one degradation
  (especially Haze) with small overall gain. **Not met** — no degradation
  shows a causally-attributable improvement; Section 8's controls are flat
  for every degradation individually.
- **NO-GO** requires `T14≈F2` and `correct≈zero≈mean≈shuffled`. **Met** —
  T14-F2 is small/inconsistent, and all four controls are statistically
  identical.
- **Redundant frequency** requires `q_F` strongly correlating with `e_D`
  or `φ(F)` and adding it changing nothing. **Also met** — CCA=0.867 with
  `e_D`, and zero incremental probe accuracy.

**Decision: NO-GO, additionally classified as REDUNDANT FREQUENCY.** Both
of the task's negative-outcome categories apply simultaneously, and they
are mutually reinforcing: `q_F` is a real, non-trivial, degradation-
discriminative signal in isolation, but it is redundant with information
the teacher-distilled `e_D` already provides, and the trained network
correctly learned to ignore it. **Close this branch** — per the task's
explicit rule, a negative frequency result is acceptable and should be
preserved, not discarded.

## Recommendation for TEST15

Do not pursue an NPU-friendly frequency-descriptor replacement (DCT, fixed
cosine filters, learned filter bank) as a direct follow-up — the
underlying informational signal was shown to be redundant with `e_D`
before reaching the point where implementation efficiency would matter.
Given the accumulated series (TEST09 FiLM, TEST10/10-R trajectory
distillation, TEST13 basis adaptation, TEST14 frequency augmentation have
all failed to beat F2/T12's validated fixed-basis, content-conditioned
operator), the evidence increasingly points toward F2's mechanism being a
stable local optimum for this specific student architecture and dataset.
Future work should either (a) accept F2 as the validated mechanism and
shift focus to deployment-oriented work (the NPU-latency profiling and
INT8 quantization explicitly deferred throughout TEST08-14), or (b) if
further restoration-quality gains are still sought, test a structurally
different mechanism family entirely (e.g., restoration-quality-targeted
regularization, or output-space supervision as recommended in TEST13's own
next-direction note) rather than continuing to vary inputs to the same
validated operator.
