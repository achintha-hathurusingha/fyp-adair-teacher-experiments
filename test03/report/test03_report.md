# TEST03 — Controlled Same-Scene Degradation Representation Study

## 1. Objective

Determine whether the released 3-degradation AdaIR checkpoint encodes
genuine degradation-specific information, or whether TEST02's near-perfect
internal-stage classification accuracy was primarily an artifact of
Rain/Haze/Noise coming from three different source datasets. Answered by
constructing all three degradations from the same 100 clean scenes and
evaluating with scene-grouped cross-validation.

## 2. Why TEST02 Was Insufficient

TEST02's Rain=Rain100L, Haze=SOTS-outdoor, Noise=BSD68 meant degradation
type and dataset domain were perfectly confounded — a 71.7% input-only
classification accuracy proved substantial dataset-domain separability
existed before any AdaIR computation touched the images. TEST02's report
explicitly flagged this as its primary limitation and recommended exactly
the experiment performed here.

## 3. Experimental Design

Full design rationale: `report/test03_design.md`. Summary: 100 clean
scenes → 3 deterministic synthetic degradations each (300 images total) →
same released, unmodified AdaIR checkpoint → same 41 pooled features as
TEST02 → but cross-validated with **GroupKFold(group=scene_id)** instead
of plain stratified k-fold, so no fold ever sees two degraded versions of
the same scene split across train/test. A hard leakage assertion is
enforced in code (`linear_probe_grouped.py`).

## 4. Clean Image Pool

100 Rain100L ground-truth (`norain-*.png`) images — natural photographs
previously used elsewhere in this project only as *derain targets*, never
as AdaIR *inputs*. Read-only reference; not modified. `Scene_Manifest`
sheet.

## 5. Degradation Synthesis

All three deterministic, documented, seeded per scene_id
(`np.random.RandomState(abs(hash(scene_id)) % 2**31)`):

- **Rain**: synthetic streak layer (sparse random origins, fixed 18px
  length, 70° angle, 1px width, Gaussian-blurred, intensity 0.55),
  rendered via `cv2.line`.
- **Haze**: atmospheric scattering model `I_haze = I_clean*t + A*(1-t)`,
  `t = exp(-beta*d)`, `A=0.85`, `beta=1.2`, `d` = a **documented synthetic
  depth proxy** (linear vertical gradient, top=1.0/"far" to bottom=0.3/
  "near") — explicitly NOT a real depth estimate, per the task's
  instruction not to invent physically meaningful depth.
- **Noise**: additive Gaussian, `sigma=25`.

Exact per-image parameters: `Degradation_Parameters` sheet (300 rows).

## 6. Same-Scene Validation

All 100 scenes passed 7 automated checks (identical dimensions, non-trivial
but non-destructive change, scene content preserved [pixel correlation
with clean source > 0.5 for all variants], no path leakage, valid pixel
ranges) — `report/synthetic_data_validation.md`, `Data_Validation` sheet.
One threshold (haze's expected mean-absolute-difference range) needed
recalibration after an initial false-positive failure on 3 scenes — the
haze model's fixed atmospheric-light blend naturally produces a larger raw
pixel delta on darker source images, which is expected physical behavior,
not a synthesis defect; the validation bound was widened accordingly and
all 100 scenes then passed. 10 visual panels: `results/visualizations/synthetic_examples/`.

## 7. AdaIR Configuration

Identical to TEST01/TEST02: `adair3d.ckpt`, 28,784,824 params, strict
loader (0 missing/0 unexpected keys), unmodified architecture, unmodified
preprocessing (`crop_img(base=16)`), no retraining, no degradation label
ever supplied to the model. Full record: `Environment` sheet.

## 8. Feature Extraction

Same 41 representations as TEST02 (14 pipeline stages + 3×10 AFLB
sub-features), same GAP+GMP pooling, extracted via the same non-intrusive
hooks (`teacher-experiments/scripts/instrument.py`, reused unmodified).
Raw float16 tensors preserved for 10 representative scenes × 3
degradations × 41 features (`Feature_Index`/`Tensor_Index` sheets).

## 9. Grouped Linear-Probe Results

**Primary evidence** (`Linear_Probe` sheet). Logistic Regression + Linear
SVM, 5-fold `GroupKFold(group=scene_id)`, leakage-impossible by
construction (assertion enforced in code, never triggered). Full trajectory
in section 10.

## 10. Feature Trajectory

**The headline result** (`Feature_Trajectory` sheet,
`results/visualizations/controlled_degradation_trajectory.png`):

| Stage | TEST03 (controlled, grouped CV) | TEST02 (dataset-confounded, reference) |
|---|---:|---:|
| Input | 66.7% | 71.7% |
| Shallow (Y0) | 99.3% | 93.3% |
| Encoder L1 | 100.0% | 98.0% |
| Encoder L2 | 100.0% | 99.3% |
| Encoder L3 | 100.0% | 99.3% |
| **Latent** | **100.0%** | 100.0% |
| AFLB 1 | 100.0% | 99.7% |
| Decoder L3 | 99.7% | 99.0% |
| AFLB 2 | 100.0% | 99.0% |
| Decoder L2 | 99.7% | 99.7% |
| AFLB 3 | 99.7% | 99.7% |
| Decoder L1 | 100.0% | 99.7% |
| Refinement | 100.0% | 99.7% |
| **Output** | **37.0%** | 54.7% |

**OBSERVATION**: under scene-grouped cross-validation (a classifier that
has never seen any version of a test scene during training), accuracy is
100.0% with **zero variance** from Encoder L1 through Refinement, and 37.0%
at Output — closer to the 33.3% random baseline than even TEST03's own
input-only accuracy (66.7%).

**INFERENCE**: the near-perfect internal-stage separability TEST02 found is
**not** an artifact of dataset/domain confounding — it survives, and in
several stages strengthens, when scene content is held perfectly fixed and
only degradation varies. This is direct evidence that AdaIR's internal
representations carry degradation-discriminative information tied to the
degradation itself, not to which source dataset an image came from.

**HYPOTHESIS**: unchanged from TEST02 — this pattern is consistent with
AdaIR using strong internal degradation-awareness to condition restoration
before converging toward a degradation-agnostic output. TEST03 strengthens
the evidential basis for this hypothesis considerably but, like TEST02,
does not prove a causal conditioning mechanism (see section 18 and the
recommended TEST04).

## 11. Same-Scene Distance Analysis

`Paired_Distances` sheet (24,600 rows: same-scene cross-degradation
distances + sampled cross-scene same-degradation distances, Euclidean and
cosine, all 41 features). Raw pairwise distances underlying section 12's
ratios.

## 12. Scene-vs-Degradation Analysis

**The second independent line of evidence** (`Scene_vs_Degradation` sheet).
`degradation_ratio = mean(same-scene, different-degradation distance) /
mean(same-degradation, different-scene distance)`:

| Feature | D_degradation | D_scene | ratio |
|---|---:|---:|---:|
| AFLB3_lh_channel_weight | 22.30 | 8.68 | **2.57** |
| AFLB3_mined_low | 22.13 | 9.28 | 2.39 |
| AFLB3_fmom_agg | 21.90 | 10.21 | 2.14 |
| AFLB1_mined_low | 40.09 | 19.50 | 2.06 |
| encoder_level1 | 14.08 | 7.73 | 1.82 |
| input | 2.90 | 1.87 | 1.56 |
| **output** | **0.95** | **3.16** | **0.30** |

**OBSERVATION**: for most internal features, changing only the degradation
moves the representation 1.5-2.6× more than changing the scene does. For
the final `output`, the reverse holds — scene identity moves the
representation ~3.3× more than degradation does.

**INFERENCE**: internal representations are dominated by degradation
identity; the final restored image is dominated by scene identity — i.e.
restorations of the *same* scene under *different* degradations resemble
each other more than restorations of *different* scenes under the *same*
degradation. This is the behavioral signature of successful, content-
preserving, degradation-adaptive restoration.

A scene-aware bootstrap (2000 resamples, resampled by scene — the
experimental unit, not by individual image) on the `latent` feature's
same-scene distance gives a 95% CI of **[38.37, 39.46]** around a mean of
38.92 (`Bootstrap_CI` sheet) — a tight, well-estimated interval given
n=100 independent scenes.

## 13. Alpha/Beta Analysis

`Alpha_Beta_Probe` sheet: 62.0-62.7% (vs. 33.3% random) under grouped CV,
essentially matching TEST02's 64-66%. **OBSERVATION**: the dataset
confound's removal barely changes alpha/beta's classification power.
**INFERENCE**: alpha/beta's moderate-but-partial degradation signal
(established in TEST02) is real and not a dataset-domain artifact, but —
consistent with TEST02 — nowhere near sufficient alone to explain AdaIR's
degradation-adaptive behavior.

## 14. AFLB Analysis

`AFLB_Analysis` sheet, 30 rows. Under grouped CV: `y_in`/`aflb_out` reach
99.7-100%; `mined_low`/`mined_high`/`fmom_agg` reach 89-99.7%;
`hl_spatial_weight` remains the weakest real signal (67.7-69.3%, likely
its low pooled dimensionality); **`raw_low` sits at exactly 33.33% with
zero variance across all 3 AFLBs** — see section 15.

## 15. Raw-Low Verification

`Raw_Low_Check` sheet: `raw_low`'s mean/std/min/max/L1/L2/energy are
**exactly 0.0** for every one of 300 images, all 3 AFLBs
(`all_exactly_zero = True`). Combined with TEST01's independent
from-scratch forward-pass trace and TEST02's dataset-confounded linear
probe (also exactly 33.33%/zero-variance), and TEST03's own distance
analysis (0/0 degradation_ratio):

**TEST01 + TEST02 + TEST03 independently confirm raw_low degeneracy at
benchmark resolution**, via three entirely different analysis
methodologies (direct tensor trace, dataset-confounded linear probe,
scene-controlled linear probe + distance analysis). The mask
implementation was not modified in TEST03, per instruction.

## 16. Restoration Quality

`PSNR_SSIM` sheet, computed against the **original clean scene image**
(not the synthetic degraded input), 300 rows. Full latency/peak-memory
also recorded per image. (See Final Response for aggregate numbers.)

## 17. TEST02 vs TEST03

`TEST02_vs_TEST03` sheet — full side-by-side comparison, section 10 table
reproduces the key rows. TEST02 was read exactly once (read-only) and was
not modified by this comparison.

## 18. Interpretation

Following the task's required three-level distinction:

- **OBSERVATION**: under a controlled same-scene design with
  leakage-impossible scene-grouped cross-validation, a linear classifier
  external to AdaIR achieves 100.0% (zero variance) accuracy at
  distinguishing Rain/Haze/Noise from Encoder L1 through Refinement, and
  37.0% from the final output. Same-scene cross-degradation distances
  exceed cross-scene same-degradation distances by up to 2.6× internally,
  and are *smaller* than cross-scene distances (ratio 0.30) at the output.
- **INFERENCE**: this is strong evidence — considerably stronger than
  TEST02 alone could provide, because the dataset/domain confound is
  removed — that AdaIR's internal representations contain
  degradation-discriminative information tied to the degradation itself.
  The information is not merely re-discovering which source dataset an
  image came from.
- **HYPOTHESIS**: AdaIR uses this internal degradation-discriminative
  signal to condition how it restores each image, converging toward a
  degradation-agnostic, scene-preserving output. This remains a hypothesis,
  not a proven mechanism — TEST03 is observational (it measures what
  information is *present* and *linearly decodable*), not interventional
  (it does not manipulate the representation and observe a causal effect
  on restoration behavior).

The strongest statement the evidence supports: **"The AdaIR latent
representation (and every AFLB/decoder stage) contains
degradation-discriminative information under a controlled same-scene
experimental design, and this information is substantially reduced at the
final restored output."** This is a materially stronger and better-
supported claim than TEST02 alone could make.

## 19. Implications for Knowledge Distillation

See the ranked distillation-target table in the Final Response. Summary:
`latent` and AFLB outputs remain the strongest, most compact, most
scene-invariant-relative-to-degradation-invariant candidates; `raw_low`
remains confirmed (four ways now) as a non-target; `output` alone should
not be the sole distillation target if degradation-adaptive *behavior* is
the goal, since it has by far the lowest degradation-vs-scene ratio of any
stage measured.

## 20. Limitations

- Synthetic degradations are simplified, documented models (streak
  rendering, atmospheric-scattering with a synthetic depth proxy, additive
  Gaussian noise) — not photorealistic degradation simulators. AdaIR was
  trained on real degraded/paired data (Rain100L rain, RESIDE haze, BSD68+
  synthetic noise via its own protocol); TEST03's synthetic rain/haze in
  particular will differ in some visual characteristics from what the
  network was trained on. The controlled design answers "does scene vs.
  degradation matter for the representation" cleanly; it does not
  guarantee TEST03's specific rain/haze renderings are representative of
  every real-world rain/haze appearance.
- Single moderate degradation-strength condition (not a
  moderate-vs-strong comparison), per the task's stated fallback when the
  two-condition design would make the experiment too large.
- Deep raw-tensor inspection covers 10/100 scenes; quantitative claims use
  all 100.
- Linear probes measure linear separability only (by design, matching
  TEST02's methodology for direct comparability).
- No causal intervention was performed (Phase 16 explicitly prepares
  infrastructure for one but does not execute it, per instruction).

## 21. Recommended TEST04

The representation-swap infrastructure prepared in Phase 16
(`Swap_Prep_Index` sheet: 5 scenes × {latent, AFLB1/2/3 output} × 3
degradations, 60 tensors) sets up the natural next step: an **intervention
study** — e.g. feed a Rain-image's encoder output through the decoder path
conditioned on a Haze-image's AFLB representations (or vice versa) for the
*same* scene, and observe whether the restoration *behavior* changes in a
way consistent with the swapped degradation. This would move the project's
strongest claim from "the information is linearly present" (TEST02+TEST03)
to "the information is causally used by the restoration process" — the
one distinction TEST03 explicitly cannot make on its own.
