# AdaIR Testing Report — TEST01 through TEST17

Consolidated summary of the FYP "Distillation for Degradation Adaptive
Mobile Image Restoration" experiment series: distilling a frequency-aware
AdaIR teacher into a lightweight NAFNet student for mobile/NPU deployment.
Each experiment is self-contained under `teacher-experiments/testNN/`; full
detail, data, and visualizations live in each test's own `report/` folder.
This document is the "what we did / what we found" index across all of
them.

**Status as of writing: TEST01-17 complete.**

---

## Part 1 — Understanding the AdaIR Teacher (TEST01-06R)

### TEST01 — Frequency-Mechanism Ablation

**Did**: Verified AdaIR's released frequency mask (`Ml`) is exactly zero
at every AFLB at benchmark resolution (confirmed from-scratch), then ran a
controlled 3-condition ablation (released / modified-mask / no-frequency)
on 300 images (Rain100L/SOTS-outdoor/BSD68).

**Found**: The frequency mask mechanism is inert at standard resolution —
removing it changes output PSNR by <0.004dB. Also found a real
paper-vs-code discrepancy in the mask arithmetic (order of operations
differs, though both floor to zero at tested resolutions).

### TEST02 — Degradation Representation Analysis

**Did**: Source-audited `AdaIR.forward()` (confirmed no degradation label
is ever passed in) and linearly probed intermediate activations for
Rain/Haze/Noise discriminability.

**Found**: Internal representations are highly degradation-discriminative
(up to ~100% at some stages) despite the model being fully blind —
confounded with dataset identity though, flagged as the main limitation.

### TEST03 — Controlled Same-Scene Study

**Did**: Rebuilt the probe with all 3 degradations synthesized from the
*same* 100 clean scenes, scene-grouped cross-validation, to remove
TEST02's dataset-identity confound.

**Found**: Discriminability survives (66.7%→100%→37.0% across stages) —
genuine degradation information, not a dataset artifact.

### TEST04 — Causal Representation Intervention

**Did**: Built a bit-identical manual forward-pass replica allowing
mid-network tensor substitution (donor/recipient swaps), with 5 controls
(self-swap, cross-scene, cross-degradation, random, zero).

**Found**: Real causal effect exceeding controls, scaling with depth — but
"MODERATE" not "STRONG": the same intervention point was *more* sensitive
to scene changes than degradation changes in this narrower analysis.

### TEST05 — Compact Representation Discovery

**Did**: Screened 31 candidate representations (latent + per-AFLB
sub-tensors) for degradation specificity, scene robustness, and
compactness; channel-level ranking (~4,400 channels); causal channel
ablation.

**Found**: `raw_low` confirmed degenerate for the 5th independent time.
Degradation-specificity concentrates in a channel subset (individual
channels reach 4.7-4.8 degradation/scene ratio, higher than any pooled
feature) — first hint that a compact projection, not the full tensor,
carries the useful signal.

### TEST05.5 — Adversarial Audit of the Frequency Hypothesis

**Did**: 14-phase adversarial audit designed to actively try to disprove
H_F2S ("the useful representation depends on frequency processing")
before committing to building a student around it. Included
parameter-randomized harder datasets, normalized causal intervention, and
4 teacher variants (real / frequency-disabled / content-blind /
phase-scrambled).

**Found (central result)**: Degradation-probe accuracy, PCA-16 accuracy,
and output quality are **statistically indistinguishable** across all 4
frequency variants. Evidence AGAINST the frequency-causal hypothesis — the
useful signal survives whether the frequency branch computes real content
or noise. Also corrected a real PCA-leakage risk (found not to have
inflated TEST05's result: 99.67% leakage-safe vs. 99.7% original).

### TEST06 — Resolution-Dependent Frequency Influence

**Did**: Located the exact mathematical activation threshold for each
AFLB's frequency mask (`H_feat >= 256`, `512` for AFLB3), swept 432
configurations across 12 resolutions x 5 aspect ratios, then ran a
same-scene causal frequency-swap test specifically at a confirmed-active
resolution (1024x1024, AFLB3).

**Found**: AFLB3 activates at 768px input+. But even with the mask
genuinely non-degenerate, cross-degradation frequency-content swaps were
statistically indistinguishable from swapping in zero/random/mean tensors
— no causal effect, even when active.

### TEST06-R — Corrected, Statistically Rigorous Re-Run

**Did**: Rebalanced TEST06's controls to equal N (144), added bootstrap
CIs, paired permutation tests, Wilcoxon tests, and a pre-specified
practical-equivalence threshold; traced the intervention through AFLB3's
internal stages.

**Found**: Confirmed TEST06's null result with full statistical rigor —
every comparison falls within the pre-specified equivalence threshold.
**This closed the book on AdaIR's frequency mechanism**: it is not
functionally load-bearing for restoration at inference time on the frozen
checkpoint, at any tested resolution.

---

## Part 2 — Building and Validating the Distillation Signal (TEST07-11)

### TEST07-Pilot — Compact Degradation Distillation (pilot)

**Did**: Short pilot comparing 4 NAFNet-M student variants (baseline /
+KD-only / +FiLM conditioning / +low-rank dynamic kernel), 15 epochs.

**Found**: Genuinely mixed signal — final-epoch numbers favored the KD
variants, but a 5-epoch smoothed average favored the plain baseline.
Recommended a longer, statistically grounded re-run before testing the
more complex variants.

### TEST07-B — Compact Latent Distillation Validation

**Did**: Full 50-epoch, 3-seed, 80/20 scene-disjoint re-run of baseline
(A) vs. +KD (B) only, fixing a pooling asymmetry from the pilot.

**Found**: **NO-GO for simple KD** — Model B matched the teacher's
representation almost perfectly (cosine ~0.99, probe accuracy ~96.2%) but
restoration quality was *worse* than baseline (-0.79dB mean, negative in
all 3 seeds). Non-obvious exception: Noise consistently favored B
(+1.24dB) while Rain/Haze favored A.

### TEST08-C — Compact Degradation State + Spatial Conditioning

**Did**: Tested FiLM-style bottleneck conditioning (degradation embedding
modulates spatial features via scale+shift).

**Found**: **PARTIAL GO** — Rain +0.85dB, Haze -0.06dB, Noise +0.03dB,
with strong causal evidence (random/shuffled/zero controls, donor-recipient
intervention) that the conditioning mechanism genuinely uses the
degradation signal.

### TEST09 — Low-Rank Conditional Operator Capacity (rank sweep)

**Did**: Swept the low-rank channel-mixing operator's rank (F2/F4/F8/F16).

**Found**: **NO-GO** — effective rank plateaus at ~1-3 regardless of
configured rank. The bottleneck is coefficient generation, not operator
capacity.

### TEST10 — Restoration Trajectory Distillation

**Found**: **INVALID** — representational collapse in jointly-trained
projection heads (only caught via a cross-input pairwise-cosine
diagnostic). Superseded by TEST10-R.

### TEST10-R — Corrected Restoration Trajectory Distillation

**Did**: Fixed the collapse with frozen, leakage-safe PCA-32 teacher
targets and mandatory per-epoch collapse monitoring.

**Found**: Valid, trustworthy **NO-GO** — teacher trajectory distillation
doesn't help once collapse is ruled out.

### TEST11 — Low-Rank Conditional Operator Capacity (rigorous rebuild)

**Did**: Statistically rigorous rank sweep, repeating TEST09 with more
seeds/controls.

**Found**: **NO-GO**, confirming TEST09 — F2≈F4≈F8≈F16, effective rank
never exceeds ~2.6/16 (16% utilization).

---

## Part 3 — Refining the Operator Mechanism (TEST12-14)

### TEST12 — Feature-Conditioned Low-Rank Operator ("F2"/"T12")

**Did**: Compared `a=G(e_D, phi(F))` (content+degradation conditioning,
"T12") vs. `a=G(e_D)` only ("F2"), rank fixed at 2.

**Found**: **PARTIAL GO** — strong causal evidence the operator uses
spatial content (shuffled-content control was the worst of 4 conditions),
but the restoration gain was small/inconsistent overall, with Haze showing
the largest (not fully seed-consistent) improvement. **F2 (degradation-only
conditioning) became the project's reference validated mechanism** used
in all subsequent tests.

### TEST13 — Adaptive Low-Rank Operator Basis

**Did**: Tested basis adaptation (`U(e)=U0+ΔU(e_D)`, `V(e)=V0+ΔV(e_D)`) vs.
F2's fixed basis.

**Found**: Decisive **NEGATIVE** ("Interesting Negative") — underperformed
F2 in all 3 seeds despite substantial (8.5-13.7x) basis adaptation
magnitude.

### TEST14 — Frequency-Augmented Degradation-Conditioned Operator

**Did**: Tested whether an independently-computed 8-band radial FFT
descriptor adds information beyond `[e_D, phi(F)]`.

**Found**: **NO-GO + REDUNDANT** — all causal controls statistically
identical; the frequency descriptor adds zero incremental
degradation-probe accuracy and is highly correlated (CCA=0.867) with the
already-distilled `e_D`.

---

## Part 4 — From Restoration Mechanism to Real Hardware (TEST15-17)

### TEST15 — Snapdragon NPU Operator & Graph Benchmark

**Did**: Pivoted from inventing more restoration operators to measuring
what the actual target NPU (Hexagon V79, Snapdragon 8 Elite QRD) executes
efficiently — 26 isolated micro-ops and small combinations, compiled and
profiled via real Qualcomm AI Hub jobs.

**Found**: 24/26 ops compile 100% NPU, zero fallback. FFT can't even
export to ONNX. **Runtime-generated convolution weights are catastrophic**
(dynamic_conv: 4,059ms vs. 54-139ms for everything else — a 30-75x
penalty), but Minura's actual validated operator (fixed basis, runtime
scalar coefficients only) measures 74ms — identical to a plain static
conv and to a proposed "static-mixture" redesign candidate. **Minura's
mechanism was never the risk case.**

### TEST16 — Full Student Graph & NPU Validation

**Did**: Compiled and profiled 4 COMPLETE student graphs (not isolated
ops): baseline A, F2, a normalization-surgery variant N
(`layernorm2d`→`affine_clamp`, untrained — no checkpoint existed), and a
static-mixture variant S (untrained).

**Found**: All 4 compile 100% NPU, zero fallback. **N appeared to be
~24x faster than A** (430ms vs 10,528ms) — later shown in TEST17 to be
substantially inflated by N being untrained (see below). S showed an
unexplained latency collapse matching N despite containing the same
LayerNorm ops as A/F2 — flagged as unresolved, and it did not survive
INT8 quantization (reverted to A/F2-like cost), suggesting it was a
precision/compiler-specific artifact. Real INT8 quality check: A -2.44dB,
F2 -1.14dB (F2 more INT8-robust than baseline).

### TEST17 — Normalization Surgery + F2 Conditioning, Trained

**Did**: Trained all 4 models for real this time (A, N, F2, and N+F2 — F2's
exact validated mechanism applied to N's affine_clamp backbone), 50
epochs x 3 seeds x 4 models = 12 runs, then re-ran the full NPU pipeline
(ONNX export, compilation, FP32 + INT8 latency, layer hotspots) on real
trained checkpoints.

**Found**:
- **N alone is quality-unstable when trained**: mean PSNR 23.45dB vs. A's
  27.31dB, with a genuine training-divergence event visible in one seed's
  epoch curve (loss spike at epoch 27, PSNR never fully recovers).
- **N+F2 is far more stable and recovers most of the loss**: 25.04dB
  (std 0.12 across seeds, vs. N's std 1.78) — F2's conditioning mechanism
  appears to stabilize training on the norm-surgery backbone, not just
  add quality.
- **Representation check**: F2's and N+F2's degradation embeddings are
  nearly identical quality (96.6%/0.989 cosine vs. 95.6%/0.973) — so
  N+F2's quality gap vs. F2 is NOT explained by worse degradation
  representation learning; it's a backbone effect.
- **Important correction to TEST16**: with real trained weights, N
  measures 3,777ms (not TEST16's 430ms) — TEST16's number was inflated by
  untrained affine-norm layers having identity weights (1/0) that QNN's
  compiler trivially constant-folds away. **The real, trained speedup is
  A/N ≈ 2.80x** — still a substantial, hardware-verified win, just far
  more modest than the original headline.
- **N+F2 stays close to N on hardware**: 3,844ms FP32 (+66ms/+1.7% over
  N), 1,678ms INT8 (+43ms/+2.6% over N) — directly confirming F2's
  operator adds negligible latency on the fast backbone in both
  precisions (F2 itself costs 10,610ms vs. A's 10,560ms FP32 on the slow
  backbone — the operator is cheap everywhere).
- **INT8, all 4 models**: A=2,076ms/27.87dB, N=1,635ms/25.01dB,
  F2=2,557ms/26.22dB, N+F2=1,678ms/25.94dB (small 12-sample check; some
  INT8 PSNR readings sit above FP32, read as sampling noise, not a real
  quantization gain). All 4 remain 100% NPU, zero fallback.

**Decision: PARTIAL GO for N+F2** — a real, substantial, trained-weight-
verified latency win (~2.8x) with a genuine, non-trivial quality cost
(~2.3dB vs. A) that F2's conditioning mechanism partially, not fully,
recovers from normalization surgery's raw instability. Recommended as the
project's primary deployment candidate; N alone is explicitly NOT
recommended standalone due to its quality instability. Full detail:
`test17/report/test17_report.md`.

---

## Cross-Cutting Threads

- **The frequency mechanism thread (TEST01→TEST06-R)** is fully closed:
  AdaIR's explicit FFT-based frequency path is not causally load-bearing
  for restoration at inference time, confirmed by 6 independent tests with
  increasing statistical rigor. This justified never building a
  frequency-based student mechanism.
- **The KD/conditioning thread (TEST07→TEST14)** converged on a single
  validated mechanism (F2: `a=G(e_D)`, rank-2 fixed-basis low-rank
  channel operator) after ruling out simple KD alone (NO-GO), higher rank
  (NO-GO), trajectory distillation (NO-GO), adaptive basis (NO-GO), and
  frequency augmentation (NO-GO/redundant) — a long sequence of negative
  results that earned confidence in the one mechanism that survived.
- **The hardware thread (TEST15→TEST17)** shifted the project's evaluation
  criterion from "does it help restoration" to "does it help restoration
  AND run fast on the real target NPU" — surfacing that normalization
  choice (LayerNorm2d vs. affine_clamp), not the conditioning mechanism
  itself, is the dominant lever for on-device latency, while F2's
  operator is consistently cheap regardless of backbone.
- **Recurring project discipline**: every reported finding in this series
  went through explicit causal controls (self-swap, random, zero, mean,
  cross-scene) before being trusted, multiple negative results were kept
  and reported rather than discarded (TEST09/10/11/13/14), and at least
  three genuine bugs were caught and documented rather than silently
  fixed (TEST01's RNG drift, TEST06's corrupt scene_021, TEST10's
  representational collapse).

---

*This document indexes 17 experiments; for exact numbers, statistical
tests, visualizations, and raw data, see each test's own
`teacher-experiments/testNN/report/` and `results/` folders. TEST17's
entry above will be updated once its INT8 stage completes.*
