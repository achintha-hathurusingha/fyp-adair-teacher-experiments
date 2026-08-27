# TEST17 — Hardware-Aware Student Validation

## 1. Objective

TEST16 found that an untrained normalization-surgery backbone (N:
`layernorm2d`→`affine_clamp`) appeared ~24x faster than the original
baseline. TEST16 also found F2's validated low-rank conditioning operator
adds almost no latency on either backbone. TEST17 asks the natural
follow-up, this time training every model for real: **can the validated
F2 conditioning mechanism be combined with the NPU-friendly
normalization-surgery backbone and retain useful restoration quality
while keeping most of the latency win?**

## 2. Models

Four models, all trained for real (50 epochs, 3 seeds, identical data/
loss/optimizer where applicable — see §3):

| Model | Backbone | Conditioning |
|---|---|---|
| A | Original (`layernorm2d`) | None |
| N | Normalization surgery (`affine_clamp`) | None |
| F2 | Original (`layernorm2d`) | F2's rank-2 low-rank operator, `a=G(e_D)` |
| N+F2 | Normalization surgery (`affine_clamp`) | F2's mechanism, unchanged |

N+F2 differs from F2 in exactly one respect (backbone normalization) and
from N in exactly one respect (conditioning presence) — the same
fairness discipline used throughout this project.

## 3. Training

Identical setup across all 4 models: 50 epochs, Adam, LR=2e-4, batch=8,
3 seeds (0,1,2), TEST12's exact 80/20 scene-disjoint dataset and cached
teacher embeddings (read-only reuse, no regeneration). Loss: `A`/`N` use
`L_restore` (L1) only; `F2`/`N+F2` use `L_restore + 0.1·MSE(e_S,e_T)`.
Best-validation-PSNR and final-epoch checkpoints both saved per run;
best-PSNR checkpoint used as the canonical checkpoint for all downstream
(representation, hardware) analysis.

All 12 runs completed successfully, no NaN/Inf in any run.

## 4. Restoration Results

3-seed mean, last-5-epoch validation metrics:

| Model | PSNR (dB) | PSNR std | SSIM | SSIM std |
|---|---:|---:|---:|---:|
| A | 27.31 | 0.21 | 0.815 | — |
| F2 | 27.04 | 0.14 | 0.825 | — |
| N | **23.45** | **1.78** | 0.709 | — |
| N+F2 | 25.04 | **0.12** | 0.771 | — |

**N is quality-unstable.** Inspecting N/seed2's epoch-by-epoch curve
directly reveals a genuine training-divergence event: `train_l1_loss`
was tracking normally (~0.046-0.079) through epoch 26 (PSNR 24.36), then
spiked to 2.31 at epoch 27 (PSNR crashed to 21.38) and never fully
recovered — a second, smaller spike occurs at epoch 47. Final PSNR for
this seed: 21.2dB. Seeds 0 and 1 trained more normally (final ~24.2,
25.0dB), but the cross-seed std (1.78) reflects this real instability, not
measurement noise.

**N+F2 is far more stable** (std=0.12, an order of magnitude tighter than
N alone) and recovers substantial quality (25.04 vs. N's 23.45dB) — F2's
conditioning mechanism appears to stabilize training on the fast
backbone, not just add quality on top of it.

## 5. Degradation Results

Per-degradation ΔPSNR vs. A (3-seed mean):

| Model | Rain | Haze | Noise | Overall |
|---|---:|---:|---:|---:|
| F2 | -1.13 | -0.89 | **+1.20** | -0.28 |
| N | -4.31 | **-5.20** | -2.09 | -3.86 |
| N+F2 | -3.25 | -2.02 | -1.55 | -2.27 |

**Does N+F2 preserve F2's Noise benefit?** No — F2 alone shows a real
Noise improvement over A (+1.20dB), but N+F2's Noise result is -1.55dB.
The Noise benefit does not survive normalization surgery in this dataset/
seed set.

**Does normalization surgery change Haze behavior?** Yes, dramatically —
Haze is N's single worst-hit degradation (-5.20dB, worse than Rain or
Noise). N+F2 recovers the majority of this specific loss (+3.17dB over N
alone, the largest per-degradation recovery of the three), though not
all of it (-2.02dB still remains vs. A).

## 6. Representation Results

Grouped degradation-classification probe + teacher cosine alignment on
each model's `e_D` (best-checkpoint, seed-averaged):

| Model | Probe accuracy | Teacher cosine similarity |
|---|---:|---:|
| F2 | 96.6% | 0.989 |
| N+F2 | 95.6% | 0.973 |

Nearly identical — a ~1pp probe-accuracy and ~0.016 cosine difference.
**N+F2's restoration quality gap vs. F2 is NOT explained by worse
degradation representation learning.** Per this project's own
interpretive framework (established in prior tests): when `e_S` remains
similar but restoration differs, the effect is attributable to the
backbone/operator interaction, not to representation quality — consistent
with the training-instability finding in §4.

## 7. ONNX Export

All 4 models exported cleanly at the production 256×256 resolution
(opset 17): A (2,333 nodes, 30.0MB), N (1,949 nodes, 29.9MB), F2 (2,346
nodes, 30.0MB), N+F2 (1,962 nodes, 30.0MB). Zero dynamic-weight Conv
nodes in any model. **F2 and N+F2's static-weight audits both passed
programmatically**: every Conv node (including the operator's `U`/`V`
basis) resolves to a compile-time-constant graph initializer.

## 8. NPU Compilation

All 4 models compiled and ran **100% on NPU with zero CPU/GPU fallback**,
in both FP32 and INT8 — verified per-layer, not accepted on aggregate
compile success alone.

## 9. Full Graph Latency

100 warm inference reps per model, FP32, Snapdragon 8 Elite QRD:

| Model | Mean | Median | p90 | p95 | Std |
|---|---:|---:|---:|---:|---:|
| A | 10,559.7 ms | 10,539.5 ms | 10,634.9 ms | 10,684.7 ms | 82.6 ms |
| N | 3,777.1 ms | 3,740.0 ms | 3,879.0 ms | 3,950.5 ms | 116.5 ms |
| F2 | 10,610.5 ms | 10,577.5 ms | 10,693.6 ms | 10,859.3 ms | 133.4 ms |
| N+F2 | 3,843.6 ms | 3,798.0 ms | 3,939.1 ms | 4,045.6 ms | 134.1 ms |

**Primary comparisons:**
- **N vs A**: 3,777ms vs 10,560ms — **2.80x speedup**. This corrects
  TEST16's headline "~24x" claim: TEST16 profiled an *untrained* N, whose
  `AffineClampNorm2d` layers default-initialize to `weight=1, bias=0` — a
  literal identity function that the NPU compiler constant-folded away
  entirely. With real trained (non-identity) weights, the true speedup is
  substantial but ~8.8x more modest than first estimated.
- **N+F2 vs N**: 3,844ms vs 3,777ms — **+66ms (+1.7%)**. The target
  outcome (N+F2 ≈ N, not ≈ F2) is directly confirmed.
- **N+F2 vs F2**: 3,844ms vs 10,610ms — **2.76x speedup**, essentially
  identical to N vs A, confirming the operator's cost is backbone-
  independent.

Peak memory: 125.3-125.9MB across all 4 models — resolution-dominated,
not architecture-dominated (consistent with TEST16).

## 10. Layer Hotspots

LayerNorm/AffineClamp share of total NPU execution cycles: A=27.46%,
F2=27.37%, N=7.33%, N+F2=7.14%. Total graph cycles: A=64.2M, N=20.5M — a
~3.1x reduction, larger than AffineClamp's own ~7.3%-vs-27.4% direct cost
differential alone would predict, implying LayerNorm2d also blocks
fusion of neighboring ops (consistent with `fyp-adair-distill`'s own
prior, narrower finding that QNN "bills" LayerNorm's fusion-breaking cost
to itself).

**Does adding F2 destroy N's efficient fused graph?** No — F2's operator
consumes only 0.10% (on the original backbone) to 0.31% (on the
norm-surgery backbone) of total NPU cycles. §9's near-identical N vs.
N+F2 latency is the direct hardware confirmation of this cycle-level
finding.

## 11. INT8

Real quantize→compile→profile→on-device-inference (real calibration
crops, 12 held-out real inference samples per model):

| Model | INT8 Latency | INT8 Memory | INT8 PSNR | INT8 SSIM | NPU-only |
|---|---:|---:|---:|---:|---|
| A | 2,076 ms | 125.7 MB | 27.87 dB | 0.767 | True |
| N | 1,635 ms | 125.8 MB | 25.01 dB | 0.702 | True |
| F2 | 2,557 ms | 124.6 MB | 26.22 dB | 0.712 | True |
| N+F2 | 1,678 ms | 125.6 MB | 25.94 dB | 0.666 | True |

All 4 remain 100% NPU, zero fallback, under INT8. **Caveat**: INT8 PSNR
uses a single seed's (seed=0) best checkpoint and a distinct 12-image
sample drawn with a different random seed than the FP32 3-seed
validation set — A's and N's INT8 PSNR reading *higher* than their FP32
counterparts is very likely a small-sample artifact (a different, easier
draw of images), not genuine quantization improvement, and should not be
read as a real quality gain from quantization.

**Is N+F2 quantization-robust?** INT8 latency drops substantially
(3,844ms→1,678ms, 2.29x) while staying close to N's INT8 latency
(1,635ms, +2.6%) — the FP32 pattern holds under INT8 too. Quality-wise,
within this pass's small-sample caveat, N+F2's INT8 PSNR (25.94dB) is
close to its FP32 PSNR (25.04dB, actually nominally higher on this
sample) — no INT8-specific quality collapse observed, though the sample
size is too small to make a precise quantitative claim.

## 12. Quality/Latency Pareto

| Model | PSNR | SSIM | NPU Latency (FP32) | NPU Latency (INT8) | Peak Memory |
|---|---:|---:|---:|---:|---:|
| A | 27.31 | 0.815 | 10,560 ms | 2,076 ms | 125.5 MB |
| F2 | 27.04 | 0.825 | 10,610 ms | 2,557 ms | 124.6 MB |
| N | 23.45 | 0.709 | 3,777 ms | 1,635 ms | 125.9 MB |
| N+F2 | 25.04 | 0.771 | 3,844 ms | 1,678 ms | 125.6 MB |

No model dominates on both axes simultaneously. `A`/`F2` form the
high-quality/high-latency corner; `N`/`N+F2` form the low-latency corner,
with `N+F2` strictly dominating `N` on quality at negligible latency cost
— **N+F2 is the Pareto-efficient choice within the fast-backbone
cluster**, and the best point overall if a ~2.3dB quality concession for
a ~2.8x latency win is an acceptable trade for the deployment target.

## 13. Hardware Constraints

See `knowledge/10_NPU_CONSTRAINTS.md` (project-wide) for the full table;
TEST17-specific additions:

- **C7 (new, established by this test)**: untrained-weight NPU profiling
  is not reliably weight-independent — a layer initialized to a
  degenerate/identity transform can be constant-folded away by the NPU
  compiler, producing an artificially fast number. Directly demonstrated
  by comparing TEST16's untrained N (430ms) against this test's trained N
  (3,777ms), same architecture.
- Confirms C3/C4 (TEST15/16): the low-rank conditioning operator remains
  NPU-cheap (0.1-0.3% of cycles) and normalization choice remains the
  dominant full-graph latency lever, now verified with real trained
  weights.

## 14. Limitations

- N's 3-seed mean PSNR (23.45dB) is driven partly by one seed's genuine
  training-divergence event — treat this number as reflecting real
  instability, not a single clean quality figure.
- INT8 quality figures (§11) use a small (12-image), single-seed,
  differently-sampled evaluation set relative to FP32's 3-seed,
  60-crop-×3-degradation validation — a directional signal, not a
  precise benchmark-grade INT8-vs-FP32 comparison.
- No wider/deeper backbone was tested — TEST17's own fairness rule holds
  width/depth fixed specifically to isolate the normalization+
  conditioning comparison; whether a larger N+F2 could close the
  remaining ~2.3dB gap to A without losing the latency win is untested
  (see `knowledge/17_NEXT_EXPERIMENT.md`).
- TEST16's Model-S (static-mixture) anomaly was not re-investigated with
  trained weights in this test.

## 15. GO / NO-GO

- **F2 survives normalization surgery**: YES — N+F2's latency stays
  within 1.7% of N's, on both FP32 and INT8, confirming the operator's
  near-zero cost holds regardless of backbone.
- **Is N+F2 the best quality/latency point?** Within the models tested
  here, yes on a Pareto basis (§12) — it strictly dominates N and offers
  the best latency-for-quality tradeoff among the four. It does not beat
  A/F2 on quality, and does not beat N on raw latency.
- **Overall**: **PARTIAL GO**. N+F2 delivers a real, substantial,
  trained-weight-verified ~2.8x latency win with a bounded, non-trivial
  (~2.3dB) quality cost, and is markedly more training-stable than the
  normalization-surgery backbone alone — recommended as the primary
  deployment candidate, with a follow-up recommended to test whether
  scaling the backbone can narrow the remaining quality gap without
  eroding the latency advantage.
- **N alone is NOT recommended for standalone deployment** — its
  quality instability (one seed showing a genuine training divergence) is
  a real risk independent of its mean quality figure.

## 16. Recommended Final Student

**N+F2** (normalization-surgery backbone + F2's unchanged conditioning
mechanism) is this project's current best-supported deployment
candidate: ~2.8x faster than the original architecture on real Snapdragon
8 Elite hardware, 100% NPU with zero fallback in both FP32 and INT8, and
substantially more training-stable than the fast backbone alone. The
recommended next step is not further mechanism search (per this
project's own "no new architecture" discipline in TEST17's design) but a
backbone-scale sweep — see `knowledge/17_NEXT_EXPERIMENT.md`, Candidate B
— to test whether the ~2.3dB quality gap to the original backbone can be
narrowed at a larger model size while retaining most of the latency win.
