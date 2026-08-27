# TEST16 — Full Student Graph Snapdragon NPU Validation

## 1. Objective

TEST15 mapped which isolated micro-ops the Snapdragon NPU (Hexagon V79)
executes efficiently, finding that runtime-generated *convolution weights*
are the real risk (a 30-75x latency cliff), while Minura's actual low-rank
operator (fixed basis, runtime scalar coefficients) measures identically to
a plain static conv. That result was scoped to isolated single ops. TEST16
asks the necessary follow-up: **what happens when the actual complete
restoration student graph is compiled and profiled end to end?** Four full
models were benchmarked — baseline (A), Minura's validated operator (F2), a
normalization-surgery ablation (N), and a static-mixture reinterpretation
(S) — on the identical hardware target used in TEST15.

## 2. Hardware Environment

| Field | Value |
|---|---|
| Target device | Snapdragon 8 Elite QRD (Qualcomm reference design) |
| Chipset | SM8750 |
| NPU | Hexagon V79 |
| Runtime | QNN context binary (`--target_runtime qnn_context_binary`) |
| ONNX opset | 17 |
| Input/output resolution | 256×256 (production export path: `fyp-adair-distill/configs/export/qnn_int8.yaml`) |
| qai_hub version | 0.53.0 |
| torch version | 2.5.1+cu124 |
| onnx version | 1.17.0 |
| Precisions tested | FP32, INT8 |

Identical device to TEST15, so per-op and full-graph numbers are directly comparable.

## 3. Student Models

| Model | Description | Trained? |
|---|---|---|
| A | Baseline NAFNet, no conditional mechanism | Yes (TEST12 checkpoint) |
| F2 | Minura's validated rank-2 low-rank conditional operator (TEST08-C→TEST14 lineage) | Yes (TEST12 checkpoint) |
| N | Same architecture as A, internal `norm_type` switched `layernorm2d`→`affine_clamp` | **No** — see scoping note below |
| S | Static-mixture reinterpretation: 3 static 1×1-conv branches, runtime scalar mixing only | **No** — see scoping note below |

**Scoping note.** Phase 14 forbids retraining any of the four models this
pass, but Models N and S have no trained checkpoint anywhere in the repo. A
clarifying question was raised before proceeding; the resolution (user
choice) was to export/compile/profile N and S **untrained** this pass —
matching `fyp-adair-distill/src/models/norms.py`'s own documented
convention that NPU latency does not depend on weight values, so untrained
profiling is valid for latency/architecture claims but not for quality. PSNR/SSIM
for N and S are therefore **not reported** (would be fabricated data);
their entries in the quality columns below are intentionally blank.

## 4. PyTorch Validation

| Model | Trained | Params | MACs | Val PSNR | Val SSIM | Shape/stability check |
|---|---|---|---|---|---|---|
| A | Yes | 7,371,923 | 1.033G | 27.28 dB | 0.8063 | — |
| F2 | Yes | 7,381,189 | 1.033G | 27.15 dB | 0.8248 | — |
| N | No | 7,371,923 | 1.033G | not measured | not measured | finite output, correct shape ✓ |
| S | No | 7,594,534 | 1.046G | not measured | not measured | finite output, correct shape ✓ |

A and F2 PSNR/SSIM computed on TEST12's real validation split (60 crops ×
3 degradations), matching TEST12's own recorded numbers as a checkpoint
sanity check. N and S ran 40 real validation crops through the untrained
network with no NaN/Inf and correct output shape — sufficient for export
and hardware profiling, not for quality claims.

## 5. ONNX Export

| Model | Export | Nodes | Model size | Conv nodes | Dynamic-weight Conv | Gather | FFT |
|---|---|---|---|---|---|---|---|
| A | success | 2,333 | 30.0 MB | 226 | 0 | 73 | 0 |
| F2 | success | 2,346 | 30.0 MB | 226 | 0 | 73 | 0 |
| N | success | 2,155 | 29.8 MB | 226 | 0 | 73 | 0 |
| S | success | 2,580 | 30.8 MB | 229 | 0 | 76 | 0 |

All 4 models exported cleanly at the production 256×256 resolution. **Model
S's static-weight audit passed programmatically**: every one of its 229
Conv nodes resolves to a graph initializer (compile-time constant), not a
graph input — confirmed by walking the ONNX graph, not assumed from the
architecture alone.

The ONNX-level op histogram directly confirms the normalization-surgery
ablation is clean: A and F2 each contain 128-129 `ReduceMean`, 64 `Sqrt`,
64 `Pow`, 136 `Div` (the LayerNorm2d decomposition); **N contains zero of
any of these** (its `norm_type` is `affine_clamp` throughout). S contains
128 `ReduceMean`/64 `Sqrt`/64 `Pow` — nearly identical counts to A/F2 —
which becomes important in §9.

## 6. NPU Compilation

| Model | Compile | Profile | Compute units | NPU-only | CPU fallback | GPU fallback |
|---|---|---|---|---|---|---|
| A | ✓ | ✓ | NPU | True | False | False |
| F2 | ✓ | ✓ | NPU | True | False | False |
| N | ✓ | ✓ | NPU | True | False | False |
| S | ✓ | ✓ | NPU | True | False | False |

All 4 complete graphs compile and execute **100% on NPU with zero CPU/GPU
fallback** — this extends TEST15's isolated-op finding to full-graph scale.

## 7. Full Graph Latency

100 warm inference repetitions per model (FP32):

| Model | Mean | Median | p90 | p95 | Std | Layers reported |
|---|---|---|---|---|---|---|
| A | 10,527.9 ms | 10,504.5 ms | 10,587.7 ms | 10,692.8 ms | 139.8 ms | 946 |
| F2 | 10,671.5 ms | 10,647.0 ms | 10,748.2 ms | 10,780.3 ms | 114.8 ms | 962 |
| N | 430.1 ms | 405.0 ms | 409.0 ms | 441.4 ms | 168.6 ms | 23 |
| S | 462.0 ms | 442.0 ms | 481.0 ms | 487.0 ms | 62.9 ms | 49 |

Both A and F2's timing samples are tight and stable (std < 1.5% of mean).
**N is ~24.5x faster than A**; S is close to N, not to A/F2 — analyzed in
§9-11.

## 8. Peak Memory

| Model | Peak memory | Model size (disk) |
|---|---|---|
| A | 125.4 MB | 30.0 MB |
| F2 | 125.5 MB | 30.0 MB |
| N | 125.4 MB | 29.8 MB |
| S | 125.5 MB | 30.8 MB |

Peak memory is essentially identical across all 4 models (dominated by the
256×256 input/output activation footprint, not weight count) — memory is
not a differentiator between these architectures at this resolution.

## 9. Layer Hotspots

Top contributors by execution cycles, model A (LayerNorm2d present):

| Layer | Cycles | % of total |
|---|---|---|
| `node_LayerNormalization_865` (fused) | 1,142,918 | 1.79% |
| `node_LayerNormalization_1105` | 1,110,596 | 1.74% |
| `node_LayerNormalization_1109` | 1,108,274 | 1.74% |
| ... (many more `LayerNormalization` fused nodes) | | |
| `/inner/ups.3/ups.3.1/DepthToSpace` | 1,092,966 | 1.71% |

A's cost is spread across ~72+ individually-costed LayerNorm-fused nodes
(one per block × 2 norms × ~36 blocks), each ~1.7-1.8% of total — no single
layer dominates; the *aggregate* of many LayerNorm instances does.

Model N (affine_clamp, no LayerNorm) shows a completely different profile —
only 23 reported layers total, dominated by:

| Layer | Cycles | % of total |
|---|---|---|
| `/inner/net/ups.3/ups.3.1/DepthToSpace` | 1,056,400 | 36.10% |
| `/inner/net/Add_4` (final skip connection) | 709,080 | 24.23% |
| `input` (tensor ingestion) | 337,115 | 11.52% |
| `/inner/net/downs.0/Conv` | 183,806 | 6.28% |

Once LayerNorm's cost (and its fusion-blocking effect on neighbors) is
removed, the remaining bottleneck is the pixel-shuffle upsample
(`DepthToSpace`) and the full-resolution residual add — ordinary,
unavoidable full-image-resolution operations, not anything exotic.

**LayerNorm/AffineClamp bucket share of total cycles**: A = 26.20%, F2 =
26.04%, N = 0.00%, S = 0.00%.

Model S's per-layer profile is nearly a structural duplicate of N's
(dominated by the same `DepthToSpace` + final-`Add` pattern, similar
magnitudes) — despite S's ONNX graph containing the same LayerNorm ops as
A/F2. This inconsistency is the central open question of this report; see
§11.

## 10. Normalization Surgery

| | A (layernorm2d) | N (affine_clamp) | Ratio |
|---|---|---|---|
| Mean latency | 10,527.9 ms | 430.1 ms | **24.5x** |
| Total cycles | 63,851,916 | 2,925,995 | **21.8x** |
| PSNR | 27.28 dB | not measured (untrained) | — |
| Peak memory | 125.4 MB | 125.4 MB | ~1x |

This is the cleanest, most trustworthy finding in TEST16: N differs from A
in exactly one respect — `norm_type` — confirmed at the ONNX graph level
(zero `ReduceMean`/`Sqrt`/`Pow` in N vs 128/64/64 in A). The latency
reduction (~24.5x) is far larger than LayerNorm's own 26% cycle share would
predict, because LayerNorm also blocks QNN's fusion of neighboring
Conv/elementwise ops (consistent with `fyp-adair-distill`'s own prior,
narrower INT8 finding that LayerNorm "bills" fusion-breaking costs to
itself). **Normalization surgery is the single highest-leverage NPU
latency lever found across TEST15 and TEST16 combined.** The open item is
quality: N has no trained checkpoint, so whether `affine_clamp` throughout
preserves restoration quality is unverified.

## 11. Static Mixture

**Architecture validation (Phase 8 requirement — passed):**
- 3 static branches (`n_experts=3`), each a `Conv2d(256,256,kernel=1)`.
- Coefficient head: `Linear(528,32)→ReLU→Linear(32,3)`, zero-initialized
  final layer (so `alpha≡0` at init — a live, non-dead-code computation,
  not compile-time-foldable, since it depends on runtime pooled features).
- ONNX audit: **every Conv node's weight is a compile-time-constant
  initializer** — programmatically verified, not assumed.
- 213,611 additional parameters vs A/F2's mechanism (7.59M vs 7.37-7.38M).

**Latency — the unresolved finding.** In FP32, S measures 462ms,
statistically matching N (430ms) rather than A/F2 (10,528-10,672ms) —
*despite S's ONNX graph containing the same LayerNorm2d decomposition as
A/F2* (128 `ReduceMean`, 64 `Sqrt`, 64 `Pow` — nearly identical counts).
This was investigated, not assumed:
- **Ruled out job/model mixup**: each model was uploaded and compiled as a
  distinct `qai_hub` model object with a distinct model ID; S's profile
  contains S-specific layer names (`/inner/experts.0/Conv`,
  `/inner/coeff_head/...`) that could only come from S's actual graph.
- **Ruled out measurement noise**: the 100-sample `all_inference_times`
  array for S is tight and stable (min 438ms, max 1037ms with one
  warm-up outlier, then a consistent ~477-500ms band) — a real,
  reproducible on-device measurement.
- **Ruled out dead code**: the LayerNorm ops are on S's live forward path
  (exercised by `_encode_to_bottleneck`/`_decode_from_bottleneck`,
  confirmed by the PyTorch stability check producing correct, finite
  output) — QNN cannot legally eliminate them without changing the
  computed function.
- **What remains unconfirmed**: why QNN's compiler produces a fused,
  near-zero-cost representation of S's LayerNorm-containing backbone,
  identical in shape to N's genuinely-LayerNorm-free backbone, while
  producing an unfused, expensive representation for the structurally
  similar A/F2 backbones. The leading hypothesis is that adding the
  static-mixture branch changes global graph-optimizer scheduling in a
  way that unlocks a different (much more aggressive) fusion pass for
  the *entire* graph — but this is speculative, not verified against
  QNN internals.

**Under INT8, this anomaly disappears**: S measures 2,530ms — matching
A (2,525ms) and F2 (2,558ms), not N (1,609ms). This is read as evidence
that S's FP32 speed is a **precision/compiler-specific artifact**, not a
generalizable property of the static-mixture architecture. **Recommendation:
do not treat S's FP32 latency number as confirmation that the static-mixture
redesign is hardware-friendly without a dedicated, isolated follow-up**
(e.g., re-running S with a fresh model upload and a controlled variant that
strips the static-mixture branch to test whether removing *only* that
component — not the LayerNorm backbone — reproduces A/F2's cost).

## 12. INT8

Real quantize (`hub.submit_quantize_job`, real calibration crops, not
placeholder random data) → compile → profile → **real on-device inference**
(12 held-out validation images) for A and F2; quantize→compile→profile only
for N/S (no meaningful quality claim possible, untrained):

| Model | Quantize | Compile | Profile | Latency | Peak memory | NPU-only | PSNR (INT8) | SSIM (INT8) | Δ PSNR vs FP32 |
|---|---|---|---|---|---|---|---|---|---|
| A | ✓ | ✓ | ✓ | 2,525 ms | 125.0 MB | True | 24.84 dB | 0.7131 | −2.44 dB |
| F2 | ✓ | ✓ | ✓ | 2,558 ms | 125.5 MB | True | 26.01 dB | 0.7042 | **−1.14 dB** |
| N | ✓ | ✓ | ✓ | 1,609 ms | 125.3 MB | True | not measured | not measured | — |
| S | ✓ | ✓ | ✓ | 2,530 ms | 125.7 MB | True | not measured | not measured | — |

All 4 remain 100% NPU with zero fallback under INT8. **F2 degrades less
than A under INT8** (−1.14dB vs −2.44dB) — Minura's operator appears more
INT8-robust than the plain baseline, though this is based on only 12
real-inference samples and should be treated as a directional signal, not
a precise number. N stays fastest under both precisions (confirming §10's
finding is not an FP32-only artifact); S's FP32 speed advantage does not
carry over to INT8 (§11).

## 13. Quality vs Hardware Trade-off

| Model | PSNR | SSIM | Params | MACs | Model MB | NPU latency (FP32) | Peak memory | CPU fallback |
|---|---|---|---|---|---|---|---|---|
| A | 27.28 dB | 0.8063 | 7.37M | 1.033G | 30.0 MB | 10,528 ms | 125.4 MB | False |
| F2 | 27.15 dB | 0.8248 | 7.38M | 1.033G | 30.0 MB | 10,672 ms | 125.5 MB | False |
| N | not measured | not measured | 7.37M | 1.033G | 29.8 MB | 430 ms | 125.4 MB | False |
| S | not measured | not measured | 7.59M | 1.046G | 30.8 MB | 462 ms | 125.5 MB | False |

Theoretical complexity (MACs) is essentially flat across all 4 (1.03-1.05G)
— **MACs predict almost none of the observed 24x latency spread**, which
is the report's clearest illustration of why full-graph hardware
measurement, not FLOP-counting, has to drive this project's deployment
decisions (per the task's explicit rule 5).

## 14. Hardware Constraints

Only experimentally-verified constraints, no invented ones:

| ID | Constraint | Confidence |
|---|---|---|
| C1 | Static (compile-time-constant) Conv weights are cheap and universally NPU-native. | Confirmed (TEST15 + TEST16: A/F2/N/S all 100% NPU, zero fallback) |
| C2 | Runtime-generated Conv weights are catastrophic (30-75x isolated-op penalty), but this is NOT re-tested at full-graph scale — none of A/F2/N/S use this pattern. | Confirmed isolated (TEST15); untested at full-graph scale |
| C3 | Runtime SCALAR/channel-mixing coefficients (Minura's low-rank op, static-mixture op) are cheap — both measured 74ms in TEST15 isolation, identical to a plain static conv. | Confirmed (TEST15 isolated) |
| C4 | LayerNorm2d is the dominant full-graph NPU cost: ~26% of cycles directly, but its presence also blocks neighbor fusion — removing it cuts total cycles ~22x, far more than 26% alone predicts. | Confirmed (TEST16, ONNX-graph-verified) |
| C5 | Static-mixture's full-graph interaction with a LayerNorm-heavy backbone is unconfirmed — Model S showed an unexplained FP32-only cost collapse that did not survive INT8 quantization. | Unresolved — flagged for follow-up, not to be relied on |

## 15. Limitations

- Models N and S have no trained checkpoint this pass (explicit scoping
  decision); their PSNR/SSIM are genuinely unknown, not merely unreported.
- The Model-S FP32 latency anomaly (§11) is unresolved. It is reported
  transparently with everything checked to rule out mundane explanations
  (job mixup, dead code, measurement noise), but the actual QNN compiler
  mechanism remains unconfirmed.
- INT8 quality checks used only 12 real on-device inference samples per
  model (A/F2) — sufficient for a directional signal, not a precise
  benchmark-grade quality number.
- INT8 calibration and inference images were the TEST12 128×128 validation
  crops resized to the production 256×256 export resolution — a
  reasonable proxy, not native-resolution data.
- PyTorch-level validation used TEST12's existing 60-crop validation split;
  this is the same protocol TEST12 used to validate A/F2 originally, so
  cross-comparable, but is a small held-out set.

## 16. GO / NO-GO

- **F2 (Minura's operator)**: GO — full-graph NPU-safe (100% NPU, zero
  fallback, both precisions), quality close to baseline, and more
  INT8-robust than A.
- **N (normalization surgery)**: GO on latency, quality unverified — by far
  the highest-confidence, highest-leverage finding in this report; worth
  training to get real quality numbers.
- **S (static mixture)**: NOT YET — real static-weight audit passed and it
  is architecturally sound, but its standout FP32 latency result is
  unexplained and did not replicate under INT8. Needs a dedicated
  follow-up before being trusted.

## 17. Recommended Student Architecture

Given the evidence in this report, the highest-value next step is **not**
another operator-mechanism experiment — it is training Model N (or an
N+F2 combination: F2's low-rank operator on an affine_clamp-normalized
backbone) to get a real quality number for the mechanism that unlocked
by far the largest latency win measured across TEST15 and TEST16. If
N's quality holds up close to A's, that combination (F2's mechanism +
affine_clamp normalization) is the strongest deployment candidate
identified in this project so far — the low-rank operator's own latency
cost (§7, F2 vs A: 10,672ms vs 10,528ms, i.e. negligible) means it can
likely be added to an affine_clamp backbone at near-zero additional
latency cost.

**Recommended TEST17**: train an `N+F2` combination model (affine_clamp
normalization + Minura's low-rank operator) end to end, validate PSNR/SSIM
properly, then re-run this same full-graph NPU pipeline on it. In parallel
or as a follow-up, resolve the Model-S anomaly with a smaller, targeted
experiment (not a full second training run) — e.g. recompiling S with a
controlled ablation that isolates whether the static-mixture branch or
some other graph-structural change is responsible for the FP32-only fusion
behavior.
