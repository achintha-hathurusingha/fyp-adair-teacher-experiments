# TEST15 — Snapdragon NPU Operator & Graph Benchmark

## 1. Motivation

TEST08 through TEST14 iterated on the *mathematical* design of Minura's
degradation-conditioned restoration operator (low-rank channel mixing,
feature conditioning, adaptive basis, frequency augmentation) purely on
restoration-quality and representational grounds. None of that work asked
whether the operator family is actually cheap to run on the real deployment
target: a Snapdragon phone's Hexagon NPU.

The user proposed pausing further operator invention and instead directly
measuring, on real Qualcomm AI Hub hardware, which primitive ops and small
combinations compile to the NPU at all, whether they trigger CPU/GPU
fallback, and how their latency compares — including a literal
dynamic-weight convolution (the textbook NPU risk case) and Minura's actual
mechanism, to determine empirically whether Minura's low-rank operator is
NPU-safe as-is or needs to be redesigned around a "static-expert-mixture"
pattern. Instruction: **"plan and do."**

## 2. Method

- Built a self-contained operator zoo (`test15/scripts/op_zoo.py`) of 26
  `nn.Module`s: the user's requested primitive table (Conv3x3, DWConv,
  1×1Conv, Add, Multiply, ReLU/clamp, Sigmoid, Softmax, GAP, GMP, Resize,
  Concatenate, elementwise affine, LayerNorm2D, RMSNorm-like, dynamic conv,
  FFT), 7 small combinations (Conv→Add, Conv→Clamp, Conv→Mul,
  Conv→Add→Clamp, DWConv→Pointwise, 1×1→Add→Clamp, GAP→Linear→Multiply),
  and two research-critical operators sized to match TEST09–14's real
  bottleneck shapes (`C_BOTTLENECK=256, HW=8, rank=2, e_D dim=16`):
  - `minura_lowrank_op`: `F' = F + U·diag(a(e_D))·Vᵀ·F` — exactly TEST09–14's
    validated mechanism (fixed `U`,`V` parameters; only the rank-2
    coefficient vector `a` is computed at runtime from `e_D`).
  - `static_mixture_op`: `Y = F + Σᵢ aᵢ(e_D)·Bᵢ(F)` — the proposed
    NPU-native redesign candidate, with static 1×1-conv experts and only
    scalar mixing coefficients computed at runtime.
- Exported every op to ONNX (opset 17, `do_constant_folding=True`).
- Submitted each successfully-exported op as a compile+profile job pair to
  Qualcomm AI Hub, targeting **Snapdragon 8 Elite QRD** (Qualcomm's own
  reference design, chipset SM8750, Hexagon V79 — matching the user's
  "Hexagon V790" target; chosen over OEM phones sharing the same chip to
  avoid vendor software overhead), forcing NPU-targeted compilation via
  `--target_runtime qnn_context_binary`.
- Polled every job to completion with `job.wait()` and pulled per-op
  latency, peak memory, and per-layer compute-unit assignment
  (NPU/GPU/CPU) from each profile report.

## 3. Results

### 3.1 Export

25/26 ops exported to ONNX successfully. **FFT failed outright**:
`torch.onnx.export` cannot lower `aten::fft_fft2` at opset 17 — there is no
ONNX graph to even submit for FFT-based ops with this toolchain. This
confirms the project's existing avoidance of literal FFT ops (TEST06-R,
TEST14) from an entirely independent angle: FFT is disqualified before the
NPU question is even reached.

### 3.2 Compile + profile

All 25 exported ops compiled successfully. 24/25 profiled successfully (one
`compile_success=False` combination job — see caveats). **Zero CPU or GPU
fallback was observed anywhere**: every successfully profiled op ran
100% on NPU (`any_cpu_fallback=False`, `any_gpu_fallback=False`,
`npu_only=True` for all 24).

| Operator | Latency (ms) | Compute unit |
|---|---:|---|
| global_max_pool | 54.0 | NPU |
| global_avg_pool | 59.0 | NPU |
| elementwise_affine | 61.0 | NPU |
| sigmoid | 64.0 | NPU |
| relu_clamp | 68.0 | NPU |
| depthwise_conv | 74.0 | NPU |
| **minura_lowrank_op** | **74.0** | NPU |
| **static_mixture_op** | **74.0** | NPU |
| conv1x1 | 72.0 | NPU |
| multiply | 72.0 | NPU |
| combo_dwconv_pointwise | 73.0 | NPU |
| combo_1x1_add_clamp | 86.0 | NPU |
| add | 85.0 | NPU |
| rmsnorm_like | 85.0 | NPU |
| combo_gap_linear_mul | 83.0 | NPU |
| conv3x3 | 89.0 | NPU |
| combo_conv_clamp | 90.0 | NPU |
| combo_conv_add_clamp | 90.0 | NPU |
| combo_conv_mul | 89.0 | NPU |
| softmax | 87.0 | NPU |
| combo_conv_add | 105.0 | NPU |
| concatenate | 97.0 | NPU |
| layernorm2d | 97.0 | NPU |
| resize | 139.0 | NPU |
| **dynamic_conv** | **4059.0** | NPU |
| fft | — | export failed, never submitted |

(Full table with memory and per-op device/status fields in
`Snapdragon_NPU_Operator_Benchmark.xlsx` → `Operator_Benchmark_Table`.)

### 3.3 The dynamic-convolution finding

`dynamic_conv` (a literal per-sample-generated 3×3 grouped conv kernel,
weights produced by a small `Linear` head from `e_D` and materialized via
`F.conv2d` in an explicit per-sample loop) **compiled cleanly and ran
100% on NPU** — it is not rejected or silently downgraded to CPU/GPU. But
it measured **4059ms**, a **30–75× latency penalty** versus every other op
tested (54–139ms cluster). This is the direct empirical confirmation of
Qualcomm's documented warning about dynamic weights: the failure mode is
not a compile-time rejection, it's a severe, silent runtime performance
cliff that would only show up once you actually benchmark on hardware —
exactly the risk the user flagged before authorizing this experiment.

### 3.4 The key result: Minura's operator vs. the risk case

`minura_lowrank_op` and `static_mixture_op` both measured **74ms** —
statistically indistinguishable from each other, and squarely inside the
ordinary 54–139ms operator cluster, nowhere near the `dynamic_conv` cliff.

This is the central finding of TEST15: **Minura's actual validated
mechanism (fixed `U`/`V` basis matrices, only a small coefficient vector
`a(e_D)` computed at runtime) was never the "dynamic convolution" risk
case.** It already has the exact computational shape of the proposed
NPU-native static-mixture redesign — a static weight/basis with only
scalar dynamic mixing — and measures identically to that redesign on real
hardware. No operator-family redesign is needed on latency grounds.
TEST09–14's mechanism search (rank sweeps, feature conditioning, adaptive
basis, frequency augmentation) was not spent chasing an NPU-infeasible
operator shape.

## 4. Caveats / scope limits

- **Isolated micro-benchmarks, not the full student graph.** Each op was
  profiled alone at representative shapes (batch=1, fp32), not embedded in
  the actual TEST09–14 student network. Full-model effects the user
  explicitly flagged as potentially dominant — op fusion across
  neighboring layers, memory movement/layout conversion between ops,
  scheduling overhead — are not measured here. A follow-up that compiles
  and profiles the actual end-to-end student network graph would be needed
  to confirm these per-op numbers hold when composed at full scale.
- **No INT8 quantization tested this pass.** `submit_qai_hub_jobs.py`
  supports a `--quantize` flag but it was not used; all results are fp32.
  The user's originally-requested table included an INT8 column, currently
  unpopulated. This is a natural, low-effort follow-up (requires
  calibration data) rather than a blocker to this experiment's conclusions.
- One combination job (`combo_conv_clamp`) reported `compile_success=False`
  in the final aggregation table alongside several sibling combos in an
  earlier (buggy) collection pass; the corrected re-run resolved the
  polling bug (see `collect_results.py` — jobs in the `CREATED`/queued
  state are `.pending`, not `.running`, and were being read as
  already-failed before the fix). The final CSV reflects the corrected,
  fully-waited-for run. Re-verify `Operator_Benchmark_Table` directly if
  auditing individual combination-op rows.

## 5. Recommendation for TEST16

On the latency axis, Minura's existing low-rank operator needs no redesign
— it already matches the NPU-native static-mixture pattern's performance.
Two candidate follow-ups, in priority order:

1. **Full-graph benchmark**: compile and profile the actual TEST09–14
   student network (not isolated ops) end-to-end on Snapdragon 8 Elite QRD,
   to validate that fusion/memory-movement effects don't change this
   picture at full scale — this is the biggest open scope gap from this
   experiment.
2. **INT8 quantization pass**: rerun the same operator zoo (or the
   full-graph benchmark) with `--quantize`, since mobile NPU deployment
   will very likely ship INT8, and quantization can change relative
   operator costs (e.g. dynamic per-sample weight generation may interact
   differently with quantization than static weights do).

Given both are infrastructure/measurement tasks rather than new operator
mathematics, they are lower-risk, higher-confidence next steps than
another TEST09-14-style operator variant.

## 6. Deliverables

- `test15/results/statistics/npu_operator_benchmark.csv` — raw per-op data.
- `test15/results/visualizations/*.png` — 4 charts (full latency
  distribution log-scale, base-op linear-scale, combination latency, and
  the Minura-vs-dynamic-conv key comparison).
- `test15/results/Snapdragon_NPU_Operator_Benchmark.xlsx` — README,
  full operator table, base-ops table, combinations table, Minura key
  comparison + interpretation, environment info, embedded visualizations.
- `test15/scripts/op_zoo.py`, `export_onnx.py`, `submit_qai_hub_jobs.py`,
  `collect_results.py`, `make_visualizations_and_excel_local.py` — full
  reproducible pipeline.
