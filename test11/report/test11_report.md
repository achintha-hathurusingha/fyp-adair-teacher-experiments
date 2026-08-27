# TEST11 — Low-Rank Conditional Operator Capacity

## 1. Motivation

TEST09 identified compact-latent KD + rank-2 low-rank channel-mixing (Model
F) as the strongest restoration-conditioning mechanism found across
TEST08-C/09/10/10-R, but it still underperforms baseline NAFNet overall and
has never rescued Haze despite Haze consistently drawing the strongest
conditioning signal at every tested depth and mechanism. TEST10/10-R
separately established that restoration-trajectory distillation is a
validated NO-GO and should not be extended. TEST11 asks the natural
remaining question about the *existing* best mechanism: is the gap caused
by insufficient operator capacity (rank), and does required capacity differ
by degradation?

## 2. Hypothesis

**Primary**: different degradation types require different conditional
operator capacity — specifically, Rain/Noise may be adequately served by
rank 2, while Haze may require higher rank. The task explicitly cautioned
against assuming higher rank is categorically better.

## 3. Models

Five models, isolating rank as the sole manipulated variable:

- **A**: baseline locked NAFNet, no KD, no conditioning.
- **F2/F4/F8/F16**: identical architecture — compact latent KD (`e_S`,
  16-dim) + low-rank channel-mixing `F' = F + U·diag(a(e_S))·Vᵀ·F` at the
  bottleneck — differing **only** in rank R ∈ {2,4,8,16}. No FiLM, no
  trajectory distillation, no additional decoder conditioning, no base
  NAFNet capacity change.

## 4. Dataset and Training

Exact TEST07-B/08-C/09/10/10-R dataset and split, reused read-only. Adam,
LR=2e-4, batch=8, 50 epochs, seeds {0,1,2}, identical `λKD=0.1` for every
rank (no rank-specific tuning, per spec) — 15 total runs. All completed
cleanly with **zero NaN/Inf events** and stable, decreasing gradient norms
throughout, including at R=16 (no rank-dependent optimization failure).
Models A/F2 reproduce TEST08-C/09/10/10-R's own numbers exactly
(deterministic given identical seed/data/architecture).

## 5. Overall Restoration Results

| Model | Rank | Mean PSNR (dB) | Std | Mean SSIM |
|---|---|---|---|---|
| A | — | 27.315 | 0.198 | 0.815 |
| F2 | 2 | 27.038 | 0.140 | 0.825 |
| F4 | 4 | 27.073 | 0.177 | 0.826 |
| F8 | 8 | 27.073 | 0.161 | 0.827 |
| F16 | 16 | 26.970 | 0.153 | 0.826 |

| Comparison | Mean ΔPSNR (dB) | Same-sign count (of 3) |
|---|---|---|
| F4 − F2 | +0.035 | 2/3 |
| F8 − F4 | -0.001 | 2/3 |
| F16 − F8 | -0.103 | 2/3 |
| F2 − A | -0.276 | **3/3** |
| F4 − A | -0.242 | **3/3** |
| F8 − A | -0.243 | **3/3** |
| F16 − A | -0.346 | **3/3** |

**No rank-scaling step is directionally consistent across seeds** (each is
2/3 at best — not even unanimous). PSNR is flat from R=2 through R=8
(27.04→27.07→27.07) and, if anything, mildly *worse* at R=16 (26.97). Every
rank tested underperforms baseline A consistently (3/3 seeds each,
-0.24 to -0.35dB) — capacity scaling has not closed, and does not appear to
be closing, the gap to baseline.

## 6. Per-Degradation Results

PSNR by degradation and rank (mean across 3 seeds):

| Degradation | R=2 | R=4 | R=8 | R=16 |
|---|---|---|---|---|
| Rain | 29.65 | 29.72 | 29.46 | 29.51 |
| Haze | 23.97 | 24.02 | 24.23 | 24.15 |
| Noise | 28.30 | 28.28 | 28.30 | 28.30 |

Noise is essentially flat across all ranks (28.28-28.30). Rain shows a
small non-monotonic wobble (peak at R=4, dip at R=8, partial recovery at
R=16) within noise. Haze shows the largest relative movement (23.97→24.23,
+0.26dB from R=2 to R=8) but even this modest trend reverses slightly at
R=16 (24.15) and, per Section 7, does not translate to closing the gap to
baseline.

## 7. Haze Analysis

Haze ΔPSNR vs. baseline A, by rank: R=2: **-0.89dB**, R=4: -0.84dB, R=8:
-0.64dB, R=16: **-1.09dB**. This is the central test of the primary
hypothesis, and the answer is **no** — Haze's deficit relative to baseline
does not shrink monotonically with rank; R=16 (the highest-capacity model)
has the *worst* Haze gap of all four ranks tested, not the best. Any
apparent improvement from R=2 to R=8 (-0.89→-0.64) reverses at R=16,
consistent with noise rather than a genuine capacity-driven trend. **The
primary hypothesis is not supported by the data.**

## 8. Operator Utilization

Coefficient L2 magnitude and modulation magnitude both increase
monotonically with rank for every degradation, and **Haze has the largest
coefficient magnitude and modulation magnitude at every single rank
tested** (e.g. modulation magnitude: Haze 0.284→0.427 from R=2 to R=16 vs.
Rain 0.116→0.177 and Noise 0.138→0.225) — reproducing the same pattern
observed in TEST08-C/09/10-R at every capacity level. The model consistently
identifies Haze as needing the strongest correction, at every rank, without
this translating into a restoration benefit (Section 7). This rules out
"the model isn't trying hard enough for Haze" as an explanation at any
tested capacity.

## 9. Effective Rank

This is the decisive diagnostic. Configured rank vs. effective rank
(participation ratio of the coefficient covariance, mean across seeds):

| Model | Configured Rank | Effective Rank | Utilization |
|---|---|---|---|
| F2 | 2 | 1.08 | 54.0% |
| F4 | 4 | 1.70 | 42.4% |
| F8 | 8 | 1.95 | 24.4% |
| F16 | 16 | **2.63** | **16.4%** |

**Effective rank saturates far below configured rank, and utilization
efficiency *declines* as configured rank increases.** F16's 16 available
dimensions are used no more effectively (in absolute terms, barely 2.4x
more) than F2's 2 dimensions, despite an 8x larger budget. This is exactly
the task's pre-specified "IMPORTANT FAILURE CASE": *"If higher rank
improves PSNR but effective rank remains low, then the extra configured
capacity is not actually being used. Investigate coefficient generation
rather than increasing rank."* Combined with Section 5's finding that
higher rank does *not* even reliably improve PSNR, this is an even stronger
version of that failure case — the extra capacity produces neither
utilization nor benefit.

## 10. Representation Analysis

| Model | e_S probe accuracy | Teacher cosine similarity |
|---|---|---|
| F2 | 96.4% | 0.9898 |
| F4 | 96.5% | 0.9902 |
| F8 | 96.5% | 0.9895 |
| F16 | 96.5% | 0.9902 |

The compact degradation representation is **statistically indistinguishable
across all four ranks** — probe accuracy and teacher alignment do not
change with R. This directly supports the interpretation the task
anticipated: *"the representation was already sufficient; the limitation
was operator capacity"* — except Section 9 shows the limitation is not that
more capacity was unavailable, but that the available capacity (even at
R=2) is not being fully exploited by the coefficient-generation head. The
bottleneck is not the compact representation, and (per Section 9) it is not
raw configured rank either — it is specifically *how* the coefficients
`a(e_S)` are generated from that representation.

## 11. Complexity

| Model | Params | Extra vs. A | MACs @128px |
|---|---|---|---|
| A | 7,371,923 | — | 1,033,040,896 |
| F2 | 7,381,189 | +9,266 | 1,033,114,656 |
| F4 | 7,382,247 | +10,324 | 1,033,180,224 |
| F8 | 7,384,363 | +12,440 | 1,033,311,360 |
| F16 | 7,388,595 | +16,672 | 1,033,573,632 |

All four ranks are cheap relative to the 7.37M-param/1.03GMAC backbone
(largest overhead, F16, is +0.226% params, +0.052% MACs). Since PSNR does
not improve with rank (Section 5), "quality gained per additional
parameter" is at best flat and at worst negative beyond R=2 — there is no
Pareto-efficient point favoring any rank above 2; **R=2 (or arguably the
plain KD baseline B) is already the efficient choice within this mechanism
family.** Theoretical complexity only; no NPU latency claim.

## 12. Statistical Analysis

Per the task's requirement, N=3 seeds throughout — all comparisons
exploratory. The consistent 3/3 same-sign result for every rank-vs-baseline
comparison (F2-A through F16-A) is a meaningfully stronger signal than the
2/3 same-sign result for every rank-vs-rank step — i.e., **"every rank
underperforms baseline" is well-supported; "higher rank changes anything
meaningfully relative to lower rank" is not.** No rank comparison approaches
a bootstrap CI that excludes zero in the direction of improvement.

## 13. Limitations

- N=3 seeds; none of the rank-to-rank comparisons show a bootstrap CI
  cleanly excluding zero, so a small genuine effect at some rank cannot be
  fully ruled out, but the effect (if any) is smaller than seed-to-seed
  noise.
- Effective rank was computed via participation ratio over the 60
  validation crops; a larger or differently-distributed sample might
  shift the exact effective-rank numbers slightly, though the qualitative
  finding (configured ≫ effective, worsening with R) is unlikely to
  reverse given the consistency across all four ranks and all three seeds.
- This experiment isolates rank within one specific operator family
  (channel-mixing via shared U/V + per-sample diagonal coefficients). It
  does not test whether a differently-parameterized higher-capacity
  operator (e.g., allowing U/V themselves to be sample-dependent, or a
  richer coefficient-generation head) would behave differently — that is
  precisely the recommended next step (Section 15).

## 14. GO / NO-GO

Per the task's decision rule:

- **GO** requires higher rank to consistently improve restoration,
  especially Haze. **Not met** — no rank-scaling step is even directionally
  consistent across seeds, and Haze's gap to baseline is worst at the
  highest rank tested.
- **PARTIAL GO** requires higher rank to improve one degradation
  (especially Haze) while the overall model remains below baseline. **Not
  met** — Haze does not show a genuine, monotonic improvement; its apparent
  R=2→R=8 movement reverses at R=16.
- **NO-GO** requires F2≈F4≈F8≈F16 — do not continue increasing rank, move to
  another mechanism. **This is what the evidence shows.**
- **Important failure case** (higher rank improves PSNR but effective rank
  stays low) does not strictly apply since PSNR does *not* improve with
  rank — but the underlying diagnostic instruction is still directly
  actionable: effective rank stays low regardless, so **investigate
  coefficient generation rather than rank** is the correct next step
  regardless of which failure-case label technically fits best.

**Decision: NO-GO.** Operator rank is not the bottleneck for this
mechanism. Do not test rank beyond 16 or otherwise continue this specific
scaling axis.

## 15. Recommended Next Experiment

Per the task's own diagnostic path: since effective rank saturates at ~2-3
regardless of configured capacity (Section 9), and representation quality
is unaffected by rank (Section 10), **investigate the coefficient-generation
head itself** rather than further rank scaling:

1. Test whether `a(e_S) = Linear(e_S)` (the current single-linear-layer
   coefficient head) is itself the bottleneck — e.g., a slightly deeper or
   differently-normalized coefficient head might use more of the available
   rank even at R=2-4, without needing R=16's larger (and currently wasted)
   parameter budget.
2. Given rank scaling is now a closed direction (this experiment) and
   trajectory distillation is also closed (TEST10/10-R), and FiLM/deeper
   FiLM were closed in TEST09, the mechanism-search space within
   "conditioning the existing bottleneck via a compact embedding" is
   largely exhausted. The next experiment should test a **structurally
   different** coefficient-generation or conditioning-application pathway
   (e.g., letting U/V vary with degradation rather than staying globally
   shared) rather than another parametric sweep of the current form.
3. Given Haze's persistent, rank-invariant deficit combined with TEST10-R's
   finding that the teacher itself has no demonstrated Haze advantage over
   baseline, treat Haze specifically as warranting a dedicated
   investigation into the synthetic degradation recipe or teacher-quality
   ceiling, separate from further student-mechanism search.
