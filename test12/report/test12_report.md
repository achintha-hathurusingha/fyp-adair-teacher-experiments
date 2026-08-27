# TEST12 — Feature-Conditioned Low-Rank Operator

## 1. Motivation

TEST11 showed that increasing the rank of the low-rank conditional operator
(R=2→4→8→16) did not produce consistent restoration gains, and that
effective rank stayed near 1-3 regardless of configured capacity — the
operator was not using more of the budget it was given. TEST12 asks a
different question about the *same* operator family: is the coefficient
generator missing an entire *input*, not just capacity? The existing
mechanism computes `a = G(e_D)` — coefficients depend only on the
degradation embedding, never on what the current image actually looks
like. TEST12 tests `a = G([e_D; φ(F)])`, where `φ(F) = GAP+GMP(F)` is a
cheap summary of the current bottleneck content.

## 2. Hypothesis

A degradation-aware operator may need to know not only *what* degradation
is present but also *what spatial content* it is currently acting on —
especially relevant to Haze, where prior experiments (TEST08-C through
TEST11) consistently found the highest conditioning magnitude paired with
the lowest restoration benefit.

## 3. Models

- **A**: baseline locked NAFNet.
- **F2**: TEST09/11's rank-2 mechanism, `a = G(e_D)`, `G` a single
  `Linear(16,2)`.
- **T12**: identical rank-2 low-rank operator (`U`, `V` unchanged), but
  `a = G([e_D; φ(F)])` via `Linear(528,32) → ReLU → Linear(32,2)`
  (528 = 16 + 512), final layer zero-initialized so the model starts at
  the identity transform, matching F2's initialization discipline.

## 4. Dataset and Training

Exact TEST07-B/09/11 dataset/split, reused read-only. Adam, LR=2e-4,
batch=8, 50 epochs, seeds {0,1,2}, `λKD=0.1` — 9 total runs, all completed
cleanly with zero NaN/Inf. Models A/F2 reproduce prior experiments' numbers
exactly (deterministic).

## 5. Overall Restoration Results

| Model | Mean PSNR (dB) | Mean SSIM |
|---|---|---|
| A | 27.315 | 0.815 |
| F2 | 27.038 | 0.825 |
| T12 | 27.117 | 0.831 |

| Comparison | Mean ΔPSNR (dB) | Same-sign (of 3) | Mean ΔSSIM | Same-sign |
|---|---|---|---|---|
| T12 − F2 | +0.078 | 2/3 | +0.0056 | **3/3** |
| T12 − A | -0.199 | 2/3 | +0.0150 | 2/3 |
| F2 − A | -0.276 | **3/3** | +0.0093 | 2/3 |

T12 beats F2 on SSIM consistently (3/3 seeds) and on PSNR in 2/3 seeds with
a positive mean. T12 also narrows the gap to baseline A relative to F2
(-0.199dB vs -0.276dB) — a real, if partial, improvement.

## 6. The Causal Test: Does the Operator Actually Use Content?

This is the central result of TEST12. Four inference-time conditions
(same trained T12 checkpoints, no retraining) on the validation set:

| Condition | Mean PSNR (dB) | Δ from Normal |
|---|---|---|
| **Normal** (real e_D, real φ(F)) | 27.19 | — |
| Degradation-only (real e_D, dataset-mean φ̄) | 25.83 | **-1.36** |
| Content-only (e_D=0, real φ(F)) | 27.16 | **-0.03** |
| Shuffled content (real e_D, WRONG image's φ(F)) | 24.28 | **-2.91** |

Two findings, both clean and unambiguous:

1. **Content-only ≈ Normal.** Zeroing the entire 16-dim degradation
   embedding costs almost nothing (-0.03dB) as long as the real content
   summary is present. Nearly all of the useful conditioning signal is
   coming from `φ(F)`, not from `e_D`.
2. **Shuffled content is the single worst condition** — worse even than
   degradation-only (which at least uses a generic, "average" content
   summary). Substituting a *wrong* image's content actively confuses the
   operator more than giving it no specific content at all. This is
   precisely the signature the task specified: *"if shuffled content
   changes restoration substantially, the operator is using image-specific
   context."* It does, decisively.

## 7. Haze Analysis

Per-degradation T12-F2 (mean across 3 seeds): Rain +0.070dB (seeds:
-0.12/+0.77/-0.44 — mixed, dominated by one outlier), Haze **+0.186dB**
(seeds: +0.55/+0.17/-0.17 — 2/3 positive, and the positive seeds are much
larger in magnitude than the one negative), Noise -0.023dB (seeds:
essentially flat, -0.005/-0.009/-0.055).

**Haze shows the largest and most encouraging per-degradation movement of
the three**, though not unanimous across seeds. Combined with Section 6's
unambiguous causal evidence, this is consistent with — though does not
fully prove at the per-degradation level — the hypothesis that Haze
specifically needed content information the degradation-only operator
could never access.

## 8. Operator Utilization

**Scene-variance of coefficients** (does `a` vary across scenes within one
fixed degradation — something F2 structurally cannot do, since its `a`
depends only on `e_D`):

| Degradation | T12 variance | F2 variance | Ratio (T12/F2) |
|---|---|---|---|
| Rain | 2.009 | 0.199 | **9.46x** |
| Haze | 0.559 | 0.166 | **3.73x** |
| Noise | 1.050 | 0.105 | **11.65x** |

T12's coefficients vary substantially more across scenes than F2's for
every degradation — confirming the mechanism is doing what it was designed
to do. Interestingly, **Haze shows the smallest variance ratio of the
three**, despite showing the largest restoration benefit (Section 7) —
this is a genuinely puzzling combination worth flagging rather than
smoothing over: Haze doesn't need the *most* additional scene-to-scene
variation to benefit the most from having it.

**Effective rank** (participation ratio of the coefficient covariance):
T12 = 1.056, F2 = 1.081 — essentially unchanged, both still low relative to
the configured rank of 2. Content-conditioning does not increase how much
of the low-rank basis is used in aggregate; it changes *what* the (still
low-dimensional) signal represents, not how many effective dimensions it
occupies. This means TEST11's "effective rank stays low" finding is not
contradicted — the benefit here comes from a qualitatively different, more
useful 1-dimensional-ish signal, not from unlocking more capacity.

## 9. Representation Analysis

| Model | e_D probe accuracy | Teacher cosine |
|---|---|---|
| F2 | 96.4% | 0.9898 |
| T12 | 96.5% | 0.9892 |

Statistically indistinguishable — the final compact degradation embedding
`e_D` itself is unaffected by whether the coefficient head additionally
sees content. This confirms the T12-F2 restoration difference is
attributable to the *conditioning mechanism*, not to a change in
degradation-representation quality.

## 10. Complexity

| Model | Params | Extra vs. F2 | MACs @128px |
|---|---|---|---|
| A | 7,371,923 | — | 1,033,040,896 |
| F2 | 7,381,189 | — | 1,033,114,656 |
| T12 | 7,398,149 | +16,960 | 1,033,131,584 |

T12's coefficient MLP adds 16,960 params (mostly the `Linear(528,32)`
layer) — still under 0.23% of the backbone. MACs overhead is negligible.
Theoretical complexity only; no NPU latency claim.

## 11. Limitations

- N=3 seeds; T12-F2's PSNR improvement is directionally positive on
  average but only 2/3 same-sign, so it should be read as suggestive, not
  definitive. The SSIM improvement is more robust (3/3 same-sign).
- The causal-control evidence (Section 6) is strong and unambiguous, but it
  establishes that the mechanism *works as designed* — it does not, by
  itself, establish that this translates into a large or fully reliable
  restoration benefit, which remains modest per Section 5.
- The Haze-specific finding (largest restoration gain, smallest
  scene-variance ratio) is a real pattern in this data but its
  interpretation is not fully resolved — more scenes or seeds would help
  determine if it is a genuine degradation-specific effect or partly
  driven by Haze's already-largest baseline modulation magnitude from
  prior experiments interacting with a new signal source.

## 12. GO / NO-GO

Per the task's decision rule:

- **GO** requires T12>F2 consistently across seeds, preferably with Haze
  improvement. **Not fully met** — PSNR same-sign is 2/3, not 3/3.
- **PARTIAL GO** requires overall gain small but Haze_T12 > Haze_F2
  meaningfully. **This matches**: overall PSNR gain is small and only
  2/3-consistent, while Haze shows the largest per-degradation movement of
  the three (+0.186dB) and the causal controls (Section 6) provide
  unambiguous, GO-level evidence that the underlying mechanism is doing
  genuine, image-specific work.
- **NO-GO** requires T12≈F2 with controls showing little change. **Not
  met** — the controls show large, decisive changes (shuffled content is
  -2.91dB, the worst of all conditions).
- **Important note** from the task: if T12 changes coefficients
  significantly but PSNR does not improve, that would motivate a
  *different operator basis*, not a larger coefficient generator. This is
  **partially relevant**: coefficients change substantially (Section 8)
  and PSNR *does* improve, but only modestly — suggesting the low-rank
  channel-mixing basis itself (not just its input) may still be a limiting
  factor, even once given the right information to act on.

**Decision: PARTIAL GO.** Content conditioning is a real, causally-verified
missing degree of freedom — not a training-noise artifact — but the
current low-rank operator basis only partially converts that information
into restoration quality. Continue developing feature-conditioned
operators, but do not conclude the mechanism is complete; the next
question is whether a richer operator basis (still content- and
degradation-aware) can convert this verified signal into a larger,
more consistent gain.

## 13. Recommended Next Experiment

Per the task's explicit guidance ("do not make the MLP larger" if the
result were NO-GO — inapplicable here, but the same restraint principle
applies to PARTIAL GO): do not simply enlarge the coefficient-generation
MLP. Instead:

1. Since content conditioning is verified useful but the rank-2
   channel-mixing basis limits how much benefit it produces, test a richer
   (but still NPU-friendly, non-attention, non-dynamic-convolution)
   operator basis that can make fuller use of a now-validated
   content-aware coefficient signal — e.g., allowing `U`/`V` themselves to
   be lightly content-modulated, rather than only the diagonal
   coefficients.
2. Investigate the Haze-specific puzzle from Section 8 directly: why does
   Haze show the *smallest* scene-variance ratio but the *largest*
   restoration gain? A targeted per-scene qualitative comparison of Haze
   outputs before/after content-conditioning would help distinguish "a
   small amount of the right content information matters a lot for Haze"
   from a smaller-sample-size artifact.
3. Combine with TEST11's finding: since raw rank is not the bottleneck
   and content is now shown to be a genuine missing input, the next
   capacity experiment (if any) should scale content-representation
   richness (e.g., a slightly larger φ(F), or a small spatial pooling
   grid instead of pure GAP+GMP) rather than rank R again.
