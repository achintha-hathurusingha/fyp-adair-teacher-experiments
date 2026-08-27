# TEST13 — Adaptive Low-Rank Operator Basis

## 1. Motivation

TEST12 established a causal result: the conditional operator benefits from
both degradation state `e_D` and current spatial content `φ(F)` — the
shuffled-content control cost -2.91dB, the worst of four tested conditions.
TEST11 separately established that increasing coefficient rank (2→4→8→16)
does not reliably improve restoration. Combined, these point toward a new
hypothesis: **H_BASIS** — the operator's fixed spatial/channel basis
`U0, V0` (not the coefficients that select among its directions) may be the
real constraint on the remaining restoration gap.

## 2. Evidence from TEST11/12

- TEST11: `F2≈F4≈F8≈F16` — more coefficients is not the correct direction.
- TEST12: shuffled-content (-2.91dB) confirms the operator is genuinely
  content-sensitive, not merely degradation-sensitive; content-only
  (e_D=0) costs almost nothing (-0.03dB).

## 3. Mathematical Formulation

`K(e,F) = K0 + U(e)·diag(a(e,F))·V(e)ᵀ`, with `U(e) = U0 + ΔU(e_D)`,
`V(e) = V0 + ΔV(e_D)`. `ΔU`, `ΔV` come from two small `Linear(16, 256·2)`
heads taking **only** `e_D` (not `φ(F)`) — this separation is intentional,
isolating whether *basis* adaptation vs. *coefficient* adaptation is the
missing mechanism. Coefficients `a = G([e_D; φ(F)])` are unchanged from
TEST12. Both heads are zero-initialized so `U(e)≈U0`, `V(e)≈V0` at start.

## 4. Models

- **A**: baseline locked NAFNet.
- **F2**: TEST12's validated operator (fixed basis, content-conditioned
  coefficients) — the reference model.
- **T13**: F2 + adaptive basis as above. Rank fixed at R=2 throughout, per
  the task's explicit scope constraint.

Parameter check (mandatory gate before training): T13 adds 17,408 params
over F2 — **0.236% of base NAFNet**, under the 0.5% target, so training
proceeded without redesign.

## 5. Training Setup

Exact TEST07-B/12 dataset/split, reused read-only. Adam, LR=2e-4, batch=8,
50 epochs, seeds {0,1,2}, `L = L_restore + 0.1·L_KD` (unchanged from
TEST12, no new loss). All 9 runs completed cleanly, zero NaN/Inf. Models
A/F2 reproduce TEST12's own numbers exactly (deterministic).

## 6. Restoration Results

| Model | Mean PSNR (dB) | Mean SSIM |
|---|---|---|
| A | 27.315 | 0.815 |
| F2 | 27.116 | 0.830 |
| T13 | 26.610 | 0.823 |

| Comparison | Mean ΔPSNR (dB) | Same-sign (of 3) | Mean ΔSSIM | Same-sign |
|---|---|---|---|---|
| **T13 − F2** | **-0.506** | **3/3** | **-0.0070** | **3/3** |
| T13 − A | -0.705 | 3/3 | +0.0080 | 2/3 |
| F2 − A | -0.199 | 2/3 | +0.0150 | 2/3 |

**T13 underperforms F2 consistently and by a clear margin** — negative in
all 3 seeds for both PSNR and SSIM, 95% bootstrap CI for PSNR entirely
negative ([-0.790, -0.324]). This is not a null result; it is a genuine
regression from adding basis adaptation.

## 7. Haze Analysis

Per-degradation T13-F2: **Rain -1.248dB** (the worst-hit degradation),
Haze -0.239dB (also negative), Noise -0.033dB (roughly flat). Haze does
**not** improve — if anything it is mildly worse, and Rain (where the
teacher and F2 already perform best) is damaged most. This directly
contradicts the "GO" and "PARTIAL GO" criteria, both of which required at
minimum `T13 > F2`.

## 8. Basis Adaptation

Relative basis change (`‖ΔU‖/‖U0‖`, `‖ΔV‖/‖V0‖`), by degradation:

| Degradation | Rel. change U | Rel. change V |
|---|---|---|
| Rain | 8.56 | 10.25 |
| Haze | 10.33 | 13.73 |
| Noise | 9.36 | 10.84 |

**These are not small.** Despite zero initialization, training pushed the
correction to become **8.5-13.7x larger than the original basis itself** —
the "light adaptation, starts near F2" design intent (Section 3) held only
at initialization, not after training. The basis is not being lightly
tuned; it is being almost entirely replaced by a per-sample correction.
Haze shows the largest relative change of the three degradations, echoing
the pattern (largest conditioning magnitude, unclear restoration payoff)
seen for Haze since TEST08-C — but here it is now true of the *entire*
basis, not just a coefficient.

## 9. Effective Basis Rank

Effective rank (participation ratio) of `ΔU`: 3.37 (mean across seeds,
individual seeds 2.63-4.48); `ΔV`: 4.05 (3.63-4.29). Both **exceed** the
configured rank of 2 — meaning the correction varies across validation
samples along more independent directions than the base operator's own
rank. Singular values of the realized `U(e)`, `V(e)` (Section README /
`Basis_Adaptation` sheet) show substantial spread (std comparable to or
larger than half the mean for several components) — the adapted basis is
neither collapsing to a single direction nor staying near `U0`/`V0`; it is
genuinely, highly variable. **The basis is being adapted with real
diversity — the problem is not insufficient adaptation, but that this
adaptation is unhelpful or actively harmful.**

## 10. Causal Controls

| Condition | Mean PSNR (dB) | Δ from Normal |
|---|---|---|
| Normal (real e_D, real φ(F)) | 26.64 | — |
| Zero e_D | 26.35 | -0.29 |
| Mean content | 26.62 | -0.02 |
| Shuffled content | 26.51 | **-0.13** |
| Shuffled basis state (donor e_D) | 25.99 | **-0.65** |

Compare directly to TEST12's T12 (identical control methodology, fixed
basis): shuffled-content there cost **-2.91dB**. Here it costs only
**-0.13dB** — the strong, unambiguous content-causal signature TEST12
established has been **substantially washed out**. In its place, a new and
larger sensitivity has appeared: **shuffled-basis-state (swapping in a
different scene's `e_D` to drive the basis correction) is now the single
worst condition**, worse than even zero-ing `e_D` outright. This indicates
T13's operator has become primarily sensitive to *which donor degradation
embedding* generated its (now-dominant) basis correction, rather than to
the actual content it is restoring — a regression in causal structure, not
just in restoration quality.

## 11. Representation Analysis

| Model | e_D probe accuracy | Teacher cosine |
|---|---|---|
| F2 | 96.5% | 0.9892 |
| T13 | 96.5% | 0.9878 |

Statistically unchanged from F2 — confirms the restoration regression is
not attributable to a worse compact degradation representation; it is
specific to the operator mechanism itself.

## 12. Complexity

| Model | Params | Extra vs. F2 | MACs @128px |
|---|---|---|---|
| A | 7,371,923 | — | 1,033,040,896 |
| F2 | 7,398,149 | — | 1,033,131,584 |
| T13 | 7,415,557 | +17,408 (0.236% of A) | 1,033,147,968 |

Parameter overhead is small and within the pre-specified 0.5% target —
this was never the limiting factor; the operator's *behavior*, not its
budget, is the problem. Theoretical complexity only; no NPU latency claim.

## 13. Limitations

- N=3 seeds, but T13-F2's negative result is unusually consistent (3/3
  same-sign for both PSNR and SSIM, tight relative to prior experiments'
  more ambiguous comparisons) — this is one of the more statistically
  confident findings in the TEST08-13 series, if a negative one.
- Only one specific basis-adaptation formulation was tested (unconstrained
  additive `ΔU`, `ΔV` from `e_D` alone via a single linear layer each). A
  constrained (e.g., norm-bounded) or differently-parameterized adaptation
  might behave differently — but per the task's own "do not add more basis
  capacity" instruction for this failure mode, that is future work, not
  this experiment's scope.
- The mechanism by which "large ΔU/ΔV" translates to "worse restoration
  AND weaker content-causality" is not fully explained by this experiment
  — plausibly the free-form correction lets training find a
  loss-minimizing basis that overfits to `e_D`-driven shortcuts rather
  than genuinely better spatial operators, but this is inference, not a
  directly measured mechanism.

## 14. GO / NO-GO

Per the task's decision rule:

- **GO** requires `T13 > F2` consistently with meaningful Haze
  improvement and small parameter overhead. **Not met** — T13 < F2
  consistently (3/3 seeds), and Haze does not improve.
- **PARTIAL GO** requires `T13 > F2` with Haze unchanged. **Not met** —
  T13 does not beat F2 at all.
- **NO-GO** requires `T13 ≈ F2` with `ΔU`/`ΔV` near-zero. **Not quite
  this either** — `T13` is not merely equal to F2, it is measurably
  worse, and `ΔU`/`ΔV` are emphatically not near-zero.
- **Interesting negative** (the task's fourth, explicitly-named category):
  `ΔU`/`ΔV` become large but restoration does not improve → the basis is
  being adapted but the operator family is not expressive in the required
  direction; motivates a different operator formulation, not more basis
  capacity. **This is exactly what occurred**, and additionally the basis
  adaptation actively *degrades* both restoration and the previously-
  established causal content-sensitivity.

**Decision: NO-GO, classified as the task's "Interesting Negative"
outcome.** Allowing the low-rank operator's basis to freely adapt from
`e_D` alone — with no constraint tying it back to F2's validated,
content-aware coefficient mechanism — produces a large, high-diversity,
but harmful correction. Per Section 10 of the task's scientific rules
("preserve negative results"), this is reported as a genuine, informative
negative finding, not discarded.

## 15. Recommended Next Direction

Do not pursue further unconstrained basis-adaptation variants of this
operator family. Instead:

1. **Revert to F2's fixed basis** as the reference point for any future
   operator work — it remains the strongest validated mechanism to date
   (TEST09→11→12).
2. If basis adaptability is still worth investigating, constrain it much
   more tightly than this experiment did — e.g., cap `‖ΔU‖/‖U0‖` and
   `‖ΔV‖/‖V0‖` well below 1.0 (a true "light adaptation," not the 8-14x
   observed here), or make the basis correction conditioned on `φ(F)` as
   well as `e_D` (mirroring F2's coefficient generator, rather than the
   deliberately separated e_D-only design used here) to test whether
   content-awareness in the basis itself — not just the coefficients —
   is the missing piece, with proper regularization to prevent the
   runaway magnitude seen in this run.
3. Given three consecutive operator-family variations (deeper FiLM in
   TEST09, trajectory distillation in TEST10/10-R, and now free basis
   adaptation in TEST13) have each failed to beat F2/T12's validated
   content-conditioned fixed-basis mechanism, treat that mechanism as a
   stable baseline and shift investigation toward orthogonal directions
   (e.g., multi-depth application of the *existing, validated* F2
   operator, or restoration-quality-targeted regularization) rather than
   continuing to search within the "modify the operator's internal
   structure" family.
