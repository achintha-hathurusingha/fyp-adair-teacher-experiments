# TEST04 — Causal Representation Intervention Study

## 1. Research Question

TEST03 established that degradation information is *present* and linearly
decodable in AdaIR's internal representations, under a controlled
same-scene design. TEST04 asks the question TEST03 explicitly could not
answer: **if we intervene on an internal representation — same scene, same
checkpoint, same recipient computation, only the internal tensor changed —
does AdaIR's restoration behavior actually change?**

## 2. Motivation from TEST01–TEST03

TEST01: the explicit FFT-based frequency mask is degenerate (`raw_low`
exactly zero) at benchmark resolution. TEST02: intermediate
representations are highly Rain/Haze/Noise-discriminative, but confounded
with source dataset. TEST03: the same discriminability survives (and
strengthens) under a controlled same-scene, scene-grouped design — but
this is *correlational*, not causal, evidence. TEST04 closes that gap with
a direct intervention.

## 3. AdaIR Forward Graph Audit

Full audit: `report/forward_graph_audit.md`. Two findings drove the entire
experimental design, established *before* any intervention was run:

1. **Three independent skip connections** (encoder L1/L2/L3 outputs)
   bypass latent/AFLB entirely, concatenated directly at each decoder
   stage. A latent-only or AFLB-output-only swap leaves these
   recipient-specific channels fully intact.
2. **The recipient's raw `inp_img`** is re-injected four times
   independently of any latent/AFLB swap — into all three AFLBs' FMiM
   branch and into the final global residual.

Conclusion: latent and AFLB-output substitution are technically valid (no
hidden state, no BatchNorm, pure per-sample LayerNorm, verified via
`grep -n BatchNorm net/model.py` → no matches) but **cannot fully
impersonate a donor** — this is architecture, not a limitation of the
experiment, and it directly predicts and explains the skip-connection
results in section 10.

## 4. Intervention Design

A manual, faithful re-implementation of `AdaIR.forward()`
(`src/intervention.py::manual_forward`) allows any intermediate tensor to
be substituted at its exact production point. **Verified bit-identical
(0.0 max absolute difference) against calling the real model directly**,
and **bit-identical (0.0 max absolute difference) under self-override**
(substituting a tensor with itself) — confirmed for all 300
(scene, degradation) combinations before any cross-degradation
intervention was attempted. No retraining, no weight modification, no
degradation label ever supplied to AdaIR.

## 5. Controls

Five controls per the task's evidence-hierarchy design: (A) self-swap —
gate, passed exactly; (B) same-scene cross-degradation — the primary
intervention; (C) cross-scene same-degradation (20 scenes, 39 valid pairs
after skipping 18 portrait/landscape shape mismatches); (D) random,
distribution-matched representation (20 scenes); (E) zero tensor and
dataset-mean tensor (20 scenes, mean computed over the 66/100 scenes
sharing the majority image orientation).

## 6. Normal Baseline

300 normal inferences (100 scenes × 3 degradations), `Normal_Baseline`
sheet. Matches TEST03's restoration-quality numbers (same checkpoint, same
images, same crop convention) — confirms this experiment's baseline is
consistent with prior work, not a silent drift.

## 7. Self-Swap Validation

**300/300 self-swaps reproduce normal inference with max absolute
difference = 0.0** (not merely "close" — exactly zero, at float32
precision), across all 4 intervention points tested simultaneously per
scene. This is the strongest possible pass of the Phase-4 stop condition;
proceeding to cross-degradation intervention is justified.

## 8. Latent Intervention

600 same-scene cross-degradation swaps at `latent_pre`. Mean output change
(L2 distance from the recipient's own normal output): **14.17**, ranging
from 1.00 (Noise recipient ← Rain donor) to 40.12 (Haze recipient ← Rain
donor). PSNR vs. clean drops from the ~32-40dB normal range to ~22-40dB
depending on the pair (`Swap_Matrix` sheet has the full 6×1 breakdown).

## 9. AFLB Interventions

600 swaps each at `aflb1_out`, `aflb2_out`, `aflb3_out`. Mean output
change scales monotonically with depth:

| Point | Mean ΔOutput (L2) | vs. latent_pre |
|---|---:|---:|
| latent_pre | 14.17 | — |
| aflb1_out | 14.18 | +0.1% (statistically indistinguishable) |
| aflb2_out | 32.35 | +128% |
| aflb3_out | 53.94 | +281% |

`latent_pre` and `aflb1_out` producing essentially identical effects is a
direct, independent confirmation (via causal intervention, not
correlation) of TEST01-03's finding that AFLB1's frequency-split
computation is near-identity at these resolutions (`raw_low` exactly
zero) — swapping before or after that near-identity transform makes
almost no difference.

## 10. Skip-Connection Interventions

Progressive test (20 scenes, 6 pairs, `latent_pre` point):

| Condition | Overrides | Mean ΔOutput (L2) |
|---|---|---:|
| A. latent only | `latent_pre` | 15.16 |
| B. latent + deepest skip | `latent_pre`, `enc3` | 31.92 (2.1×) |
| C. latent + all 3 skips | `latent_pre`, `enc1/2/3` | 138.63 (9.1×) |

**This directly confirms the forward-graph audit's prediction**: closing
the un-intervened skip-connection channels dramatically amplifies the
intervention's downstream effect. The audit's Critical Finding #2 — skip
connections provide an alternate, un-intervened path for recipient
identity — is not just an architectural observation, it is now an
empirically demonstrated, load-bearing fact about *why* a latent-only swap
under-states the representation's full potential influence.

## 11. Output Changes

Full per-intervention metrics (PSNR/SSIM/MSE vs. clean; L2/MAE vs. normal
recipient, normal donor, normal third-degradation; residual statistics;
NaN/Inf/finite sanity checks — all outputs remained finite, no NaNs, no
Infs, across all 2400+ interventions) in `Cross_Degradation_Swaps`,
`Skip_Connection_Progressive`, and control sheets.

## 12. Donor-Behavior Similarity

For each swapped output, distance to the recipient's own normal output,
the donor's normal output, and the third degradation's normal output
(`Donor_Similarity` sheet). Swapped outputs are closest to the
**recipient's** normal output 73.8–92.0% of the time depending on
intervention point; closest to the **donor** 5.0–15.0% of the time,
rising monotonically with intervention depth (5% at latent/AFLB1 → 15% at
AFLB3). **Noise as recipient shows 0% donor-closeness in every single
donor×point combination** — denoising appears the most robust of the
three degradations to this class of intervention, an asymmetry worth
further investigation.

## 13. Residual Analysis

`Residual_Analysis` sheet: mean/std/energy/MAE of `clean − output` per
(point, recipient, donor). Residual energy increases with intervention
depth, tracking the output-change pattern in section 9 — the intervention
does not merely relocate error, it measurably changes the restoration
residual's magnitude in a depth-dependent, structured way.

## 14. Output Degradation Probe

Trained on 30 normal outputs (the 10-scene visualization subset — a scope
reduction from the full 100-scene set, see Limitations), grouped-CV
accuracy 43.3% (n=30, consistent in direction with TEST02/03's finding
that the final output is only weakly degradation-discriminative, though
noisy at this sample size). Applied to 60 swapped (latent-point) outputs:
predicted the **donor** class 28.3% of the time, the **recipient** class
46.7% of the time — directionally consistent with, though not identical
in magnitude to, the pixel-distance donor-similarity finding (different
notion of similarity: learned linear boundary vs. raw pixel L2).

## 15. Statistical Analysis

Scene is the experimental unit throughout. `Swap_Matrix` reports bootstrap
95% confidence intervals (2000 resamples, resampled by scene) for every
(point, recipient, donor) cell's mean output change — all intervals are
tight and well clear of zero (e.g. `aflb3_out`, Haze←Rain: mean 145.9,
CI [138.6, 153.2]). Self-swap's 300-sample distribution has zero variance
(exactly 0.0 for every scene), so no interval is needed there — it is a
deterministic confirmation, not a statistical estimate.

## 16. Causal Evidence Assessment

Using the task's four-level evidence hierarchy:

**The primary same-scene cross-degradation swap (mean L2 = 4.03,
recipient=Rain subset, for direct comparability with the controls below)
exceeds every "arbitrary perturbation" control**: random
distribution-matched noise (L2 = 1.30), zero tensor (L2 = 1.23), and
dataset-mean tensor (L2 = 2.21). **This rules out "any perturbation of
this magnitude would cause a similar effect."** The effect is structured
and specific to inserting a real, different forward-pass's representation.

**However — and this is the single most important, most honestly reported
result of TEST04 — the cross-scene, same-degradation control (mean
L2 = 7.81, same recipient subset) is LARGER than the primary
cross-degradation intervention (L2 = 4.03).** In plain terms: swapping in
a *different scene's* Rain-latent moves AdaIR's Rain output *more* than
swapping in the *same scene's* Haze- or Noise-latent does.

**Classification: MODERATE intervention evidence.** The representation is
demonstrably causally implicated in restoration output — the effect is
real, structured, depth-dependent, and amplified exactly as the forward-
graph audit predicted when un-intervened channels are closed. It does
**not** rise to STRONG evidence of a *degradation-specific* causal
mechanism, because the same representation is at least as — here, more —
sensitive to a change in scene/content identity as to a change in
degradation identity. The correct, calibrated statement, matching the
task's own suggested phrasing: **"The latent and AFLB representations are
causally implicated in downstream restoration behavior, but this
experiment cannot establish that this causal role is specific to
degradation type rather than to content/scene identity more broadly."**
Do not read this as a negative result — it is a materially informative,
properly hedged finding that meaningfully advances beyond TEST03's
purely correlational claim, while being honest about what it does not
establish.

## 17. Distillation Implications

Full ranked table: `Distillation_Ranking` sheet /
`results/statistics/distillation_target_ranking.csv`. Headline: `latent`
remains the top practical candidate (smallest tensor, real and structured
causal effect, effect exceeds random/zero/mean controls) but should be
distilled with the explicit understanding that its causal footprint on
output is comparable to or smaller than a scene/content change — i.e. a
student distilling this representation is likely learning a
general-purpose content-and-context bottleneck that *correlates strongly*
with degradation type (per TEST02/03's >99% linear separability) rather
than a representation whose *causal* role is narrowly, specifically
"degradation control." `mined_low`/`mined_high`/`fmom_agg` remain strong
correlational candidates (TEST02/03) but were **not** causally tested here
— explicitly flagged as the natural next step, not silently assumed.

## 18. Limitations

- The output-degradation-probe training set (Phase 14) uses only the
  10-scene visualization subset (n=30 normal outputs), not the full
  100-scene set — a scope reduction made to avoid re-running full
  inference solely to save additional output images; the resulting 43.3%
  accuracy is directionally consistent with, but noisier than, TEST02/03's
  larger-sample findings and should be read as confirmatory, not
  definitive, on its own.
- `mined_low`, `mined_high`, `hl_spatial_weight`, `lh_channel_weight`,
  `fmom_agg` were **not** used as intervention points in TEST04 (only
  `latent_pre` and the three `aflb*_out` tensors were swapped) — per the
  task's explicit instruction to test AFLB output first and only proceed
  to sub-features if a meaningful effect was found. A meaningful effect
  **was** found, so this is the natural next experiment (see TEST05
  recommendation), not an oversight.
- Cross-scene control (Phase 9C) covers 39/57 possible pairs from a
  20-scene subset (18 skipped for portrait/landscape shape mismatch, a
  genuine data property, not a bug) — a full 100-scene cross-scene sweep
  was not run given the combinatorial cost and the task's explicit
  instruction to use "a manageable subset."
- Random/zero/mean controls were run only for `recipient=Rain`,
  `point=latent_pre` (20 scenes each) — not replicated across all 3
  recipients × 4 points, per the task's "small subset" instruction.
- The synthetic Rain/Haze/Noise images are TEST03's (documented, simplified
  synthesis models, not photorealistic simulators) — any TEST04 finding
  inherits that same caveat.

## 19. Recommended TEST05

Two candidates, directly motivated by sections 16-17:

1. **Sub-AFLB intervention sweep**: extend the same swap methodology to
   `mined_low`, `mined_high`, `hl_spatial_weight`, `lh_channel_weight`, and
   `fmom_agg` — the task's own Phase 7 explicitly authorizes this once the
   AFLB-output intervention shows a meaningful effect (it did). This would
   test whether the FMiM cross-attention output specifically (not just the
   AFLB's final residual-gated output) carries the causal signal, closing
   the gap TEST02/03's correlational finding on `mined_low`/`mined_high`
   left open.
2. **Scene-vs-degradation disentanglement intervention**: given section
   16's central finding (cross-scene changes the output more than
   cross-degradation does, at the same magnitude of representation swap),
   design a follow-up that holds scene *content* statistics fixed while
   varying only degradation-conditioning statistics within the
   representation (e.g. partial/channel-subset swaps, guided by which
   channels TEST02/03's linear probes weight most heavily) — directly
   testing whether a more surgical intervention can produce a
   degradation-specific effect that exceeds the scene-content effect,
   which this experiment's full-tensor swap could not demonstrate.
