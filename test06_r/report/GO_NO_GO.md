# TEST06-R — GO / NO-GO Decision (Corrected Re-Run)

## Does the original TEST06 result survive statistical rigor?

**Yes — and more decisively than before.** With balanced N=144 (24 scenes × 3 recipients × 2 donors, scene_021 excluded for a confirmed data-quality reason — see `rerun_audit.md`), paired scene-level bootstrap CIs, permutation tests, and Wilcoxon signed-rank tests, the primary same-scene cross-degradation swap is **statistically indistinguishable in practical terms** from all four controls (cross-scene, random-matched, zero, global-mean).

## The statistical nuance (important, not hidden)

The paired tests are technically "significant" (Wilcoxon p < 1e-15 for all four comparisons) — but this is a textbook case of statistical significance without practical significance. With N=144 tightly-paired, low-variance observations, even a trivially small and consistent difference becomes detectable. The actual mean differences (primary − control) are **-1.2×10⁻⁷ to -2.5×10⁻⁷** — and critically, they are **negative** (primary slightly *smaller* than controls), the opposite direction a genuine causal frequency effect would require. A pre-specified equivalence threshold (ε = 2.8×10⁻⁷, derived purely from control-vs-control noise, before any primary comparison was examined) classifies **all four comparisons as "practically equivalent."**

## The new mechanistic finding — where the signal disappears

Internal propagation tracing through AFLB3 shows a dramatic, informative cascade:

| Stage | Relative change vs. normal | Cosine similarity |
|---|---:|---:|
| raw_high | 51.2% | 0.933 |
| raw_low | 29.1% | 0.996 |
| mined_high | 2.3% | 1.003 |
| mined_low | 3.0% | 1.002 |
| agg | 0.17% | 1.002 |
| cross_agg_out | 1.15% | 1.003 |
| aflb_out | 3.2×10⁻⁷ % | 1.006 |
| final_output | 1.25×10⁻³ % | 1.000 |

The frequency-path swap **does** propagate through FMiM (`channel_cross_l`/`channel_cross_h`) and FMoM (`frequency_refine`) — `mined_high`/`mined_low`/`agg`/`cross_agg_out` all show real, non-trivial change (0.17–3%). The collapse happens at the very last step: `aflb_out = out*para1 + y*para2`. **AFLB3's trained `para1` has converged to essentially zero** (mean=-0.000155, std=0.0015 — ~200× smaller than `para2`'s mean=0.0297), so the entire frequency-processed branch (`out`) is gated to near-nothing before being combined with the spatial residual (`y`). This is a **learned** near-zero gate, not an architectural constraint, and not evidence that FMiM/FMoM "ignore" frequency information — they process it; the AFLB's own trained residual weighting then discards it.

## Per-degradation and per-donor-direction (heterogeneity check)

Noise-recipient swaps show a mean effect (0.000018) roughly 1.8× larger than Rain/Haze-recipient swaps (0.000010) — a real, reportable difference, but still 3–4 orders of magnitude below what TEST04/TEST05.5 established as a genuine causal effect at other tensor points in this architecture. No donor direction shows an outlier effect; heterogeneity exists but does not hide a masked strong effect in any specific pairing.

## Donor-Behavior

0.0% of 144 swaps moved the output closer to the donor's own behavior than to the recipient's. Mean(d_recipient − d_donor) = -0.138 (scene-level bootstrap 95% CI: [-0.187, -0.103]) — the swapped output remains firmly anchored to the recipient's own normal output, regardless of donor.

## Outcome Classification (per the task's own framework)

**OUTCOME 2**: primary frequency intervention is not larger than controls, BUT internal FMiM/FMoM features change substantially (mined_high/mined_low/agg/cross_agg_out all show real, non-negligible relative change). **Conclusion: frequency information IS consumed internally by FMiM/FMoM, but AFLB3's own trained residual gate (`para1≈0`) suppresses its effect on the final restoration output.**

## Precise Claim Discipline

- "For the released 3-degradation AdaIR checkpoint, at 1024×1024, AFLB3, under the tested same-scene cross-degradation intervention, the frequency-path effect on final restoration was **not** detectably greater than matched controls (practically equivalent, pre-specified ε=2.8×10⁻⁷)."
- Separately: "The intervention **did** measurably alter downstream FMiM/FMoM intermediate representations (mined_high/mined_low/agg/cross_agg_out), but this internal change is suppressed before reaching the final AFLB output by a learned near-zero residual gate (`para1`)."

These are two different, both-true claims — not a contradiction.

## Relation to the AdaIR Paper (preserved distinction)

TEST06-R is an inference-time intervention on a frozen, released checkpoint. It does not test or refute whether frequency-aware processing was useful during the original training process (the paper's Table 7–9 trained ablations answer a different question). TEST06-R answers: "does the frequency-path tensor currently influence this trained checkpoint's computation at this operating point?" The mechanistic finding here — that AFLB3's own trained `para1` suppresses the branch — is actually a plausible, testable explanation for why a trained-from-scratch model WITH forced frequency reliance (the paper's ablations) could show a benefit, while THIS specific checkpoint's own training run converged to a solution where AFLB3 doesn't need to lean on it. That is a hypothesis about training dynamics, not a claim TEST06-R can verify without retraining (explicitly out of scope).

## Final Decision

**NO-GO for a frequency-derived spatial-kernel student signal — confirmed, with higher confidence than TEST06's original result**, and now with a mechanistic explanation rather than a black-box null. The recommended path remains TEST05.5's validated compact-embedding distillation.

**One new, narrow, optional follow-up is justified** (not required): if there is future interest in why `para1` converged near zero for this checkpoint specifically, a lightweight audit of `para1`/`para2` across training checkpoints (if intermediate checkpoints exist) or across the paper's other released checkpoints (e.g. 5-task variant) could clarify whether this is a general pattern or specific to this 3-degradation training run — but this is explicitly NOT a frequency-band, compact-signature, or dynamic-kernel experiment, and does not change today's GO/NO-GO.
