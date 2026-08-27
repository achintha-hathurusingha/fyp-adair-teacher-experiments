# TEST06-R — Corrected Re-Run / Statistical + Internal Propagation Audit

## 1. Purpose

TEST06 established a null result (frequency-path swap indistinguishable from controls) with two acknowledged weaknesses: unbalanced N (150 primary vs. 25 per control) and mean-only comparison with no paired statistics. This corrected re-run (1) rebalances every control to N=144 (matched exactly to the primary's index), applies scene-level bootstrap CIs, paired permutation tests, and Wilcoxon signed-rank tests, and a pre-specified practical-equivalence threshold; and (2) traces the intervention through AFLB3's internal FMiM/FMoM stages to determine *where* the frequency-swap signal disappears, not just *that* it disappears.

## 2. Relation to the AdaIR Paper

Unchanged from TEST06: this is an inference-time intervention on a frozen, released checkpoint, and does not test or refute the paper's trained-architecture ablations (Table 7–9). See `GO_NO_GO.md` for the precise distinction, including a new hypothesis (untested here) connecting the mechanistic finding to training dynamics.

## 3. Phase 0 — Verification of Original TEST06

Confirmed matching: checkpoint SHA256, AdaIR source git SHA, AFLB3 intervention point, 1024×1024 resolution, degradation synthesis method. The original intervention mechanism is correctly implemented (its own self-swap control produced exact 0.0 difference). Full detail in `report/rerun_audit.md`.

**Data-quality finding**: `scene_021` in the original TEST06 dataset is 1024×104, not 1024×1024 — a bug in `test06/src/build_06e_dataset.py`'s unguarded center-crop, surfaced when this re-run's global-mean control required stacking all scene tensors. Excluded from TEST06-R (N=24 scenes); `test06/` was not modified. Full detail in `rerun_audit.md`.

## 4. Balanced Design

Primary: 144 same-scene cross-degradation swaps (24 scenes × 3 recipients × 2 donors), unchanged in kind from TEST06. Controls: cross-scene (deterministic cyclic donor, every scene appears as donor exactly once), random-matched (seeded by scene+recipient+"random", excluding donor per the task's own seeding rule), zero, and global-mean (computed from all 72 real AFLB3 tensors in the 06-E set, strictly within-dataset). Each control computed once per (scene, recipient) — mathematically identical for both donor-labeled rows of a pair — and expanded to the full 144-row table, avoiding 2× redundant GPU compute for bit-identical results (documented in `balanced_intervention.py`'s docstring).

## 5. Self-Swap Control (Phase 4)

Max L2 = 0.00000000 (exact), mean L2 = 0.00000000, max normalized L2 = 0.00000000. **PASS** — confirms the intervention mechanism before any causal claim.

## 6. Primary vs. Balanced Controls

| Condition | N | Mean normalized L2 |
|---|---:|---:|
| Primary (same-scene cross-degradation) | 144 | 0.000013 |
| Cross-scene, same-degradation | 144 | 0.000013 |
| Random, distribution-matched | 144 | 0.000013 |
| Zero | 144 | 0.000013 |
| Global mean | 144 | 0.000013 |

All five conditions round to the same value at 6 decimal places.

## 7. Paired Statistics

| Comparison | Mean difference | 95% bootstrap CI | Wilcoxon p | Practical conclusion |
|---|---:|---:|---:|---|
| Primary − cross-scene | -1.22×10⁻⁷ | [-1.46, -0.98]×10⁻⁷ | 2.1×10⁻¹⁶ | practically equivalent |
| Primary − random | -2.17×10⁻⁷ | [-2.49, -1.86]×10⁻⁷ | 2.9×10⁻²⁵ | practically equivalent |
| Primary − zero | -2.54×10⁻⁷ | [-2.89, -2.21]×10⁻⁷ | 2.4×10⁻²⁵ | practically equivalent |
| Primary − global-mean | -1.26×10⁻⁷ | [-1.50, -1.04]×10⁻⁷ | 3.8×10⁻¹⁸ | practically equivalent |

Every comparison is "statistically significant" (enormous paired power, N=144 low-variance pairs) yet **all differences are negative** (primary slightly smaller, the opposite direction a causal effect would require) and **all fall within the pre-specified practical-equivalence threshold** ε=2.8×10⁻⁷ (95th percentile of pooled control-vs-control pairwise differences, defined before any primary comparison was examined — see `statistics/epsilon_definition.txt`).

## 8. Per-Degradation and Per-Donor-Direction

Recipient=Noise shows mean effect 0.000018 (median 0.000017) vs. Recipient=Rain/Haze at 0.000010 — a real, ~1.8× difference, reported rather than averaged away. All six donor directions (Rain←Haze, Rain←Noise, Haze←Rain, Haze←Noise, Noise←Rain, Noise←Haze) cluster into two groups matching the recipient-level pattern; no single direction is an outlier masking a hidden strong effect.

## 9. Donor-Behavior

0.0% of 144 swaps moved output closer to donor than recipient. Mean(d_recipient − d_donor) = -0.138 (scene-level bootstrap 95% CI [-0.187, -0.103]) — output remains firmly anchored to the recipient's own behavior.

## 10. Internal Propagation Trace — the central new result

| Stage | Relative change | Cosine similarity |
|---|---:|---:|
| raw_high | 51.2% | 0.933 |
| raw_low | 29.1% | 0.996 |
| mined_high | 2.3% | 1.003 |
| mined_low | 3.0% | 1.002 |
| agg | 0.17% | 1.002 |
| cross_agg_out | 1.15% | 1.003 |
| aflb_out | 3.2×10⁻⁷ % | 1.006 |
| final_output | 1.25×10⁻³ % | 1.000 |

FMiM cross-attention (`mined_high`/`mined_low`) and FMoM gating (`agg`, `cross_agg_out`) all show real, non-trivial propagation of the swapped frequency content — an 8-order-of-magnitude collapse does NOT happen gradually across these stages; it happens in one discontinuous step between `cross_agg_out` (1.15%) and `aflb_out` (3.2×10⁻⁷%).

## 11. Mechanistic Explanation — the `para1` gate

`aflb_out = out*para1 + y*para2`. Direct inspection of the trained checkpoint's AFLB3 parameters: `para1` mean=-0.000155, std=0.0015 (range [-0.0045, 0.0035]); `para2` mean=0.0297, std=0.0252. `para1` is ~200× smaller in magnitude than `para2` and clusters tightly around zero — a **learned**, not architectural, near-total suppression of the frequency-processed branch's contribution to AFLB3's output. This is the direct mechanistic cause of the collapse in §10, and it is AFLB3-specific: AFLB1's `para1` std (0.036) and AFLB2's `para1` std (0.057) are 24–38× larger, suggesting this suppression pattern may not generalize to the other two AFLBs (neither of which was confirmed frequency-active by TEST06's resolution sweep, so this remains an open question, not tested here).

## 12. Case D Check — compensation/cancellation

Case D (intermediate tensors change substantially, final output still barely changes, suggesting later compensation) does not apply cleanly: the suppression is not a cancellation between competing signals but a direct multiplicative near-zero gate. This is closer to Case B in the task's own framework (mined features and agg change; final output does not) with a precise, verified mechanism rather than an unexplained "downstream architecture suppresses it."

## 13. Visualization

Three primary figures: `primary_vs_controls_balanced.png` (bar chart with SEM error bars, all five conditions visually indistinguishable), `internal_propagation_trace.png` (log-scale relative-change cascade, the key figure), `internal_propagation_cosine.png` (cosine similarity by stage). 18 absolute-difference heatmaps (3 scenes × 6 donor directions) in `visualizations/propagation_heatmaps/` — visual corroboration only, not used as standalone evidence.

## 14. Does the Corrected Result Change TEST06's Conclusion?

**No — it strengthens it**, while adding real mechanistic content TEST06 lacked. The null result is now statistically rigorous (paired, bootstrapped, pre-specified equivalence threshold) rather than a mean-only comparison, and we now know *why* (the `para1` gate) rather than only *that* the frequency path doesn't matter for final output.

## 15. GO/NO-GO

**NO-GO for a frequency-derived spatial-kernel student signal**, confirmed with higher confidence. See `GO_NO_GO.md` for full detail, including the precise two-claim distinction (internal features DO change; final output does NOT) and the one narrow, optional, non-required follow-up question about training dynamics.

## 16. Recommendation

Proceed to TEST07+ with TEST05.5's validated compact-embedding distillation (`z_T → e_D`). No further frequency-band, degradation-specific-frequency, or compact-frequency-signature experiments are justified — these remain correctly gated on a non-null 06-E result, which was not obtained even under the corrected, statistically rigorous design.
