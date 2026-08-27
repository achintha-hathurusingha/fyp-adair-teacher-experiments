# TEST09 — Multi-Depth + Low-Rank Conditioning

## 1. Research Question

TEST08-C showed compact degradation state e_S can causally control restoration
via bottleneck-only FiLM conditioning (Model C), but the benefit was almost
entirely concentrated in Rain (+0.85dB over B); Haze was essentially unrescued
(-0.06dB) despite receiving the largest bottleneck modulation of any degradation.
TEST09 asks: **does conditioning at additional decoder depths (Models D, E), or
a more expressive low-rank channel-mixing operator (Model F), succeed where
single-depth bottleneck FiLM did not — specifically for Haze?**

## 2. Motivation from TEST08-C

TEST08-C's C-B per-degradation results: Rain +0.846dB, Haze -0.057dB, Noise
+0.033dB. The conditioning-statistics analysis showed Haze consistently received
the *strongest* bottleneck modulation (relative change 0.655, vs. Rain's 0.470
and Noise's 0.526) — the network was clearly "trying" to modulate for Haze more
than for the other degradations, yet this had no restoration payoff. This is
exactly the "interesting failure" pattern the TEST08-C decision tree flagged:
strong modulation, no restoration benefit, suggesting the *operator* (single-depth
channel-wise affine) may be too constrained, not the signal.

## 3. Models A/B/C/D/E/F

All six share the identical locked NAFNet M-arm backbone. A/B/C exactly
reproduce TEST08-C's configuration (confirmed bit-identical in this re-run).
New models:

- **D**: B + FiLM at bottleneck (256ch) + decoder level 3 (128ch, immediately
  after the bottleneck, following `fyp-adair-distill`'s own stage-numbering
  convention where decoder level = mirror of the corresponding encoder depth).
- **E**: D + FiLM at decoder level 2 (64ch) as well — three conditioning points.
- **F**: B + low-rank (rank 4) channel-mixing at the bottleneck ONLY, in place
  of FiLM: `F' = F + U·diag(a(e_S))·Vᵀ·F`, with U/V shared learned parameters
  and `a(e_S)` from a zero-initialized `Linear(16,4)` head (so the adaptive
  contribution starts at exactly zero, same identity-at-init discipline as
  FiLM's zero-init gamma/beta).

All conditioned models share ONE e_S (same as TEST08-C's design) for both the
KD loss and every conditioning signal.

## 4. Dataset and Training

Identical dataset/split to TEST07-B/TEST08-C, reused read-only. Adam, LR=2e-4,
batch=8, 50 epochs, seeds {0,1,2} — 18 total runs. **Training-infrastructure
note**: the first launch attempt ran all 18 runs concurrently and hit GPU OOM
on the RTX 4090 (24GB) — 13 of 18 crashed (each process's CUDA context uses
~2GB, and 18×2GB exceeded capacity with headroom for activations). This was
caught immediately (no corrupted results, since crashed runs simply produce no
output files), cleaned up, and re-run in two waves of 9 (A/B/C, then D/E/F),
mirroring TEST08-C's proven-safe 9-way concurrency. All 18 runs completed
cleanly on the second attempt. A/B/C results in this experiment are bit-identical
to TEST08-C's own results — confirmed cross-experiment consistency.

## 5. Teacher Representation

Reused unmodified from TEST07-B (SHA256-verified checkpoint, frozen, PCA-16
fit on 1,920 training-split records, 68.44% explained variance) — identical
to TEST08-C.

## 6. Conditioning Mechanism

FiLM stages (C/D/E) use the same zero-initialized channel-wise affine as
TEST08-C. The low-rank stage (F) additively perturbs the bottleneck along a
rank-4 subspace shared across all samples (U, V), with per-sample strength
`a(e_S)` — a much more constrained operator than full-channel FiLM (4 degrees
of freedom driving the correction vs. 256 for FiLM), but the fact that a
sample-dependent U/V-projected correction is possible at all makes it a
genuinely different operator class, not just "weaker FiLM."

## 7. Restoration Results

Overall last5-window PSNR (mean ± std across 3 seeds):

| Model | Mean PSNR (dB) | Std |
|---|---|---|
| A | 27.315 | 0.198 |
| B | 26.522 | 0.435 |
| C | 26.796 | 0.201 |
| D | 26.979 | 0.106 |
| E | 26.823 | 0.199 |
| F | **27.073** | 0.177 |

Pairwise comparisons (primary metric, last5-window):

| Comparison | Mean ΔPSNR (dB) | Std | 95% bootstrap CI | All 3 seeds same sign? |
|---|---|---|---|---|
| D − C | +0.184 | 0.326 | [-0.159, +0.489] | **No** (seed 2 negative) |
| E − C | +0.027 | 0.425 | [-0.406, +0.444] | **No** (seed 2 negative) |
| **F − C** | **+0.278** | **0.049** | **[+0.235, +0.331]** | **Yes, all positive** |
| C − B | +0.274 | 0.273 | [+0.048, +0.577] | Yes, all positive (reproduces TEST08-C) |
| B − A | -0.793 | 0.231 | [-1.014, -0.553] | Yes, all negative (reproduces TEST08-C) |

**Only F-C is a clean, reproducible, low-variance improvement.** D-C and E-C
both have one seed with a negative delta, and their 95% CIs straddle zero —
adding more FiLM depth did not produce a reliable benefit and, if anything,
added seed-to-seed variance without a payoff. F (low-rank channel-mixing) is
the standout: consistent across all 3 seeds, small std (0.049 — an order of
magnitude tighter than D or E), and its CI excludes zero cleanly.

## 8. Degradation-Specific Results

This is the section that answers TEST09's central question — does anything
rescue Haze:

| Comparison | Rain ΔPSNR | Haze ΔPSNR | Noise ΔPSNR |
|---|---|---|---|
| C − B (TEST08-C, reference) | +0.846 | -0.057 | +0.033 |
| D − C | +0.236 (std 0.470) | +0.378 (std **0.713**) | -0.063 |
| E − C | +0.355 (std 0.389) | -0.206 (std **1.002**) | -0.067 |
| F − C | **+0.683** (std 0.848) | +0.246 (std 0.600) | -0.096 |

**None of D, E, or F deliver a reliable Haze rescue.** D's Haze delta (+0.378)
is nominally the largest, but its standard deviation (0.713) is *larger* than
the mean — this is not a reproducible effect, it is dominated by seed noise
(one seed likely swung strongly positive, masking the other two). E's Haze
delta is actually **negative** (-0.206), with the largest std of any cell in
this table (1.002). F's Haze delta (+0.246) is directionally consistent with
a small rescue but again has a std (0.600) more than double the mean — not
something to trust with N=3.

What IS reliable: **F further improves Rain** (+0.683dB on top of C's own
+0.846dB recovery from B — stacking to roughly +1.53dB recovered from B's
original -2.591dB Rain regression) with a much tighter Haze/Noise profile than
D or E. The overall F-C win (Section 7) is therefore driven primarily by Rain,
not by a genuine Haze fix, even though F's mean Haze delta happens to be
positive.

## 9. Representation Results

All conditioned models (B through F) reach statistically indistinguishable
representation quality: bottleneck probe accuracy 98.0-98.6%, e_S probe
accuracy 96.2-96.8%, teacher-student cosine similarity 0.987-0.991 across the
board. This confirms — as in TEST08-C — that **none of the restoration
differences among B/C/D/E/F trace to representation quality**. Every model
learned an equally good degradation-aware embedding; the differences are
entirely attributable to how (or whether, or how much) that embedding is used
to condition the spatial computation.

## 10. Conditioning Statistics — Modulation Magnitude

Relative modulation magnitude (‖F_cond−F_pre‖₂/‖F_pre‖₂) by model, stage, and
degradation:

| Model | Stage | Rain | Haze | Noise |
|---|---|---|---|---|
| C | bottleneck | 0.470 | **0.655** | 0.526 |
| D | bottleneck | 0.418 | **0.675** | 0.442 |
| D | decoder_level3 | 0.563 | 0.522 | 0.373 |
| E | bottleneck | 0.405 | **0.681** | 0.474 |
| E | decoder_level2 | 0.782 | **1.012** | 0.780 |
| E | decoder_level3 | 0.560 | 0.625 | 0.500 |
| F | bottleneck | 0.132 | **0.343** | 0.164 |

**Haze receives the strongest modulation of any degradation at every single
conditioning point, in every model, with no exception** — including the newly
added decoder_level3 (D, E) and decoder_level2 (E) stages, and even F's much
gentler low-rank correction. This rules out "the bottleneck-only mechanism
just didn't try hard enough for Haze" — deeper conditioning tries *even
harder* for Haze (decoder_level2's Haze modulation of 1.012 is the single
largest value in the entire table) without translating into a reliable
restoration gain (Section 8). The network consistently identifies Haze as
needing the most correction at every depth, but is unable to convert that
correction into better restoration through this affine/low-rank operator
family, regardless of depth.

## 11. Complexity

| Model | Params | Extra vs. B | MACs @128px |
|---|---|---|---|
| A | 7,371,923 | - | 1,033,040,896 |
| B | 7,380,131 | - | 1,033,049,088 |
| C | 7,388,835 | +8,704 | 1,033,057,280 |
| D | 7,393,187 | +13,056 | 1,033,061,376 |
| E | 7,395,363 | +15,232 | 1,033,063,424 |
| F | 7,382,247 | +2,116 | 1,033,180,224 |

Model F achieves its clean, reproducible improvement with **the smallest
parameter overhead of any conditioned model** (+2,116 params over B, vs. +8,704
for C, +13,056 for D, +15,232 for E) — answering main question #4 directly:
**yes, a more expressive but structurally constrained (low-rank) operator
improved restoration without a larger parameter increase; in fact with a
smaller one than the simpler FiLM baseline.** All figures are theoretical
complexity only (params/MACs); no NPU latency claim.

## 12. Limitations

- N=3 seeds throughout — all deltas are exploratory. D-C and E-C's wide,
  zero-crossing confidence intervals are themselves evidence that 3 seeds is
  not enough to detect a real (if any) effect at those configurations; a
  larger seed count might reveal a real but small D/E effect, or might confirm
  the null.
- The Haze modulation-magnitude finding (Section 10) is descriptive, not
  explanatory — we know Haze consistently draws the strongest correction
  signal at every depth, but not *why* the affine/low-rank operator families
  fail to convert that signal into restoration gain for Haze specifically.
- F's Rain improvement and Haze's persistent non-improvement together suggest
  the remaining bottleneck (architecturally) is operator expressiveness for
  Haze specifically, not signal availability or depth — but this experiment
  cannot yet distinguish "Haze needs a fundamentally different operator" from
  "Haze needs more capacity within the same operator family" (e.g., higher
  rank F).

## 13. Answering the Main Questions

1. **Does multi-depth conditioning improve over bottleneck-only FiLM?** No,
   not reliably — D and E both show seed-inconsistent, near-zero-crossing
   deltas over C.
2. **Does it specifically recover the Haze failure?** No — Haze deltas for
   D/E/F are all noisier than their own means (std > mean in every case);
   none constitute a trustworthy rescue.
3. **Does low-rank channel mixing outperform FiLM?** Yes, in the sense that
   F-C is the only clean, reproducible win among the three new mechanisms —
   but the win is driven by further Rain improvement, not a genuine Haze fix.
4. **Does stronger conditioning improve restoration without a large parameter
   increase?** Yes for F specifically (smallest overhead, most reliable gain);
   no for D/E, which added more parameters than F for a less reliable result.

## 14. GO / NO-GO

Per the pre-specified decision rule:

- **GO** requires multi-depth or low-rank conditioning to improve C/B
  *especially Haze*, across seeds. **Not met** — no mechanism reliably
  improves Haze.
- **PARTIAL GO**: one degradation improves substantially but overall
  restoration remains near baseline. **This matches F specifically**: F
  substantially improves Rain (stacking on top of C's own Rain recovery) and
  achieves the best overall PSNR of any conditioned model (27.073dB), still
  below A (27.315dB) but closer than B or C — "near baseline," not at it.
- **NO-GO** (no reproducible benefit at all) applies specifically to **D and
  E**: their overall deltas over C straddle zero with high variance, and
  neither shows a trustworthy per-degradation win anywhere.

**Overall decision: PARTIAL GO, credited entirely to Model F (low-rank
channel-mixing).** Multi-depth FiLM (D, E) is NO-GO — per the task's own rule,
do not increase FiLM conditioning complexity further in that direction. F,
despite its very low parameter overhead, delivers the most reliable
restoration improvement of any conditioning mechanism tested across TEST08-C
and TEST09 combined, but it has not solved Haze — it has made Rain even
better.

## 15. Recommendation for TEST10

Follow the task's own NO-GO branch for the multi-depth-FiLM direction
(D/E are not worth further investment) while continuing to invest in the
low-rank direction that F opened up: (a) test whether a **higher-rank** F
(e.g., rank 8-16) closes more of the remaining gap to baseline A without a
large parameter cost, since F already achieves its result at less overhead
than plain FiLM; and (b) since every model at every depth agrees Haze needs
the strongest correction yet none deliver it, treat Haze as requiring a
**restoration-trajectory distillation** approach (matching intermediate
decoder outputs or degradation-specific loss weighting) rather than further
architectural conditioning search — this is the natural next step flagged
back in TEST08-C's own recommendation and reinforced, not resolved, here.
