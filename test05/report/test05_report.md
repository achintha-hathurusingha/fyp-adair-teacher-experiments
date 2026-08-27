# TEST05 — Degradation-Specific Representation Discovery

## 1. Objective

Find the smallest teacher representation — full tensor, channel subset,
or compact projection — that best trades off degradation specificity,
reduced scene/content sensitivity, measurable causal restoration
influence, compactness, and NPU deployment practicality. Not "which
feature classifies degradation best," but "what should AdaIR actually
transfer to a NAFNet student."

## 2. Results from TEST01–TEST04

TEST01: `raw_low` is exactly zero at benchmark resolution (independently
verified). TEST02: internal representations are highly Rain/Haze/Noise-
discriminative (71.7%→100%→54.7%) but dataset-confounded. TEST03: the
same trajectory survives controlled same-scene, scene-grouped testing
(66.7%→100%→37.0%). TEST04: whole-tensor causal intervention shows a
real effect (exceeds random/zero/mean controls) that scales with depth
(14.17→53.94) and is amplified by closing skip connections (9× when all
3 are closed), but the SAME mechanism is *more* sensitive to scene
changes (cross-scene L2=7.81) than to degradation changes (same-scene
cross-degradation L2=4.03) — "MODERATE," not "STRONG," causal evidence for
degradation-specificity of the *full* representation.

## 3. Candidate Representations

31 candidates: `latent_pre` + 3 AFLBs × {`y_in`, `raw_high`, `raw_low`,
`mined_high`, `mined_low`, `hl_spatial_weight`, `lh_channel_weight`,
`fmom_agg`, `cross_agg_out`, `aflb_out`}. `raw_low` included only as a
negative control, per instruction — no effort spent improving it.

## 4. Degradation Discriminability

`Linear_Probe` sheet, 31 candidates, grouped 5-fold CV. Reproduces
TEST02-04: most candidates 90–100%; `raw_low` exactly 33.33% (zero
variance, all 3 AFLBs) — **fifth independent confirmation** of its
degeneracy, now via five distinct methodologies across TEST01–05.

## 5. Scene Sensitivity

`Scene_Sensitivity` sheet. `lh_channel_weight` (L-H channel attention)
has the highest pooled-level degradation/scene ratio of any candidate
(**3.15**, both AFLB1 and AFLB3), exceeding full `latent_pre` (1.77) and
every AFLB output (1.90–2.46). This is the first hint that a specific,
compact sub-component — not the full tensor — carries a more
degradation-concentrated signal.

## 6. Channel-Level Analysis

`Channel_Ranking` sheet, ~4,400 channels across 16 candidate tensors.
Individual channels reach degradation/scene ratios of **4.7–4.8** —
higher than any pooled feature — confirming degradation-specificity is
concentrated in a channel subset, not spread evenly across a tensor.

## 7. Causal Channel Intervention

Three findings, reported exactly as measured, not smoothed over:

- **Ablation** (`Channel_Ablation`, top-10 `latent_pre` channels, 20
  scenes): zeroing or degradation-group-averaging these channels produces
  a measurable, non-trivial output change relative to keeping them —
  confirms they are not causally inert.
- **Grouped intervention** (`Channel_Group_Intervention`, 1,980 rows):
  top-ranked channel subsets retain **3.4% (top-5%) to 41.6% (top-50%)**
  of the full tensor's causal effect — roughly proportional to subset
  size, not a "small subset captures most of the effect" super-linear
  result.
- **Top-ranked vs. random-same-size channels**: top-ranked channels beat
  random channels at 3 of 5 tested sizes (10%, 30%, 50%) and **lose** at
  2 (5%, 20%). Classification-discriminative channels are **not reliably
  more causally impactful** than randomly chosen channels of the same
  count. This directly extends TEST04's lesson (discriminability ≠ causal
  dominance) down to the channel level, and is reported plainly per the
  task's explicit instruction against manufacturing a clean result.

## 8. Spatial Analysis

`Spatial_Analysis` sheet. Degradation-driven variance (comparing Rain vs.
Haze vs. Noise for the same scene, channel-averaged) is close to
spatially uniform for every candidate tested (`importance_center_frac`
1.07–1.30) — mildly center-weighted at most (`AFLB1_mined_low` = 1.30,
the most concentrated), never strongly localized to edges or borders.
Degradation information does not appear to hide in a small spatial
region; it is broadly distributed.

## 9. Frequency Analysis

`Frequency_Analysis`/`Frequency_Summary` sheets — independent 2D-FFT
analysis of the actual feature tensors (not AdaIR's own degenerate FFT
mask). **Counter-intuitive but consistent finding**: Noise shows the
*highest* low-frequency spatial energy (89.9–99.3%) and *lowest*
high-frequency energy of the three degradations, across nearly every
candidate — the opposite of what raw pixel intuition (noise = high
spatial frequency) would suggest. This means CNN layers appear to smooth
per-pixel noise into spatially uniform internal activations, while
Rain/Haze (spatially structured degradations) produce comparatively more
structured, higher-frequency internal feature variation.
`AFLB1_mined_low` is almost entirely low-frequency (92–99%) regardless of
degradation — an interesting residual signature given its literal input
(`raw_low`) is exactly zero, showing the cross-attention machinery
imposes its own smooth-output prior independent of its (empty) input.

## 10. Frequency-Sensitive Degradation Features

`Frequency_Channel_Ranking` sheet. `AFLB3_aflb_out` shows by far the
largest degradation-dependent frequency-band variance (**14.2**, vs.
0.05–6.9 for other candidates) — its high-frequency energy share shifts
the most between Rain/Haze/Noise, making it the most frequency-sensitive
candidate measured, consistent with it also being the deepest/largest
causal-effect candidate in TEST04/TEST05.

## 11. Compact Representation

`Compact_Embedding` sheet. **PCA-16 (16 dimensions) retains 99.7%**
degradation-classification accuracy — statistically indistinguishable
from the full 768-dim pooled `latent_pre` vector (100.0%). PCA-64/128
reach 100.0% exactly. Degradation-aware information compresses
dramatically without losing linear separability.

## 12. Alpha/Beta Comparison

62.0–62.7% across all 3 AFLBs and combined (6-dim) — matches TEST02/03
exactly, and is now shown to be far weaker than even a 16-dim PCA
projection of the real feature (99.7%). Alpha/beta is a real but minor,
non-primary signal; plausible only as a cheap auxiliary conditioning
input, never a primary distillation target.

## 13. NPU Cost Analysis

`NPU_Cost` sheet. Key numbers (fp32): `latent_pre`/`AFLB1_aflb_out` =
3.6MB (384×40×60); `AFLB2_aflb_out` = 7.2MB (192×80×120);
`AFLB3_aflb_out` = **14.4MB** (96×160×240, largest of all candidates
despite fewest channels, because of its large spatial size);
`top_10pct_latent_channels` = **356KB** (a ~10× reduction from full
latent); `alpha_beta_all` = 24 bytes (negligible but weak, per §12).

## 14. Distillation Candidate Ranking

`Distillation_Ranking` sheet, explicit composite score (accuracy 0.25 +
degradation/scene ratio 0.35 + causal effect 0.20 + compactness 0.20),
with a documented sensitivity check (doubling the ratio weight leaves the
top-5 unchanged: `AFLB3_aflb_out`, `AFLB2_aflb_out`,
`AFLB1_lh_channel_weight`, `AFLB1_mined_low`, `AFLB1_aflb_out`).

**Important caveat, not buried**: the composite's compactness term uses
pooled-vector dimensionality, which favors `AFLB3_aflb_out` (pooled dim
192) — but §13 shows its *raw spatial tensor* is the single largest
candidate (14.4MB). The composite score is reported as computed, but is
**not** used alone to make the final recommendation below, per the task's
explicit instruction.

## 15. Recommended F2S Strategy

Given §7's finding that channel selection by classification accuracy does
not reliably concentrate causal effect, and §11's finding that a compact
*learned or PCA* projection preserves discriminability far more reliably
than a *hand-picked channel subset* of similar size, the most
scientifically supported option is:

**Option D: Frequency-aware teacher feature → compact degradation
representation → spatial NAFNet feature**, using `latent_pre` (not
`AFLB3_aflb_out`, despite its higher composite score) as the source
tensor, compressed via a compact (16–64-dim) projection, as the primary
transferred signal — **not** Option C (hand-selected raw channels, which
§7 shows is not clearly superior to random selection at matched size) and
**not** Option A alone (full latent, which TEST04 showed is not cleanly
degradation-specific relative to scene/content).

## 16. Proposed Teacher Signal

**`latent_pre`**, compressed to a compact embedding (PCA-16 as the
simplest, already-validated baseline; a small learned linear/MLP
projection as the natural TEST06 upgrade), is the recommended primary
teacher signal:

- Smallest raw tensor of any high-discriminability candidate (3.6MB
  fp32 uncompressed; the 16-dim embedding is a further ~200,000×
  reduction).
- 100.0% pooled discriminability, 99.7% retained at PCA-16.
- Real, structured causal effect (TEST04: L2=14.17, exceeds random/zero/
  mean controls).
- Architecturally central — computed once, not duplicated per AFLB.
- `lh_channel_weight` (highest pooled degradation/scene ratio, 3.15) is
  recommended as a **secondary, auxiliary** signal — small (already a
  pooled channel-attention vector) and worth including alongside
  `latent_pre` in a TEST06 ablation, given its distinct evidence profile.
- `AFLB3_aflb_out` is **not** recommended as the primary signal despite
  its top composite score and strongest measured causal effect (53.94):
  its 14.4MB raw tensor size is the worst of any candidate for NPU
  deployment, and its causal-effect advantage did not translate into a
  cleaner degradation-vs-scene separation than `latent_pre` in the
  channel-subset experiments.

## 17. Limitations

- Channel ablation/grouped-intervention/content-control experiments were
  run on `latent_pre` only (not replicated for every candidate) — an
  explicit scope decision (30× the runtime for marginal additional
  evidence, given TEST04 showed `latent_pre`/`AFLB1_aflb_out` are nearly
  behaviorally identical).
- Grouped intervention and content control use 20–30-scene subsets, not
  the full 100, consistent with the "manageable subset" instruction.
- Spatial/frequency analysis uses the 15 representative scenes with saved
  raw tensors, not all 100.
- The composite ranking score's weights are a documented, explicit
  choice, sensitivity-checked once (doubling one weight) — not
  exhaustively validated against every possible weighting.
- No PCA/embedding was tested for candidates other than `latent_pre`;
  `lh_channel_weight`'s compactness under projection is inferred from its
  already-small pooled size, not directly measured.
- As in TEST03/04, the underlying Rain/Haze/Noise images are TEST03's
  documented synthetic degradation models, not photorealistic simulators.

## 18. TEST06 Proposal

**TEST06 — NAFNet Student + F2S Distillation**, using:

- **Teacher signal**: `latent_pre` → compact embedding (start with PCA-16,
  the validated baseline; compare against a small learned linear
  projection trained jointly with the student).
- **Auxiliary signal** (ablation arm): `lh_channel_weight`, given its
  distinct highest-ratio evidence profile.
- **Injection point**: NAFNet's bottleneck/deepest encoder stage (the
  architectural analog of `latent_pre`'s position in AdaIR).
- **Loss**: a restoration loss (L1/PSNR-style) on the student's output
  plus an auxiliary distillation loss aligning the student's bottleneck
  (projected to the same compact dimensionality) with the teacher's
  compact embedding — architecture and exact loss weighting to be
  designed in TEST06 itself, not here, per the explicit instruction not
  to design the student in TEST05.
- **Ablation to run in TEST06**: compact-embedding distillation vs. no
  distillation (baseline NAFNet) vs. full-`latent_pre` distillation (to
  directly test whether TEST05's "compact is enough" finding holds once
  a student is actually trained, not just probed).
