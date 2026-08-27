# TEST02 — Degradation Representation Analysis of AdaIR

## 1. Research Question

Where does the released 3-degradation AdaIR checkpoint encode
degradation-specific (Rain/Haze/Noise) information, and can that
information be identified from intermediate representations using only an
external, non-intrusive linear probe? This follows directly from TEST01's
finding that the explicit FFT-based frequency mask is degenerate at
benchmark resolution and barely affects restoration quality — if the
binary mask isn't the mechanism, something else must let one shared
checkpoint behave differently for rain, haze, and noise.

## 2. Hypothesis

AdaIR performs blind restoration (no explicit degradation label is ever
supplied — confirmed in section 4). We hypothesize that degradation
information is nonetheless implicitly encoded somewhere in the network's
intermediate activations, is linearly decodable at least in part, and that
its strength varies systematically with depth — plausibly building up
through the encoder (where the network needs to recognize what it's
looking at) and being at least partially discarded near the output (where
the goal is a degradation-agnostic clean image).

## 3. AdaIR Architecture Relevant to This Experiment

4-level Transformer encoder-decoder. Shallow feature extraction
(`OverlapPatchEmbed`) → 3 encoder levels (increasing depth, decreasing
spatial resolution, 48→96→192 channels) → latent bottleneck (384 ch,
H/8×W/8) → 3 decoder levels interleaved with **AFLB** (Adaptive Frequency
Learning Block, `FreModule`) instances → refinement → output conv with
global residual. No separate rain/haze/noise branches anywhere — a single
shared set of weights processes every image identically regardless of
degradation type. Full detail: `report/source_audit.md`.

## 4. Source-Code Audit

Full audit in `report/source_audit.md`. Headline finding: `AdaIR.forward(self,
inp_img, noise_emb=None)` — `noise_emb` is the *only* parameter besides the
image, appears exactly **once** in the entire 475-line source file (its own
signature), and is never read or used anywhere in the forward pass.
**No explicit degradation label, embedding, classifier, prompt, or routing
mechanism exists anywhere in the released implementation.** This was
verified by direct source inspection (`grep -n noise_emb net/model.py`),
not inferred from the paper. Any degradation-type information the network
uses must therefore be implicitly derived from the image content itself —
exactly the premise this experiment tests.

## 5. Experimental Setup

- Model: released, **unmodified** AdaIR (`decoder=True`), `adair3d.ckpt`
  (3-degradation teacher), 28,784,824 params, loaded via the same strict
  loader used throughout this project (0 missing/0 unexpected keys). No
  retraining, no architecture change, no preprocessing change — the only
  addition is non-intrusive forward hooks, reusing test01's
  `instrument.py` unmodified.
- Dataset: identical 300-image manifest to test01 (100 Rain100L / 100
  SOTS-outdoor / 100 BSD68). Labels (Rain=0/Haze=1/Noise=2) used **only**
  for post-hoc external analysis, never fed into AdaIR.
- Host: devon (RTX 4090). Same known-flaky-cores mitigation as test01/test02
  (`taskset -c 0-7,12-31` on every invocation; CSV is the source of truth,
  `.xlsx` rendered on a separate machine).
- Determinism: global seed 0 + per-image deterministic noise seeding
  (carried over from the bug fix discovered in test01).
- Full environment record: `results/environment.txt` / `Environment` sheet.

## 6. Feature Extraction Locations

41 representations per image, all captured via forward hooks or the
monkey-patched (non-invasive) AFLB capture points reused from test01 —
**no source file was edited**:

- **11 global pipeline stages**: `input` (raw degraded pixels), `shallow_Y0`
  (`patch_embed` output), `encoder_level1/2/3`, `latent`, `decoder_level3/2/1`,
  `refinement`, `output` (final restored image).
- **3 AFLBs × 10 internal sub-features each**: `y_in` (AFLB's input feature),
  `raw_high`/`raw_low` (pre-cross-attention FMiM split), `mined_high`/`mined_low`
  (post-cross-attention FMiM output), `hl_spatial_weight` (H-L unit),
  `lh_channel_weight` (L-H unit), `fmom_agg` (FMoM output), `cross_agg_out`
  (final CA merge), `aflb_out` (AFLB residual output).

Every tensor is pooled to a compact GAP+GMP-concatenated vector (2×channel-count
dimensions) for the 300-image classifier/distance analysis; full-precision
raw tensors (float16 `.pt`) are additionally kept for 15 representative
images (5/degradation) for deeper inspection — see `Feature_Index`/`Tensor_Index`
sheets.

## 7. Linear Probe Results

Primary quantitative evidence (`Linear_Probe` sheet, `results/classifiers/linear_probe_results.csv`,
82 rows = 41 features × 2 classifiers). Logistic Regression and Linear SVM
agree closely throughout (both external to AdaIR). Full pipeline stages
range from 71.7% (input) to 100.0% (latent), collapsing to 54.7% at output
— detailed in section 9. AFLB-internal sub-features range from **exactly
33.33% (raw_low, all 3 AFLBs, zero variance)** to 100.0% (`AFLB1_y_in`) —
detailed in section 11.

## 8. Degradation Separation Results

`Distance_Analysis` sheet (`results/statistics/degradation_separation.csv`).
Separation ratio = mean inter-class distance / mean intra-class distance
(Euclidean, standardized features); larger = more separated. Top-ranked:
`AFLB3_lh_channel_weight` (2.26), `AFLB1_mined_low` (2.22), `AFLB3_mined_low`
(2.01). `raw_low` features could not be scored (cosine distance undefined
for exact-zero vectors — itself confirmatory of the zero-tensor finding).
This distance-based ranking broadly agrees with the linear-probe ranking
but is not identical (distance and linear separability are related but
distinct notions) — both are reported per the task's requirement not to
rely on a single method.

## 9. Feature Trajectory

**The headline result of this experiment** (`Feature_Trajectory` sheet,
`results/visualizations/degradation_information_trajectory.png`):

| Stage | Accuracy (5-fold CV, logreg) |
|---|---:|
| Input | 71.7% ± 3.8 |
| Shallow (Y0) | 93.3% ± 1.5 |
| Encoder L1 | 98.0% ± 2.4 |
| Encoder L2 | 99.3% ± 0.8 |
| Encoder L3 | 99.3% ± 1.3 |
| **Latent** | **100.0% ± 0.0** |
| AFLB 1 | 99.7% ± 0.7 |
| Decoder L3 (post-AFLB1) | 99.0% ± 1.3 |
| AFLB 2 | 99.0% ± 1.3 |
| Decoder L2 (post-AFLB2) | 99.7% ± 0.7 |
| AFLB 3 | 99.7% ± 0.7 |
| Decoder L1 (post-AFLB3) | 99.7% ± 0.7 |
| Refinement | 99.7% ± 0.7 |
| **Output (restored)** | **54.7% ± 4.5** |

**OBSERVATION**: degradation information rises sharply through the encoder,
peaks at exactly 100% by the latent bottleneck, stays at or near ceiling
(99-100%) through every AFLB and decoder stage, then drops to 54.7% at the
final restored output — barely above the 71.7% achievable from raw input
pixels alone, and far below every internal stage.

**INFERENCE**: AdaIR's internal representations carry near-perfect
degradation-discriminative information from the encoder onward; the final
output conv + residual substantially (not completely) removes it.

**HYPOTHESIS**: this pattern is consistent with AdaIR using strong internal
degradation-awareness to condition how it restores an image, while
converging toward a degradation-agnostic "clean image" manifold at the
output — the behavior one would want from a good blind restoration network.
We do not claim this conditioning is causal or mechanistic; we observe the
correlational pattern a linear probe reveals.

## 10. Alpha/Beta Analysis

`Alpha_Beta_Probe` sheet. `[alpha, beta]` alone (2 raw scalars per AFLB) —
**not** treated as a degradation label, tested empirically:

| Input | Accuracy | vs. random (33.3%) |
|---|---:|---:|
| AFLB1 [α,β] | 64.3% | +31.0 |
| AFLB2 [α,β] | 65.0% | +31.7 |
| AFLB3 [α,β] | 64.7% | +31.3 |
| All 3 AFLBs' [α,β] combined (6-dim) | 66.0% | +32.7 |

**OBSERVATION**: substantially above random, but far below the 93-100%
every full feature representation achieves. **INFERENCE**: alpha/beta carry
*some* real, non-trivial degradation signal (not negligible — over 30
points above chance) but are nowhere near sufficient alone to explain how
AdaIR distinguishes degradations; the classification-relevant information
must live predominantly in the full-channel feature tensors, not in this
2-number bottleneck. Per the task's instruction, this is reported without
overinterpreting: alpha/beta are informative but clearly not the primary
mechanism.

## 11. AFLB Analysis

`AFLB_Analysis` sheet — ranks every AFLB-internal sub-feature by
degradation classification accuracy (30 rows = 10 sub-features × 3 AFLBs):

| sub-feature (avg. across AFLB1-3) | mean accuracy |
|---|---:|
| y_in | 99.6% |
| aflb_out | 99.4% |
| cross_agg_out | 99.0% |
| mined_high | 97.4% |
| fmom_agg | 97.1% |
| mined_low | 96.8% |
| lh_channel_weight | 94.1% |
| raw_high | 89.9% |
| hl_spatial_weight | 65.9% |
| **raw_low** | **33.3% (exactly, zero variance, all 3 AFLBs)** |

**raw_low scoring exactly at the random baseline with zero variance is a
clean, independent corroboration of test01's finding** that this tensor is
a constant zero vector at every AFLB, every image, at benchmark resolution
— two entirely different analysis methods (a from-scratch forward-pass
trace in test01; a 300-image linear-probe sweep in test02) agree exactly.
Every *other* AFLB-internal representation carries strong-to-near-perfect
degradation signal, most strikingly `mined_low`/`mined_high` (the
post-cross-attention FMiM outputs) at 96-97% despite `raw_low` itself
being uninformative — confirming (as test01's report already inferred)
that the cross-attention machinery downstream of the degenerate FFT split
is where the useful signal is actually produced, not the frequency split
itself. `hl_spatial_weight` is the weakest "real" signal, plausibly because
it is the lowest-dimensional feature probed (a single-channel spatial gate,
pooled to only 2 dimensions).

## 12. PCA/t-SNE/UMAP

`results/visualizations/pca/pca_all_stages.png`,
`.../tsne/tsne_key_stages.png`, `.../umap/umap_key_stages.png`. Qualitative
only, per the task's instruction — not primary evidence. Visually
consistent with the linear-probe trajectory: PCA/t-SNE/UMAP show three
increasingly tight, well-separated clusters from the encoder through the
AFLBs, and a visibly less-separated (though not fully overlapping) cluster
structure at the output.

## 13. Relationship to PSNR/SSIM

`PSNR_SSIM_Correlation` sheet, exploratory only (task-mandated: no
causation claimed). Strongest correlations (|Pearson r| > 0.8) are between
encoder-level feature-energy statistics and SSIM (e.g. `encoder_level1`
energy vs. SSIM, r=-0.88). **This is very likely confounded by degradation
type itself** rather than a direct mechanistic link: SSIM is systematically
lower for Noise than for Rain/Haze in this dataset (see test01's
`Output_Metrics`), and encoder features are (per section 9) themselves
near-perfectly degradation-discriminative — so a correlation between
"encoder feature energy" and "SSIM" is largely re-discovering "these are
different degradation types with different typical SSIM," not evidence
that the feature statistic *causes* or *predicts* quality within a fixed
degradation type. Reported as exploratory per the task's instructions, not
as a distillation-relevant finding on its own.

## 14. Interpretation

Distinguishing the three levels the task requires:

- **OBSERVATION**: a linear classifier external to AdaIR can distinguish
  Rain/Haze/Noise from GAP+GMP-pooled intermediate features with >99%
  accuracy from Encoder L2 through Refinement, but only 54.7% from the
  final restored output, and 71.7% from the raw input.
- **INFERENCE**: AdaIR's internal representations contain
  near-complete degradation-discriminative information; this information
  is substantially reduced (not eliminated) by the time the image is
  reconstructed.
- **HYPOTHESIS**: AdaIR internally maintains a strong, implicit sense of
  "what kind of degradation is this" throughout its encoder/decoder/AFLB
  pipeline, uses it to condition restoration, and only lets go of most of
  that signal at the very last step. This is plausible and consistent with
  the observations, but this experiment does not manipulate or ablate
  anything to prove the internal representations are *causally used for*
  conditioning (as opposed to being an incidental byproduct of processing
  visually distinct inputs) — that would require an intervention study
  (e.g. representation swapping/steering), explicitly out of scope here.

## 15. Implications for Knowledge Distillation

- The **latent representation** (100.0% accuracy, single bottleneck,
  384 channels, smallest spatial size of any pipeline stage) is the single
  best combination of degradation-separability, compact dimensionality,
  and computational accessibility (it's computed once, centrally, not
  duplicated across 3 AFLBs).
- **AFLB outputs (`aflb_out`, 99.0-99.7%) and `y_in`/`cross_agg_out`
  (99.0-100%)** are close seconds and directly reusable if the student
  needs degradation-awareness at multiple decoder depths, not just once.
- **`mined_low`/`mined_high` (96-97%) matter more than the AFLB's nominal
  "frequency" framing suggests** — they carry strong signal despite
  `raw_low` (the literal frequency-split input to them) being exactly zero,
  confirming test01's inference that FMiM's cross-attention machinery,
  not the frequency split, is the meaningful computation.
- **Do not distill from the final output alone** if the goal is to
  transfer degradation-adaptive *behavior* — the output has already
  discarded most of the linearly-decodable degradation signal; a student
  trained only to match final pixels would not necessarily learn *how*
  the teacher adapted internally.
- **`raw_low` (and by extension the explicit FFT/mask path)**: confirmed,
  independently, as a non-target — carries zero information by two
  different methods now (test01's direct trace, test02's linear probe).

## 16. Limitations

- **Degradation type is confounded with dataset-of-origin.** Rain images
  come only from Rain100L, Haze only from SOTS-outdoor, Noise only from
  BSD68 (synthetically noised) — the same 300-image set used throughout
  this project. A classifier trained on this set cannot fully distinguish
  "this network encodes rain-ness" from "this network encodes
  Rain100L-scene-content-ness." The strong 71.7% input-only baseline
  already shows substantial separability exists in raw pixels before any
  AdaIR computation — some of that is genuinely the degradation
  (rain streaks, haze color cast, noise texture look visually distinct)
  and some may be scene/domain differences between the three source
  datasets. This is a limitation inherited from the original 300-image
  benchmark design, not introduced by TEST02, but it tempers every
  accuracy number in this report: **high probe accuracy is evidence of
  "Rain100L/SOTS/BSD68-discriminative information," which very plausibly
  includes but is not proven to be purely degradation-type information.**
  A cleaner follow-up would apply multiple synthetic degradation types to
  the *same* underlying clean images (as BSD68's noise already partially
  does) to fully decouple degradation from scene content.
- Deep raw-tensor inspection covers only 15/300 images; the quantitative
  claims (probe accuracy, distances) use all 300.
- Linear probes measure *linear* separability only, by design (per the
  task's explicit instruction) — a nonlinear probe could reveal additional
  structure in stages that scored lower here (e.g. `output` at 54.7%);
  this was intentionally not tested, since the task's purpose was to
  measure the simplest, most distillation-relevant notion of
  "discoverable" information.
- t-SNE/UMAP are qualitative and were not used as primary evidence, per
  the task's instruction.
- PSNR/SSIM correlations are exploratory and likely confounded by
  degradation type itself (section 13) — not usable as a standalone
  distillation signal without further decomposition.

## 17. Recommended Next Experiment

Two candidates:

1. **Decouple degradation from dataset domain**: apply all three
   degradations (or at least two) synthetically to a shared clean-image
   pool (as already done for BSD68/Noise), then rerun the exact same
   linear-probe pipeline. If the trajectory shape (rise → peak at latent →
   collapse at output) survives with true within-dataset control, the
   "implicit degradation conditioning" inference in section 14 becomes far
   stronger evidence, not just plausible.
2. **Intervention study**: swap or steer the latent/AFLB representation of
   one degradation type into the decoder of another and observe whether
   restoration *behavior* changes accordingly — this would move the
   claim from "the information is linearly present" (this report) to
   "the information is causally used" (the open question section 14
   explicitly leaves unresolved).
