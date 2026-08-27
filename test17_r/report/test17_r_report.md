# TEST17-R — R2R Reproduction and Student Quality Benchmark

## 1. Objective

Determine whether the ~1 dB gap between the current student ("A", 27.31 dB, TEST12/17
protocol) and the reported R2R denoising numbers (34.10/31.45/28.22 dB) reflects a real
architecture/capacity gap, or a protocol mismatch — before making any further
architecture changes. Per the brief's own rule, this experiment does not chase PSNR by
modifying the architecture; it audits the comparison itself.

**Status of this pass: Phases 0-3 complete (recipe, target origin, dataset audit,
evaluation crosscheck). Phases 4-9 (full R2R retraining, matched-protocol NAFNet
retraining, architecture MAC comparison, leakage audit, gap decomposition) are NOT yet
run** — they require ~40+ GPU-hours (R2R's own paper-reported training time) and a
decision on which protocol to standardize on, and are scoped for explicit go-ahead
rather than run silently. See §12/§13.

## 2. Exact R2R Recipe

Full sourced recipe in `R2R_exact_recipe.md` (architecture, loss, optimizer/schedule,
data, evaluation — every row cited to a paper page or a `file:line`). Two items marked
UNKNOWN (NAFBlock's internal normalization/activation choice, training-time random
seed) because they were not directly verified in this pass — not defaulted, not guessed.

**Headline correction**: the brief's "Epoch 15/25/50" is a misread. **15/25/50 are
Gaussian-noise σ levels**, not training epochs. R2R trains one model for a fixed
240-epoch pretrain + 30-epoch finetune schedule, then evaluates that single checkpoint
three times at three noise severities. There is no "epoch-15 checkpoint" — see
`R2R_exact_recipe.md`, "CRITICAL CORRECTION" section, and `target_origin.md`.

## 3. Target Origin

Full trace in `target_origin.md`. Summary: the 34.10/31.45/28.22 numbers come from the
paper's **officially released 3D (denoise+derain+dehaze) all-in-one checkpoint**
(`train_ckpt_3D_f/last.ckpt` + `save_prompts_3D/last`), evaluated on **CBSD68** (68
images) using **R2R's own eval code** (`test_3D.py --mode 0`). This was independently
re-run in this project (on `devon`, RTX 4090, 2026-08-17) and reproduced the paper's
numbers to the hundredth of a dB. The target values are real, verified, and correctly
attributed to a genuine converged model — they are not an inflated or misreported
number. The problem is not that the target is wrong; it's that it is currently being
compared against a differently-sourced student number (see §4).

## 4. Dataset Audit

Full comparison in `dataset_comparison.csv`. Key mismatches, all independently
verified by reading both codebases:

1. **Different source images entirely.** R2R: BSD400+WED (train) / CBSD68 (test),
   standard curated denoising benchmarks. TEST12/TEST07-B: 100 DIV2K validation images,
   split 80/20.
2. **Different test-image scale.** CBSD68 uses full natural images (~321×481).
   TEST12/TEST07-B evaluate on a single fixed 128×128 crop per val scene — a 12x
   smaller test region per image, with far less spatial content for SSIM's local
   windowing to average over.
3. **Different degradation types compared per number.** R2R's 34.10/31.45/28.22 are
   each **pure denoising only**, isolated from derain/dehaze. The student's 27.31 dB is
   a **blended average across Rain+Haze+Noise** in one number
   (`test17/scripts/train.py:194`, `val_df.psnr.mean()` across all three degradation
   rows). Averaging in two much harder, non-denoising degradations will pull the
   number down regardless of underlying model quality.
4. **Rain/Haze are procedurally synthesized, not real.** R2R's derain/dehaze numbers
   (not directly part of the 27.31 comparison, but relevant to the overall claim) use
   real photographed Rain100L/SOTS pairs. TEST07-B's Rain/Haze are synthetic
   (`cv2.line` streaks + Gaussian blur; linear-depth atmospheric-scattering equation) —
   a different, likely easier-in-some-ways/harder-in-others distribution.
5. **Noise synthesis is the closest match** — both use the same core AWGN formula
   (`clip(clean + randn*σ, 0, 255)`), but R2R evaluates at three *fixed* σ per run
   (15/25/50) while TEST07-B randomizes σ ~ Uniform(12,40) per crop. This is the one
   degradation type where a controlled, apples-to-apples comparison is plausible
   without new data collection (see §12 recommendation).
6. **Different model capacity.** R2R: 19.7M params + an external retrieval bank.
   Student "A": 7.37M params (2.7x smaller), no bank. A capacity gap this size would
   independently predict *some* PSNR gap even under identical data/protocol.

**Conclusion: dataset, degradation mix, and model capacity all differ simultaneously.**
This is a triple confound, not a single clean variable.

## 5. Evaluation Audit

Full comparison in `evaluation_crosscheck.md`. Finding: the two metric pipelines agree
on PSNR/SSIM formula, channel axis, color space, and border policy, but **this
project's `psnr_ssim()` quantizes to uint8 before computing the metric; R2R computes
directly on the float `[0,1]` restoration.** This is a real discrepancy, but the
expected magnitude (a few hundredths to ~0.1-0.15 dB) is far too small to explain a
multi-dB gap on its own. It should be controlled for in any Phase 4/5 run, but it is
not the primary suspect.

## 6. R2R Reproduction

**Already done, at the evaluation level, prior to this test.** The officially released
3D checkpoint was re-evaluated on CBSD68 and reproduced 34.10/31.45/28.22 exactly (see
§3). **Full from-scratch retraining (Phase 4 as specified — training R2R's own 270-epoch
schedule from random init) has NOT been run** — the paper itself reports ~40 GPU-hours
for this on an RTX 5090; on this project's RTX 4090 it would likely take longer. This
is a real resource commitment and is being surfaced as a decision point rather than
silently started (§12).

## 7. NAFNet Under Identical Protocol

**Not yet run.** Requires deciding which protocol to standardize the comparison on
(R2R's CBSD68/Rain100L/SOTS, or the project's own DIV2K-80/20/Rain-Haze-Noise, or both)
before a "matched protocol" run is well-defined — see §12.

## 8. Architecture Comparison

| | R2R | Student "A" (locked NAFNet) |
|---|---:|---:|
| Params | 19.7M | 7.37M |
| MACs | 12G @ 224×224 | ~1.03G @ 128×128 |
| Retrieval bank | Yes (external, per-degradation) | No |
| Backbone blocks | NAFBlock, `[1,1,1,28]` enc / `[1,1,1,1]` dec, width 32 | Locked NAFNet (exact block config not re-verified in this pass — see `test12_report.md` §10 for param/MAC source) |

R2R is ~2.7x larger in parameters and has no directly comparable MAC figure yet (input
resolutions differ, 224² vs 128²) — a real capacity difference exists independent of
any protocol issue, and should be treated as a second, architecture-level explanatory
factor alongside the dataset mismatch in §4.

## 9. Quality Gap Decomposition

**Not yet run** — requires the controlled single-factor ablations specified in the
brief's Phase 8, which in turn require the Phase 4/5 training runs. Deferred pending
§12 decision.

## 10. Leakage / Reproducibility Audit

Not run in this pass. Flagging one relevant fact surfaced incidentally: TEST07-B's
80/20 split reuses the *same* 100-image DIV2K pool as TEST06, with scene-disjointness
documented as the leakage-relevant constraint "within this experiment"
(`build_dataset.py` header comment) — i.e. the split's disjointness has already been
asserted by the original experiment design, not newly verified here. No R2R-side
leakage risk applies to the current findings since no R2R training was run.

## 11. Interpretation

The comparison as currently framed ("student 27.31 dB" vs "R2R 34.10/31.45/28.22 dB")
mixes at least three independent variables: **different test data, different
degradation mix (blended 3-task average vs isolated single-task), and a smaller student
model with no retrieval bank.** None of the evidence gathered in this pass suggests the
raw ~1 dB-ish framing in the brief is even the right comparison to make — the *actual*
gap, once like is compared with like, is unmeasured. This lands closer to the brief's
own **Case A/C territory** (protocol mismatch must be resolved before the numbers mean
anything) rather than Case B (a clean architecture-driven gap) — but this cannot be
stated with full confidence without either (a) evaluating the actual R2R checkpoint on
the student's own Noise-only val images (cheap, ~minutes, no training), or (b) full
retraining under a unified protocol (expensive, ~40+ GPU-hours, requires new data
acquisition for a fair Rain/Haze comparison).

## 12. GO / NO-GO

**NO-GO on proceeding to architecture changes (N+F2 successors, frequency ops, etc.)
until the comparison is resolved — this confirms the brief's own premise.**

**NO-GO on immediately running full Phase 4/5 retraining** — not because it's
unwarranted, but because a much cheaper diagnostic should run first and may make part
of it unnecessary or reshape its scope:

**Recommended immediate next step (cheap, ~10-20 minutes GPU time, no training):**
Run the already-downloaded, already-verified R2R 3D checkpoint through **this
project's own Noise-only val images** (20 fixed 128×128 crops,
`~/teacher-experiments/test07_b/data/val/noise/` + `val/clean/`), using R2R's own
inference code, and score the result with **both** metric pipelines (float and uint8)
to directly quantify the evaluation-quantization confound. This isolates the one
degradation type where R2R's and the project's synthesis functions are closest in
spirit (both plain AWGN), removes the "blended 3-task average" confound (Noise only,
matching what a controlled comparison needs), and requires zero new training. It
directly tests: does R2R's real, verified-strong model score close to 27-28 dB (Case
C — protocol/data explains the gap) or does it still score in the low-to-mid 30s dB
even on the project's own tiny, harder-cropped images (Case B — real architecture
gap)? That result should determine whether Phase 4/5's much larger investment is worth
making, and in which direction (unify on R2R's protocol, or bring R2R's architecture
onto the project's protocol).

## 13. Recommended Next Experiment

1. **First** (cheap, no training): the diagnostic in §12 — R2R's real checkpoint,
   scored on the project's own Noise val set, both metric pipelines. Est. 10-20 min.
2. **Then, branching on that result**:
   - If R2R scores near the student's range on the project's own data → the gap is
     data/protocol-driven; recommend NOT running the expensive Phase 4/5 full
     retraining, and instead investing effort in making the project's own eval
     protocol match a recognized benchmark (or explicitly documenting why it
     shouldn't) — Case A/C resolution without 80 GPU-hours spent.
   - If R2R still scores dramatically higher even on the project's own harder data →
     proceeds to Phase 5 (train NAFNet-A under R2R's *exact* protocol on CBSD68/etc,
     much cheaper than full R2R retraining since only one side needs retraining) before
     committing to Phase 4 (full R2R retraining from scratch).

## Mandatory Final Table

| Model | σ / condition | PSNR | SSIM |
|---|---:|---:|---:|
| R2R reported (paper) | σ=15 | 34.10 | 0.9356 |
| R2R reproduced (this pass, official ckpt, CBSD68) | σ=15 | 34.10 | 0.9356 |
| NAFNet (student "A", TEST12 protocol, blended) | — | 27.31 (blended, not σ=15-comparable) | 0.815 |
| R2R reported (paper) | σ=25 | 31.45 | 0.8931 |
| R2R reproduced (this pass, official ckpt, CBSD68) | σ=25 | 31.45 | 0.8931 |
| NAFNet (student "A") | — | *(same 27.31, not σ=25-comparable — see §11)* | 0.815 |
| R2R reported (paper) | σ=50 | 28.22 | 0.8064 |
| R2R reproduced (this pass, official ckpt, CBSD68) | σ=50 | 28.22 | 0.8064 |
| NAFNet (student "A") | — | *(same 27.31, not σ=50-comparable — see §11)* | 0.815 |
| R2R on project's own Noise-only val set (proposed, §12) | — | **NOT YET RUN** | **NOT YET RUN** |
