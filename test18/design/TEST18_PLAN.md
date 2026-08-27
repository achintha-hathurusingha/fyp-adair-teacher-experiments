# TEST18 — AdaIR Component Ablation (Paper-Style Retraining) + Frequency-Domain Diagnostics

## 0. Why this test, and how it differs from TEST01-06R

TEST01 through TEST06-R established, with increasing statistical rigor,
that AdaIR's frequency mechanism is not causally load-bearing for
restoration quality **on the frozen, released checkpoint, at inference
time**. That is an important but narrower claim than "the frequency
mechanism doesn't matter architecturally" — it says nothing about what
happens if the mechanism is actually removed *before* training, the way
the paper's own Table 7 does it.

This was directly challenged (Himeth's counter-argument: the frequency
modules sit in the main forward path, execute every inference call, and
are literally the mechanism the model is named after). The resolution
isn't to re-argue from the architecture diagram — it's to test the claim
the diagram actually supports: **does removing/degrading each component
*before training* change what AdaIR learns to do**, matching the paper's
own methodology (Table 7) rather than TEST01-06R's frozen-checkpoint
interventions.

**Explicit scope decision (per user instruction, this session)**: unlike
TEST01-06R, this test DOES retrain AdaIR from scratch, multiple times,
one per ablation variant — a materially bigger undertaking than anything
else in this project (previous student-training tests trained a ~7.4M-
param NAFNet-M; AdaIR is ~28.8M params, a full Transformer restoration
network). Per user instruction, training uses the **3-in-1 degradation
setting** (dehaze + derain + denoise, matching the paper's Table 1
protocol) rather than the paper's Table 7 ablation protocol (dehaze-only,
20 epochs) — this is a deliberate scope choice: fewer, more thorough
variants on the actual all-in-one setting this project cares about,
rather than a single-task minimal reproduction.

## 1. What the paper's own ablation (§4.4, Table 7) tests

Read directly from `knowledge/literature/01_AdaIR_2403.14614.pdf`,
pages 12-14 (§4.4 "Ablation Studies"):

| Row | FMiM (Fixed) | FMiM (MGB) | FMoM (L-H) | FMoM (H-L) | Paper's PSNR (dehaze-only, 20ep) |
|---|---|---|---|---|---:|
| (a) Baseline | | | | | 28.21 |
| (b) +FMiM, fixed mask | check | | | | 29.79 |
| (c) +FMiM, learned MGB | | check | | | 30.37 |
| (d) + L-H | | check | check | | 30.52 |
| (e) + L-H + H-L (= full AdaIR) | | check | check | check | 31.24 |

Also: Table 8 compares mask strategies (average-pool / Gaussian-filter /
their learned mask); Table 9 compares mining from the raw degraded image
vs. from intermediate features (30.52dB → 29.29dB, i.e. raw image wins).

## 2. Architecture ablation points — read from `net/model.py`

`FreModule` (the AFLB, `net/model.py:289-366`) breaks into exactly the
pieces Table 7 tests:

- `self.fft()` (lines 337-366): computes the high/low frequency split.
  The mask comes from `rate_conv`+`threshold` (learned, MGB) — this is
  the piece to replace with a **fixed square mask** for variant (b).
- `channel_cross_l` / `channel_cross_h` (lines 319-320): FMiM's
  cross-attention mining of high/low against the current feature `y`.
- `self.frequency_refine` (`FreRefine` class, lines 269-285): this IS
  FMoM. Its `SpatialGate` (uses `high` to weight `low`) is the **H-L**
  unit; its `ChannelGate` (uses `low` to weight `high`) is the **L-H**
  unit.
- `channel_cross_agg` + `out * para1 + y * para2` (lines 322-325): final
  merge back into the decoder feature, residual-scaled
  (`para1` init 0, `para2` init 1 — AFLB starts as an identity function
  and learns to deviate).

**Implementation plan**: build `AblatableFreModule` (subclass/reimplementation
of `FreModule`, in `test18/scripts/ablatable_model.py`, read-only importing
the rest of `net/model.py` — AdaIR source untouched) with 3 flags:
`mask_mode ∈ {None, "fixed", "learned"}`, `use_lh: bool`, `use_hl: bool`.

- `mask_mode=None` → `forward()` returns `y` unchanged (= paper's
  "Baseline", no AFLB at all).
- `mask_mode="fixed"` → paper says "Fixed uses a fixed square mask with
  sides of 10" (Table 7 caption) — reproduced exactly: a static
  `h_=w_=10` box, replacing the learned `rate_conv`/`threshold` step.
- `use_lh=False, use_hl=False` (i.e. mask on, FMoM off) → **documented
  design decision, not specified by the paper**: `frequency_refine` is
  skipped and `low_feature + high_feature` is summed directly before
  `channel_cross_agg`, since the paper's Table 7 doesn't specify the
  exact fallback aggregation when both FMoM gates are absent. Flagged
  explicitly in the report as an interpretive choice.
- `use_lh=True, use_hl=False` → only `ChannelGate` (L-H) runs inside a
  reduced `FreRefine`.
- `use_lh=True, use_hl=True` → full `FreRefine`, i.e. released AdaIR.

Five variants, matching Table 7 exactly in structure:

| Variant | mask_mode | use_lh | use_hl |
|---|---|---|---|
| A_baseline | None | — | — |
| B_fixed_mask | fixed | False | False |
| C_learned_mask | learned | False | False |
| D_plus_lh | learned | True | False |
| E_full | learned | True | True (== released architecture, retrained from scratch) |

E_full is also our pipeline sanity check: if our from-scratch
reproduction doesn't land near the released checkpoint's known
performance under the same protocol, that's a red flag to investigate
before trusting A-D's numbers.

## 3. Data — real, matching the paper where available; documented
## substitution where not

Devon inventory check (this session):

| Degradation | Paper's source | What's actually available on devon | Decision |
|---|---|---|---|
| Dehaze | RESIDE OTS (~72k train pairs), SOTS (500 test) | `~/FYP/reside_scratch/nested/OTS_BETA/{haze,clear}` (12GB, real OTS haze/clear pairs) for train; `~/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/{input,target}` (500/492) for test | **Real data used as-is.** |
| Derain | Rain100L (RainTrainL, 200 pairs) | `~/FYP/Workspace/Himeth/data/rain100L/RainTrainL/RainTrainL/` (200 rain-*/norain-* pairs, real) | **Real data used as-is.** |
| Denoise | BSD400 + WED (clean pool, noise synthesized online) | BSD400/WED not found anywhere on devon | **Documented substitution**: `dataset_utils.py`'s denoise path only needs a clean natural-image pool (Gaussian noise is added online, `_add_gaussian_noise` — not a paired dataset). Substituting this project's existing DIV2K clean-image pool (already used throughout TEST01-17, high-quality, large enough). This is a real deviation from the paper's exact denoise corpus, reported here transparently, not hidden. |

Dataset directory layout follows `AdaIR/options.py`'s expected structure
exactly (`data/Train/Dehaze/`, `data/Train/Derain/`, `data/Train/Denoise/`,
`data_dir/{hazy,rainy,noisy}/*.txt` list files) so the **unmodified**
`utils/dataset_utils.py` and `utils/degradation_utils.py` can be reused
read-only, per this project's established read-only-reuse discipline.

## 4. Training protocol

- `de_type = ['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze']`
  (3-in-1 degradation, per user instruction — matches Table 1's setting,
  not Table 7's single-task dehaze-only protocol).
- Optimizer/schedule: AdamW, lr=2e-4, `LinearWarmupCosineAnnealingLR`
  (warmup 15 epochs) — same as the released `train.py`, unmodified.
- Batch size 8 (single GPU, vs. paper's 32 across 4 GPUs — devon has 1
  RTX 4090). Loss: L1, same as released.
- **Epochs: rescoped from an initial 30-epoch target to 8, after a
  timing calibration.** A smoke test (20-30 real training steps, timed
  directly) showed: fp32 batch=8 OOMs on a 24GB RTX 4090; fp32 batch=6
  fits (20.4GB) at 0.253s/step; AMP (`torch.amp.autocast`+`GradScaler`)
  batch=8 fits (13.9GB) at 0.210s/step. Even with AMP, one epoch over
  the FULL real dataset (97,035 samples: 72,135 real OTS dehaze +
  24,000 real Rain100L-derived derain + 900 denoise) would be
  ~12,129 steps ≈ 42 minutes/epoch — **≈21 hours for 30 epochs, per
  variant, x5 variants ≈ 106 hours.** Not remotely an overnight budget.
  **Rescoped, transparently**: dehaze subsampled to a real (not
  fabricated) seeded random 10,000-image subset of the 72,135 (full list
  preserved at `data_dir/hazy/hazy_outside_FULL_72135.txt` for a future
  full-scale run); derain and denoise kept at full real scale (derain's
  own 200 unique pairs are already repeated x120 by the released
  `dataset_utils.py`, matching upstream convention). Resulting epoch:
  34,900 samples, 4,362 steps at batch=8 — measured **~1s/first-step,
  reasonable per-step cost thereafter** (see `logs/train_A_baseline.log`
  for the live calibration). **8 epochs per variant** was chosen as the
  epoch count that plausibly completes all 5 variants sequentially in
  one overnight run at this corpus size — "considerable" relative to
  what a component-ablation comparison needs to show a directional
  trend, though short of the paper's own 20-epoch single-task ablation
  protocol (which used a far smaller, heavily-repeated dataset).
  Checkpointed every epoch (every-5th kept, others pruned to save disk)
  so partial progress is never lost if the run needs to be cut short.
  **If initial results (A vs E especially) look promising and time
  allows, extending select variants' epoch count is the natural
  follow-up — not re-running everything from scratch.**
- **Execution order: sequential, not concurrent.** AdaIR (~28.8M params,
  Transformer, patch 128×128) is far larger than this project's usual
  ~7.4M-param NAFNet-M student — concurrent multi-variant training (the
  pattern used throughout TEST07-17) is not attempted here; each variant
  trains to completion before the next starts, launched via
  `nohup`+`disown` (learned from TEST17's SSH-disconnect incident) inside
  a single wrapper script (`run_all_variants.sh`) so the whole sequence
  survives connection drops and runs unattended overnight.
- Single seed per variant this pass (not 3-seed, unlike the NAFNet
  student tests) — explicitly a scope-vs-time tradeoff, flagged as a
  limitation: this test establishes whether components matter at all,
  not seed-level statistical confidence. A follow-up could add seeds to
  whichever variants turn out interesting.

## 5. Frequency-domain diagnostics ("draw diagrams in frequency domain
## to see actually what happened")

For each trained variant (as its checkpoint becomes available) and for
the original released checkpoint (as a reference), on a fixed set of
representative images (one per degradation: Rain/Haze/Noise):

- FFT magnitude spectrum (log-scale, shifted to center) of: the AFLB's
  input feature `y`, the mined `high_feature`/`low_feature` (post-mask,
  pre-FMoM), the FMoM output `agg`, and the final AFLB output — at each
  of the 3 AFLB positions, for variants where the corresponding AFLB
  component is actually active.
- The learned mask itself, visualized as a box overlay on the spectrum
  (for `mask_mode="learned"` variants) vs. the fixed 10×10 box (for
  `mask_mode="fixed"`) — a direct visual answer to "what is the mask
  actually doing."
- One composite figure per (image, AFLB position) showing the pipeline
  left-to-right: input spectrum → mask → low/high split → FMoM-modulated
  → merged output — extending TEST06's single-example-per-scene approach
  (which only covered AFLB3 on the released checkpoint) to all 3 AFLBs
  and to every trained variant, directly implementing this task's
  "check and draw diagrams in frequency domain" instruction.
- Script: `test18/scripts/frequency_diagrams.py`. Can run against the
  released checkpoint immediately (doesn't need to wait for training),
  and against each variant's checkpoint as it finishes.

## 6. Quantitative evaluation

Per variant, per degradation (dehaze/derain/denoise σ=15/25/50), on
held-out test splits (SOTS-outdoor test 500, a held-out Rain100L split,
and a fixed synthetic-noise BSD68-style set) — matching the paper's own
evaluation protocol as closely as available data allows: PSNR, SSIM.
Reported both per-degradation and averaged, in the same shape as the
paper's Table 7, plus a column comparing against E_full's from-scratch
numbers (pipeline sanity check) and against the released checkpoint's
published numbers (external sanity check, not a reproduction claim —
different data scope, so not expected to match exactly).

## 7. Documentation (mandatory, per instruction)

- `test18/design/TEST18_PLAN.md` — this file.
- `test18/report/test18_report.md` — final report, same structure
  discipline as every prior test (objective, method, results,
  statistical caveats, decision, source files).
- `test18/results/Snapdragon_*` — N/A this test (no NPU phase; this is
  purely an AdaIR-teacher-side investigation, not a student/deployment
  one).
- `test18/results/*.xlsx` — quantitative results workbook (per-variant,
  per-degradation PSNR/SSIM tables, matching Table 7's shape).
- `test18/results/frequency_diagrams/` — all FFT visualization PNGs.
- Every real-data-availability decision, substitution (denoise corpus),
  and interpretive choice (FMoM-off aggregation fallback) is documented
  inline above and restated in the final report — this project's
  established discipline of never silently substituting or smoothing
  over a limitation.

## 8. Nested "small plan after results" (per instruction)

This section is deliberately left as a **template to fill in once
Phase 4-6 results land**, not pre-written now, because the actual next
step depends on what's found — pre-committing to a specific follow-up
before seeing data would repeat the mistake TEST01-06R's whole arc was
built to avoid (assuming from the architecture rather than measuring).

Decision framework for what to write once results are in:

- **If E_full (our retrained full AdaIR) reproduces the paper's Table 7
  ablation trend (monotonic PSNR increase a→e) AND the frequency
  diagrams show the mask/FMoM visibly doing something structured**: this
  would be a genuine tension with TEST01-06R's frozen-checkpoint null
  result, worth its own focused follow-up test reconciling "matters at
  training time" vs. "doesn't matter for a frozen checkpoint's inference
  behavior" — these are not actually contradictory claims, but the
  follow-up should say precisely why.
- **If E_full does NOT reproduce the paper's trend** (e.g. flat or
  inconsistent across a-e): investigate training-protocol gaps first
  (epoch count, single-task vs. 3-task, denoise-corpus substitution)
  before concluding anything about the architecture — a failed
  reproduction is evidence about this test's setup, not about AdaIR,
  until ruled out.
- **If the frequency diagrams show the learned mask converging to
  degenerate behavior (e.g. always-maximal or always-minimal box,
  independent of input)** even though PSNR still improves a→e: that
  would suggest the ablated components help via a *different* mechanism
  than genuine frequency-adaptivity (e.g. added capacity/nonlinearity
  from the extra conv/attention layers, not the frequency split itself)
  — directly testable by comparing the learned-mask variant's mask
  images across Rain/Haze/Noise inputs.

The actual filled-in small plan will be written into
`test18/report/test18_report.md`'s final section once results exist.
