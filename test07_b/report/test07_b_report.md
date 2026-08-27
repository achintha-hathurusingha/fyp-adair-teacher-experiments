# TEST07-B: Compact Latent Distillation Validation

## 1. Executive Summary

Models A (baseline NAFNet) and B (baseline + compact teacher-latent distillation,
lambda_kd=0.1) were trained for 50 epochs, 3 seeds each, on a fresh 80-train/20-val
scene-disjoint DIV2K-derived dataset (Rain/Haze/Noise degradations). Model B's
representation head learned to match the teacher's PCA-16 latent almost perfectly
(cosine similarity ~0.988-0.990, degradation-probe accuracy ~96.2% vs. the teacher's
own 96.1%). Despite this, Model B did **not** produce a restoration improvement over
Model A: mean delta on the primary metric (last-5-epoch mean validation PSNR) was
**-0.79 dB**, negative in all 3 seeds. A secondary, non-obvious finding survived the
per-degradation breakdown: Noise consistently favored B (+1.24 dB mean, positive in
all 3 seeds) while Rain (-2.59 dB) and Haze (-1.03 dB) consistently favored A.
**Decision: NO-GO FOR SIMPLE KD** — representation-matching alone, via this
compact-latent MSE loss, is insufficient to improve restoration quality on this
architecture/dataset.

## 2. Motivation and Correction from TEST07-Pilot

TEST07-Pilot's Models A-D all landed within ~2 points of each other on a degradation
probe (55-57%), and the restoration comparison between A and B specifically was noisy
and inconclusive (final-epoch numbers favored B/C/D, a 5-epoch smoothed average
favored A). TEST07-Pilot's honest conclusion was that a longer, statistically
grounded, apples-to-apples re-run of Model B alone (vs. A) was needed before testing
C or D. TEST07-Pilot's student bottleneck pooling was also acknowledged as an
asymmetry: GAP-only (256-dim), unlike the teacher's GAP+GMP (768-dim) pooling.
TEST07-B corrects this: Model B's bottleneck is now pooled identically to the
teacher's convention, GAP+GMP (512-dim, since the student bottleneck has 256
channels vs. the teacher's 384), then projected via `Linear(512, 16)`.

## 3. Dataset

- Source: the 100 DIV2K validation images already downloaded read-only for TEST06
  (`test06/data/div2k_val/DIV2K_valid_HR/`), reused without modifying TEST06.
- Split: 80 train scenes / 20 val scenes, **scene-disjoint** (verified by explicit
  set-intersection assertion in `build_dataset.py`, confirmed 0 overlap).
- Training crops: 8 random 128x128 crops per train scene (640 total), each with its
  own seeded RNG so crops are reproducible but non-degenerate.
- Validation crops: 1 fixed, deterministic 128x128 crop per val scene (20 total),
  identical across every model/seed.
- Degradations: Rain, Haze, Noise, synthesized with randomized parameters
  independently per crop (same synthesis methodology as TEST05.5/TEST07-Pilot).
- Patch-sampling option used: **Option 2 (precomputed multi-crop cache)** — chosen
  over online AdaIR-during-training extraction for run-time tractability given an
  overnight/same-session budget. All 660 crops (640 train + 20 val) x 3 degradations
  = 1980 images were pre-synthesized to disk, and the teacher's latent was extracted
  once per (crop, degradation) pair from that exact cache — so the teacher embedding
  a student sees during training corresponds to the *exact same pixels* the student
  is being trained on, satisfying the task's "same crop" requirement without paying
  the cost of running the (large, frozen) teacher forward pass every training step.

## 4. Teacher Extraction and PCA-16

- Checkpoint: `weights/adair3d.ckpt`, SHA256 verified to match
  `f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb` before any
  extraction ran.
- Teacher frozen: all parameters `requires_grad_(False)`; no gradient ever flows
  into the teacher during Model B's training.
- `latent_pre` pooled via GAP+GMP -> 768-dim raw vector.
- StandardScaler + PCA-16 fit **strictly on the 1920 training-split records**
  (leakage-safe; the 60 validation records never touch the fit). Explained variance:
  **68.44%** (comparable to TEST05.5's/TEST07-Pilot's leakage-safe fits on similar
  degradation-conditioned latents).
- Per-component explained variance is front-loaded (component 1 alone: 32.8%),
  consistent with degradation type being the dominant axis of variation in this
  latent, as established in earlier tests.

## 5. Model Architectures

Both models share the exact same locked NAFNet M-arm backbone from
`fyp-adair-distill` (width=16, enc_blk_nums=[2,2,4,8], middle_blk_num=12,
dec_blk_nums=[2,2,2,2], layernorm2d, affine_clamp full-res norm), imported
read-only and composed via a faithful forward-pass replica (`PilotNAFNetBase`)
that exposes the bottleneck tensor before the decoder.

- **Model A**: plain NAFNet, no projection head, no KD loss.
- **Model B**: adds `self.proj = nn.Linear(512, 16)` on the GAP+GMP-pooled
  bottleneck, producing `e_S` (16-dim), trained with
  `loss = L1(out, clean) + 0.1 * MSE(e_S, e_T)`.

Parameter overhead of B over A: **8,208 extra params (+0.111%)**. MACs overhead
at 128px input: **8,192 extra MACs (+0.0008%)**, i.e. negligible — the projection
head is a single small linear layer, and both figures come from the same validated
`count_macs` (FlopCounterMode-based) utility from `fyp-adair-distill`. These are
theoretical complexity numbers only, not NPU latency claims.

## 6. Training Configuration and Baseline Sanity Checks

Identical across A and B except for the KD branch:

| Check | Status |
|---|---|
| Same base NAFNet init per seed (same `torch.manual_seed(seed)` before model construction) | OK |
| Only B has the projection head + KD loss term | OK |
| Identical input normalization ([0,1] float, no extra normalization) | OK |
| Identical data ordering where possible (same `DataLoader` generator seeded per-seed) | OK |
| Identical validation set (20 fixed val crops x 3 degradations = 60 examples, deterministic) | OK |
| Teacher frozen + eval mode, no gradients through teacher | OK (`requires_grad_(False)` on all teacher params, verified in code) |

Hyperparameters: 50 epochs, Adam, LR=2e-4, batch=8, lambda_kd=0.1 (never tuned
against validation PSNR, per the task's explicit rule — numerically stable
throughout, so no escalation to 0.01 was needed).

**Data-integrity note**: the first attempt at all 6 parallel training runs hit a
shared-CSV write race (6 processes concurrently read-modify-wrote the same
`epoch_metrics.csv`/`seed_summary.csv`), which silently corrupted the summary
statistics (identical, wrong `final_psnr` across all 6 rows). This was caught
before any downstream analysis, root-caused, fixed (each run now writes to its own
per-run CSV, merged in a separate deterministic step), and **all 6 runs were
retrained from scratch**. Every number in this report comes from that clean re-run.

## 7. Restoration Results — Primary Metric

Primary metric per the task spec: last-5-epoch mean validation PSNR/SSIM (smoothed
final-training-window average, not a single lucky epoch).

| Seed | A last5 PSNR | B last5 PSNR | Delta (B-A) | A last5 SSIM | B last5 SSIM | Delta (B-A) |
|---|---|---|---|---|---|---|
| 0 | 27.182 | 26.168 | -1.014 | 0.8055 | 0.8153 | +0.0098 |
| 1 | 27.210 | 26.398 | -0.812 | 0.8109 | 0.8226 | +0.0117 |
| 2 | 27.552 | 26.999 | -0.553 | 0.8295 | 0.8219 | -0.0076 |

Mean delta PSNR: **-0.793 dB** (std 0.231, 95% bootstrap CI over the 3 seeds:
[-1.014, -0.553]). Mean delta SSIM: **+0.0046** (std 0.0106, 95% CI [-0.0076,
+0.0117]) — essentially flat, straddling zero. PSNR is negative and consistent
across all 3 seeds; SSIM shows no consistent direction. Secondary metric
(best-epoch PSNR) tells the same story: mean delta -0.750 dB, CI [-0.882, -0.572].

**This is exploratory 3-seed evidence, not a claim of statistical significance** —
but the consistency of direction across all 3 independently-seeded runs (same sign
in 3/3 for PSNR) is a stronger signal than the pilot's single-seed noise permitted.

## 8. Per-Degradation Breakdown

Averaging across degradations would hide a real, seed-consistent pattern:

| Degradation | Mean delta PSNR (dB) | Std | All 3 seeds same sign? |
|---|---|---|---|
| Rain | -2.591 | 0.869 | Yes, all negative |
| Haze | -1.027 | 0.286 | Yes, all negative |
| Noise | **+1.238** | 0.418 | Yes, all **positive** |

Noise is the one degradation where the compact-latent KD signal consistently helped
restoration, in all 3 seeds. Rain shows the largest and most consistent harm.
This degradation-specific split is a genuine finding, not noise — it should inform
the design of any follow-up distillation approach (e.g., a degradation-conditioned
loss, rather than a single scalar KD weight applied uniformly).

## 9. Student Representation Probe

GroupKFold (by scene) logistic-regression degradation-classification probe:

| Representation | Accuracy | Notes |
|---|---|---|
| Teacher PCA-16 | 96.1% | Reference upper bound (seed-independent) |
| Model A bottleneck (GAP+GMP, no KD) | 86.3% (std 6.9pp across seeds, seed0 outlier at 78.4%) | Learns substantial degradation info even with NO explicit KD signal |
| Model B bottleneck (GAP+GMP) | **98.4%** (std 0.3pp) | Slightly *exceeds* the teacher's own probe accuracy |
| Model B compact `e_S` (16-dim projection) | 96.2% (std 0.1pp) | Matches the teacher's 96.1% almost exactly |

Two things stand out: (1) even the undistilled Model A bottleneck is reasonably
degradation-aware (NAFNet's own representations pick up degradation cues from the
restoration objective alone), and (2) Model B's KD loss makes the *bottleneck as a
whole* even more degradation-discriminable than the teacher, and its explicit
projection head `e_S` matches the teacher's own probe accuracy almost exactly. The
distinction the task asked for — "does the student learn degradation info, or does
it only learn it inside the projection head" — resolves clearly: **both**. The
representation transfer objective was met.

## 10. Teacher-Student Embedding Alignment (Model B)

| Seed | Cosine(e_S, e_T) | MSE(e_S, e_T) |
|---|---|---|
| 0 | 0.9901 | 0.5737 |
| 1 | 0.9882 | 0.7262 |
| 2 | 0.9883 | 0.7120 |

Cosine similarity is consistently ~0.99 across all 3 seeds — Model B's compact
embedding is very well aligned with the teacher's PCA-16 target.

## 11. Interpretation Hierarchy

The task specifies a strict order: (1) did the student match the teacher, (2) did
the student become more degradation-aware, (3) did restoration improve — #3 is
decisive, and improvement on #1/#2 alone must not be reported as "KD works."

1. **Did Model B match the teacher's embedding?** Yes — cosine ~0.99, MSE ~0.6-0.7,
   consistent across seeds.
2. **Did Model B become more degradation-aware?** Yes — both its bottleneck (98.4%)
   and its explicit projection head (96.2%) reach or exceed the teacher's own probe
   accuracy (96.1%).
3. **Did restoration improve?** **No** — mean delta PSNR -0.79 dB, negative in all
   3 seeds on the primary metric.

Per the task's explicit rule, #3 is decisive: representation-probe and alignment
success **cannot** be reported as evidence that "KD works." It did not, for
restoration quality, on this architecture and this loss formulation.

## 12. Complexity / Overhead

Model B adds a single `Linear(512, 16)` head: **+8,208 params (+0.111%)**,
**+8,192 MACs at 128px (+0.0008%)** — both effectively negligible relative to the
base 7.37M-param, ~1.03 GMACs (128px) NAFNet backbone. This overhead is
theoretical-complexity-only; no NPU latency profiling was run (out of scope for
this experiment), and per prior findings (F1), normalization — not MACs — dominates
INT8 Hexagon NPU latency, so this small MACs overhead is not expected to be the
limiting cost driver regardless.

## 13. Decision Rule Applied

Per the task's pre-specified decision rule:

- GO: consistent ΔPSNR>0, ΔSSIM>=0 across seeds, low variability → **not met**
  (ΔPSNR negative in 3/3 seeds).
- WEAK GO: representation transfer improves but restoration is small/inconsistent
  → **not met** (restoration harm is consistent, not merely inconsistent/small
  — it's a clear, reproducible -0.79dB mean deficit with the same sign in 3/3
  seeds).
- **NO-GO FOR SIMPLE KD: B<=A across seeds AND e_S matches e_T well → MET.**
  Representation-matching alone, via this compact-latent MSE loss, is insufficient
  to improve restoration quality on this architecture/dataset. This is distinct
  from a "failure of representation transfer" (which would require alignment/probe
  to ALSO fail — they did not).
- FAILURE OF REPRESENTATION TRANSFER: e_S/e_T similarity AND probe both fail to
  improve → **not applicable** — both succeeded.

## 14. Decision: NO-GO FOR SIMPLE KD

**Model B's compact-latent distillation (bottleneck -> GAP+GMP -> Linear(512,16),
MSE against a leakage-safe teacher PCA-16 target) does not improve restoration
quality over the plain baseline NAFNet, despite successfully transferring the
teacher's degradation representation.** The representation-matching hypothesis for
this specific loss formulation is not supported by restoration outcomes. The one
qualifying nuance is a consistent, seed-stable Noise-specific benefit (+1.24dB
mean), which should not be generalized into an overall "GO," per the task's
explicit "do not average away a degradation-specific failure" instruction working
in reverse — do not average away a degradation-specific *success* into a broader
claim either.

## 15. Recommended Next Experiment

Per the decision rule's NO-GO branch: move away from simple bottleneck-embedding
MSE distillation toward either (a) **restoration-trajectory distillation** (e.g.,
matching intermediate decoder feature maps or output-level soft targets, which
carries more spatially-localized information than a single 16-dim bottleneck
vector), or (b) **degradation-conditioned spatial conditioning** (e.g., FiLM-style
modulation informed by the degradation-aware embedding, rather than an auxiliary
loss term on it) — TEST07-Pilot's untested Model C (FiLM conditioning) becomes the
natural next candidate, since a good degradation-aware embedding already exists
(#9/#10 above), but this experiment shows that embedding needs to actively steer
the decoder rather than merely exist as an auxiliary training signal. The
Noise-specific consistent benefit (Section 8) is also worth a targeted follow-up:
does a degradation-conditioned mechanism recover the Rain/Haze regression while
keeping the Noise gain?
