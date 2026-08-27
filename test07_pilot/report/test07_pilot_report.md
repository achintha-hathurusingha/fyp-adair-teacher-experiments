# TEST07 Pilot — Compact Degradation Distillation

## 1. Objective

Determine, with a short pilot, whether (1) compact AdaIR latent distillation helps a small spatial NAFNet student, (2) using the compact representation as a spatial conditioning signal helps further, (3) a low-rank dynamic spatial kernel adds benefit, and (4) which direction the team should invest in next — not to obtain final numbers.

## 2. Models

All four models share the exact same LOCKED NAFNet architecture (`fyp-adair-distill`'s M arm: width=16, enc=[2,2,4,8], middle=12, dec=[2,2,2,2], N-F+clamp normalization), read-only, unmodified — composed via a faithful forward-pass replica (see `report/student_architecture_audit.md`), not edited in place.

- **A — Baseline**: plain NAFNet, `L = L_restore` (L1).
- **B — Compact latent distillation**: + `e_S = Linear(GAP(bottleneck))`, `L = L_restore + 0.1·MSE(e_S, e_T)`, `e_T` = teacher PCA-16 of `latent_pre`.
- **C — B + affine (FiLM) conditioning**: `gamma, beta = heads(e_S)`, `F_mod = (1+gamma)·F + beta`, near-identity init.
- **D — B + low-rank (R=2) dynamic kernel**: `K(e) = K_0 + a_1(e)K_1 + a_2(e)K_2`, one depthwise 3×3 conv at the bottleneck only, dynamic term zero-initialized.

## 3. Training Setup

Identical across all four: 40 training scenes × 3 degradations (120 patches), 10 validation scenes × 3 degradations (30 patches), fixed 128×128 crops from DIV2K validation images (read-only reuse of TEST06's already-downloaded set, images 40-79/90-99, disjoint from TEST06's own usage), parameter-randomized Rain/Haze/Noise (TEST05.5-style synthesis, fresh implementation). No augmentation (fixed crops, so cached teacher embeddings stay valid all pilot — a documented simplification). Adam, LR=2e-4, batch=8, 15 epochs, seed=0, λ_kd=0.1 (stable throughout — no reduction to 0.01 needed).

## 4. Teacher Representation

`latent_pre` (768-dim pooled GAP+GMP) extracted for all 150 pilot images with the exact TEST01-06R checkpoint (SHA256 confirmed matching). PCA-16 fit on **training rows only** (120 samples, leakage-safe per TEST05.5's methodology) — explained variance 75.5% (lower than TEST05.5's near-100% on 300 images, expected given only 120 training samples for a 768-dim fit).

## 5. Results

| Model | Best-epoch PSNR | Final-epoch PSNR | Smoothed (ep 11-15) PSNR | Smoothed SSIM |
|---|---:|---:|---:|---:|
| A | 20.82 (ep14) | 19.98 | **20.49** | **0.596** |
| B | 20.63 (ep15) | 20.63 | 20.04 | 0.543 |
| C | 21.11 (ep15) | 21.11 | 20.18 | 0.545 |
| D | 21.06 (ep15) | 21.06 | 20.25 | 0.534 |

**This is a genuinely mixed signal, reported honestly rather than cherry-picked.** Final-epoch-only comparison makes B/C/D look clearly better (+0.6-1.1dB) — but Model A's final epoch happens to be a noisy dip (its own best epoch, 14, is 20.82, close to C/D's best). The smoothed 5-epoch average — a more reliable statistic given only 30 validation patches — shows **Model A equal to or slightly ahead of B/C/D on both PSNR and SSIM**. SSIM in particular is consistently *worse* for B/C/D across every aggregation window, a real, if small, degradation-quality cost the KD loss may be introducing at this λ/epoch count.

## 6. Degradation Probe

| Representation | Dim | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Teacher PCA-16 | 16 | **93.3%** | 0.933 |
| Model A (raw bottleneck GAP) | 256 | 55.3% | 0.544 |
| Model B | 16 | 56.7% | 0.553 |
| Model C | 16 | 57.3% | 0.564 |
| Model D | 16 | 56.7% | 0.542 |

Distillation produced a small, consistent, directionally-correct signal: B/C/D all probe 1.4-2.0 percentage points above A. This confirms *some* degradation information transferred into the student's compact representation — but the student is nowhere close to the teacher's 93.3%, and this representation-level gain did not clearly translate into restoration-quality gain (§5).

## 7. Complexity

| Model | Params | MACs (128×128) | Extra params vs. A |
|---|---:|---:|---:|
| A | 7,371,923 | 1.033G | 0 |
| B | 7,376,035 | 1.033G | 4,112 |
| C | 7,384,739 | 1.033G | 12,816 |
| D | 7,382,981 | 1.033G | 11,058 |

All theoretical cost estimates (per the task's explicit instruction — NOT NPU latency). Overhead is negligible (<0.2% of total params) for all three mechanisms. Given `fyp-adair-distill`'s own finding (F1: normalization, not MACs, dominates NPU latency), this pilot's MAC-based cost comparison likely understates C/D's true on-device cost if their extra ops introduce additional normalization or irregular memory access — untested here, flagged for a future NPU-aware pass.

## 8. What Worked

- The read-only NAFNet reuse and faithful forward-replica pattern worked cleanly — all four models trained without incident.
- Leakage-safe PCA-16 teacher-embedding extraction and caching worked as designed.
- The representation probe shows real, if weak, evidence that compact-latent distillation transfers *some* degradation information to a spatial student, even at pilot scale.

## 9. What Failed

- Restoration-quality improvement from B/C/D over A is **not established** at pilot scale — the signal is within run-to-run/epoch-to-epoch noise, and SSIM points mildly negative for all three KD variants.
- 15 epochs / 120 training patches / one seed is too small a pilot to distinguish a true small effect from noise — the smoothed-vs-final-epoch disagreement above is itself the clearest evidence of this.

## 10. Recommended Next Direction

Per the task's own decision rule, **do not force a positive conclusion.** State it as specified: *"Teacher representation was highly degradation-discriminative, but short-pilot distillation did not yet produce a measurable restoration improvement."* The representation probe's small positive signal (B/C/D > A) is real but not yet restoration-quality-validated.

**Recommended next experiment**: repeat Model A vs. B only (drop C/D pending B's validation — no point layering more mechanism on an unconfirmed base effect), with (a) 3 seeds instead of 1, (b) a larger pilot (more scenes, closer to TEST05.5's 300-image scale, still short-epoch), and (c) report mean±std across seeds rather than a single run, before deciding whether affine conditioning (C) or dynamic kernels (D) are worth their added complexity.
