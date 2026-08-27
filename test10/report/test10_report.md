# TEST10 — Restoration Trajectory Distillation

## 1. Motivation

TEST09 established that low-rank channel-mixing (Model F) is the most
reliable conditioning mechanism found so far, but that Haze consistently
receives the *strongest* conditioning signal of any degradation at every
tested depth without ever obtaining a reliable restoration improvement.
H_TRAJ proposes that the compact degradation state describes WHAT
degradation is present but not HOW the teacher progressively restores the
image — so the student should distill the teacher's restoration
*trajectory* across multiple internal stages, not just its final compact
latent.

## 2. Evidence from TEST07-09

- TEST07-B: e_S ≈ e_T (representation transfer succeeds), restoration does
  not improve.
- TEST08-C: e_S can causally control restoration (verified via random/
  shuffled/zero-embedding controls and donor-recipient intervention), but
  the benefit is concentrated in Rain; Haze is unrescued despite the
  strongest bottleneck modulation.
- TEST09: deeper FiLM (D/E) shows no reproducible benefit; low-rank
  channel-mixing (F) is a clean, reproducible win over C, but again driven
  by Rain, not Haze — and Haze receives the strongest modulation at every
  depth tested, including the newly added decoder stages.

## 3. Teacher Stage Audit

Full audit in `report/teacher_stage_audit.md`. Key finding: the teacher
(AdaIR) has 3 encoder/decoder downsample levels (128→64→32→16) while the
student NAFNet has 4 (128→64→32→16→8) — the two networks' "bottlenecks" are
NOT at the same spatial resolution (16×16 vs 8×8), so naming-based
correspondence (matching "both bottlenecks") would have been wrong. Stages
were instead matched by **spatial resolution**, the one architecture-
invariant quantity both networks share: teacher `AFLB1/2/3.aflb_out`
(16×16/32×32/64×64) against student `decoders[0]/[1]/[2]` output (same three
resolutions). AFLB outputs were chosen over the plain encoder/decoder hook
tensors at the same resolutions because `aflb_out` IS the teacher's
frequency-conditioning transformation at that depth — the actual trajectory
information H_TRAJ is about.

## 4. Student Stage Correspondence

| Stage | Teacher | Student | Teacher ch | Student ch |
|---|---|---|---|---|
| 0 (deepest) | AFLB1.aflb_out | decoders[0] | 384 | 128 |
| 1 (mid) | AFLB2.aflb_out | decoders[1] | 192 | 64 |
| 2 (shallow) | AFLB3.aflb_out | decoders[2] | 96 | 32 |

Channel counts differ substantially (student is far narrower, as expected
for a 7.4M vs. 28.8M-param model), which is exactly why per-stage learned
projections (GAP+GMP → Linear → 32-dim) were required rather than any
direct tensor comparison.

## 5. Trajectory Distillation Formulation

`e_T^l = P_T^l(F_T^l)`, `e_S^l = P_S^l(F_S^l)`, both GAP+GMP pooled then
projected to 32-dim by a per-stage `Linear` head. Both `P_T^l` (3 heads,
43,104 params total, in a standalone `TeacherTrajectoryHeads` module) and
`P_S^l` (3 heads, part of Model G) are **freely, jointly trained** alongside
the student, per the task's explicit instruction ("the goal is to learn a
trainable stage representation," not a fixed PCA). `L_stage_l =
MSE(L2normalize(e_S^l), L2normalize(e_T^l))`, `L_traj = mean` of the 3 stage
losses (equal weights, `w_l = 1/3`). Model G's total loss: `L_restore +
0.1·L_KD + 0.1·L_traj`.

**⚠ This is the formulation that collapsed — see Section 9.**

## 6. Training Setup

Adam, LR=2e-4, batch=8, 50 epochs, seeds {0,1,2}, dataset reused read-only
from TEST07-B/08-C/09 (identical scene split, crops, degradation synthesis).
9 total runs (A/F/G × 3 seeds). Model F reproduces TEST09's mechanism with
rank=2 (vs. TEST09's rank=4, per this task's explicit spec).
Model G additionally runs the frozen AdaIR teacher **online** during
training (not from a precomputed cache, since raw per-stage tensors are
needed, not just the final PCA-16 embedding) — a standalone timing test
confirmed this adds no meaningful per-step slowdown (~36s/epoch standalone,
identical to A/F), though 3-way concurrent training of G (each loading the
28.8M-param teacher, ~4GB/process) ran at ~84s/epoch due to GPU contention,
not teacher overhead itself.

## 7. Restoration Results

Overall last5-window PSNR (mean ± std across 3 seeds):

| Model | Mean PSNR (dB) | Std |
|---|---|---|
| A | 27.315 | 0.198 |
| F | 27.038 | 0.140 |
| G | 26.705 | 0.239 |

| Comparison | Mean ΔPSNR (dB) | 95% bootstrap CI | All 3 seeds same sign? |
|---|---|---|---|
| **G − F** | **-0.334** | [-0.524, -0.110] | Yes, all negative |
| G − A | -0.610 | [-0.775, -0.337] | Yes, all negative |
| F − A | -0.276 | [-0.351, -0.227] | Yes, all negative |

G underperforms F in all 3 seeds, consistently. **This must NOT be read as
"trajectory distillation doesn't help restoration"** — see Section 9 for why
this comparison is confounded by representational collapse.

## 8. Haze Analysis

| Seed | Haze ΔPSNR (G−F) | Haze ΔSSIM (G−F) |
|---|---|---|
| 0 | (per-seed values in `statistics/per_degradation_deltas.csv`) | |
| Mean | **-0.176** | -0.003 |

Per-degradation G-F means: Rain -0.739, Haze -0.176, Noise -0.087. None
improve; Rain is hurt most. Given the collapse finding, these numbers
reflect the collapsed trajectory loss acting as unhelpful training noise
across the board, not a targeted Haze effect.

## 9. Stage Alignment — Representational Collapse

This is the central finding of TEST10. Measuring cosine similarity between
`e_S^l` and `e_T^l` alone gives a misleadingly positive picture (cosine
0.9999-1.0000, normalized MSE ≈0.0000 for all 3 stages, all 3 seeds) — this
looks like *perfect* alignment. But per-sample variation must also be
checked: computing the mean **pairwise** cosine similarity among the 60
validation embeddings (varying scene AND degradation) reveals it is *also*
~0.9999 on **both** the student and teacher sides, for **every stage and
every seed** (9/9 combinations flagged "collapsed" by the diagnostic, using
a >0.98 threshold):

| | Student mean pairwise cosine | Teacher mean pairwise cosine |
|---|---|---|
| range across all 9 stage×seed combos | 0.99986–0.99994 | 0.99907–0.99948 |

An embedding that cannot distinguish a Rain crop from a Haze crop from a
Noise crop — nor one scene from another — carries essentially zero
information. **Both `P_S^l` and `P_T^l` learned a trivial constant mapping**,
which trivially drives `MSE(L2normalize(e_S),L2normalize(e_T))` toward 0
without any genuine correspondence. This is the textbook collapse failure
mode of a jointly-trained, negative-free normalized-MSE objective (the same
failure mode SimSiam/BYOL-style methods guard against with stop-gradient,
predictor asymmetry, or explicit variance regularization — none of which
were specified or implemented here, matching the task's own "start simple"
instruction, which in this case was insufficient).

**Contrast with the final 16-dim KD embedding** (`e_S`, `e_T` — the
mechanism validated since TEST07-B): in this exact same Model G run, that
embedding's cosine similarity is 0.78-0.80 — healthy, informative, and
consistent with every prior experiment. The difference is training
discipline: the final embedding's teacher target is a **fixed** leakage-safe
PCA-16 transform (never jointly optimized, cannot collapse to satisfy the
student), whereas the trajectory heads were **freely** jointly trained on
both sides with no such anchor. This isolates the collapse specifically to
the new mechanism's design, not to this run, this codebase, or the KD
approach in general.

## 10. Restoration Trajectory Analysis

`output_to_input_change` (Section 11's explanatory metric) is comparable
across A/F/G (44.6 / 45.9 / 46.9 for Haze, in the same ballpark), meaning G
does move roughly as far from the input as F and A do — it is not simply
doing nothing. But since the direction of that movement is guided by a
collapsed (uninformative) trajectory signal, this movement is not
meaningfully "teacher-like."

## 11. Teacher/Student Residual Comparison

An unplanned but important finding: the teacher's own residual-to-GT for
Haze (15.73, in raw pixel-L2 units) is **worse** than baseline Model A's own
Haze residual (14.41) — `residual_gap_to_teacher` for A is **-1.32** (A is
*closer* to the ground truth than the teacher is, for Haze specifically, on
this synthetic degradation recipe). F is also closer than the teacher
(-0.28); only G is (barely) further (+0.26). For Rain and Noise, by
contrast, the teacher is clearly better than all three students (positive
gaps of 3.4-6.1). **This suggests part of the persistent Haze failure across
TEST08-C/09/10 may reflect a genuine ceiling in the teacher's own Haze
restoration quality on this synthetic recipe, not purely a student-mechanism
limitation** — a hypothesis worth verifying directly (e.g., inspecting
teacher outputs on Haze crops) before further architecture search on the
student side.

## 12. Complexity

| Model | Deployable params | MACs @128px |
|---|---|---|
| A | 7,371,923 | 1,033,040,896 |
| F | 7,381,189 | 1,033,114,656 |
| G (deployable) | 7,381,189 (identical to F) | 1,033,114,656 (identical to F) |
| G (training-time, incl. traj_heads) | 7,395,621 | n/a (training only) |

`verify_inference_graph.py` confirms G's restoration output and `e_S` are
**bit-identical** with or without `traj_heads` present — the trajectory
projection heads have zero causal effect on inference and are correctly
discarded for deployment (14,432 params removed), along with the entire
28.8M-param frozen AdaIR teacher and the separate `TeacherTrajectoryHeads`
module, neither of which is ever saved into the student checkpoint. The
final student inference graph is confirmed self-contained: no AdaIR, no PCA,
no teacher or trajectory projections, no training-only supervision branches.

## 13. Limitations

- The central limitation IS the finding: the trajectory-distillation
  mechanism as specified collapsed, so **H_TRAJ was not actually tested** —
  only a broken implementation of it was.
- N=3 seeds, exploratory statistics throughout, per project convention.
- Phase 4's optional transformation/delta distillation ablation was
  deliberately NOT run, per the task's own "only run after basic trajectory
  model is working" instruction — compounding a collapsed mechanism with a
  second loss term would not have produced interpretable results.
- The teacher-Haze-quality finding (Section 11) is suggestive, not
  conclusive — it uses raw pixel-L2 residual, not PSNR/SSIM, and was
  computed on only 20 validation scenes; it should be verified with a
  dedicated teacher-quality audit before being treated as established.

## 14. Whether G Fixes the Haze Failure

No — and the result is uninterpretable as a test of H_TRAJ regardless,
because the supervision signal driving G's trajectory loss carried no
information after collapse. Any interpretation ("trajectory info doesn't
help" or "it actively hurts") would be attributing meaning to a signal that,
per Section 9, does not exist in the trained model.

## 15. GO / NO-GO

Per the task's decision rule, the *observed* numbers alone would suggest
NO-GO (G≈F/worse for all degradations). But the task's own scientific-claim
discipline explicitly requires: "(1) teacher trajectory can be represented,
(2) student learns teacher trajectory, (3) student restoration changes, (4)
student restoration improves, (5) Haze failure is reduced — only #4
establishes restoration benefit." **Step #1 itself failed** — the teacher
trajectory was never meaningfully represented on either side, so steps #2-5
cannot be evaluated at all, let alone concluded against.

**Decision: NO-GO — but NOT on H_TRAJ.** This is a NO-GO on *this specific
implementation* (freely jointly-trained, unregularized, negative-free
per-stage projections), driven by a collapse failure mode that is
well-understood and fixable, not a NO-GO on the underlying hypothesis that
restoration-trajectory information could help. Per the task's own framing,
this should NOT be read as "move toward a different restoration operator" —
that would be the correct reading only if the trajectory signal had been
real and still failed to help.

## Recommendation for TEST11

Re-run the trajectory-distillation experiment with a **collapse-resistant**
formulation before drawing any conclusion about H_TRAJ:

1. **Fix the teacher-side projections** (`P_T^l`) rather than jointly
   training them — e.g., fit a leakage-safe per-stage PCA-32 (or even a
   fixed random projection) on training-split crops only, exactly matching
   the discipline already proven to work (and not collapse) for the final
   16-dim KD embedding. A fixed target cannot be gamed by a trivial
   constant mapping.
2. If joint training of both sides is still desired, add an explicit
   collapse-prevention term (e.g., a variance-preservation penalty per
   stage, in the spirit of VICReg, or a stop-gradient/predictor asymmetry
   as in BYOL/SimSiam) rather than a bare negative-free MSE.
3. Separately, verify the teacher-Haze-quality hypothesis from Section 11
   directly (visually inspect a handful of teacher Haze restorations
   against GT) — if the teacher itself has a real ceiling on this synthetic
   Haze recipe, no amount of distillation mechanism design will close that
   gap, and the right fix is upstream (better Haze synthesis parameters or
   accepting Haze as a genuine teacher-quality limitation rather than a
   student-side problem).
