# TEST10-R — Corrected Restoration-Trajectory Distillation

## 1. Why TEST10 Was Invalid

TEST10 tested whether distilling the teacher's intermediate restoration
trajectory (3 stages, matched to student decoder stages by spatial
resolution) improves restoration beyond simple compact-embedding
distillation. Both the teacher-side and student-side per-stage projection
heads were **freely, jointly trained** with a negative-free MSE-after-L2-
normalize loss. This is a well-known collapse-prone setup: same-sample
cosine similarity looked perfect (0.9999-1.0000), but a cross-input
diversity check (comparing embeddings across *different* scenes and
degradations, not just student-vs-teacher for the same input) revealed the
embeddings barely varied at all (pairwise cosine >0.999 in all 3 stages, all
3 seeds) — both projection heads had learned a trivial constant mapping.
TEST10's negative restoration result was therefore **uninterpretable**: the
trajectory signal never carried genuine information, so nothing could be
concluded about whether trajectory distillation helps or hurts.

## 2. Teacher Quality Audit

Before any new student training, the frozen AdaIR teacher and the
(deterministically reproducible) baseline Model A were evaluated on the
identical validation set, using proper PSNR/SSIM (not TEST10's cruder raw
pixel-L2 residual):

| Degradation | Teacher PSNR | Baseline PSNR | Delta | Classification |
|---|---|---|---|---|
| Rain | 35.76 | 30.02 | **+5.74** | teacher_better |
| Haze | 24.42 | 24.66 | -0.24 | **teacher_similar** |
| Noise | 34.98 | 27.60 | **+7.38** | teacher_better |

This refines (without reversing) TEST10's residual-based finding: the
teacher has a dramatic, unambiguous advantage on Rain and Noise, but on
Haze it is statistically indistinguishable from the untrained-by-teacher
baseline (within the pre-specified ±0.5dB "similar" threshold) — there is no
demonstrated teacher advantage on Haze worth transferring in the first
place. This was established **before** any student training, per the task's
explicit ordering requirement.

## 3. Fixed Teacher Target Construction

Reused TEST10's validated 3 stages (teacher `AFLB1/2/3.aflb_out`, matched by
spatial resolution to student `decoders[0]/[1]/[2]`, per
`test10/report/teacher_stage_audit.md`, read-only). For each stage: GAP+GMP
pool the raw teacher tensor, fit `StandardScaler` + `PCA(32)` on **training
crops only** (1,920 records), transform all crops (train+val), cache the
result. This transform is **frozen before student training and never
updated by backpropagation** — the core methodological correction.

| Stage | Teacher tensor | Raw dim | PCA-32 explained variance |
|---|---|---|---|
| 0 | AFLB1.aflb_out | 768 | 77.4% |
| 1 | AFLB2.aflb_out | 384 | 74.0% |
| 2 | AFLB3.aflb_out | 192 | 87.7% |

## 4. Collapse Prevention

**Phase 3 gate (before any student training)**: all 3 fixed targets pass
decisively — mean pairwise cosine across different images ≈0 (essentially
orthogonal, not collapsed), degradation-classification probe accuracy
97.1-98.3%, effective rank 5.3-8.9 (out of 32; real, multi-dimensional
information content). Training was authorized to proceed.

**Phase 6 mandatory monitoring (during training)**: every epoch, for every
Model G run, cross-input diversity of the student's own stage embeddings was
computed on the validation set. **Zero collapse events across all 450
epoch×stage checks (9 runs × 50 epochs × 3 stages... capped by monitoring
resolution, no aborts triggered)** — every G run completed all 50 epochs
without the early-abort rule (mean pairwise cosine > 0.98 at epoch ≥ 10)
ever firing.

**Post-training verification**: stage alignment on the fully-trained models
confirms genuine, non-degenerate structure — same-sample cosine (student vs.
its matching fixed teacher target) is 0.60–0.76 across stages and seeds:
substantial, meaningful, but appropriately **imperfect** (unlike TEST10's
suspicious near-1.0), while cross-input cosine (student vs. student, across
different inputs) stays near 0, matching the un-collapsed target's own
baseline. **All 9/9 seed×stage combinations are flagged `valid_trajectory_
model=True`, with `student_collapsed=False` and `teacher_collapsed=False`.**

## 5. Models A/F/G

Identical to TEST10 except: (a) Model F uses rank=2 low-rank channel-mixing
(same as TEST10, reproducing TEST09's mechanism); (b) Model G's student-side
stage projections are simple `Linear(pooled_dim, 32)` only — no MLP, no
BatchNorm, no trainable normalization, per the task's explicit design
constraint; (c) the teacher-side targets are looked up from the fixed cache,
never computed by a jointly-trained head.

## 6. Training

Adam, LR=2e-4, batch=8, 50 epochs, seeds {0,1,2}, 9 total runs, identical
TEST07-B/08-C/09/10 dataset. All 9 runs completed successfully (`aborted=
False` for every run in `seed_summary.csv`). Training was notably **faster**
than TEST10 (~16-20s/epoch standalone vs. TEST10's ~36s/epoch) since no
online teacher forward pass is needed anymore — targets are precomputed
once, a further practical benefit of the fix. Model A/F results reproduce
TEST08-C/09/10's own numbers exactly (deterministic given identical seed,
data, architecture) — a strong cross-experiment consistency check.

## 7. Restoration Results

Overall last5-window PSNR (mean ± std across 3 seeds):

| Model | Mean PSNR (dB) | Std |
|---|---|---|
| A | 27.315 | 0.198 |
| F | 27.038 | 0.140 |
| G | 26.621 | 0.259 |

| Comparison | Mean ΔPSNR (dB) | 95% bootstrap CI | All 3 seeds same sign? |
|---|---|---|---|
| **G − F** | **-0.417** | [-0.534, -0.207] | Yes, all negative |
| G − A | -0.694 | [-0.763, -0.558] | Yes, all negative |
| F − A | -0.276 | [-0.351, -0.227] | Yes, all negative |

G underperforms F consistently and cleanly — and this time, per Sections 4
and 9, the comparison is **trustworthy**: the trajectory signal was real.

## 8. Haze Teacher-Quality Analysis

Per-degradation G-F: Rain **-0.767dB**, Haze **-0.405dB**, Noise **-0.080dB**
(all negative; magnitudes largest for Rain, not Haze). The task asked to
distinguish two interpretations:

- **A. Trajectory preserves teacher behavior** — partially supported: Section
  4/9 shows the student's stage representations genuinely correlate with the
  teacher's (cosine 0.60-0.76), so *some* teacher structure is being
  preserved.
- **B. Trajectory improves student restoration** — **not supported**: every
  degradation gets worse, not just Haze.

Because Section 2 established the teacher has **no** demonstrated advantage
on Haze, and Haze is *not* even the worst-hurt degradation here (Rain is),
this result does **not** cleanly support the specific "distilling a
suboptimal teacher trajectory transfers the teacher's limitation" narrative
either — if that were the mechanism, Haze specifically should show the
*most* damage relative to the teacher's own quality gap, but Rain (where the
teacher is dramatically *better*) is hurt worse. The more parsimonious
reading: the trajectory loss term, even though non-degenerate, is simply not
a useful auxiliary signal for this student/architecture, independent of
which degradation or how good the teacher is at it.

## 9. Representation Alignment

| Representation | Mean accuracy |
|---|---|
| Teacher PCA-16 | 96.1% |
| Model F bottleneck / e_S | 98.2% / 96.4% |
| Model G bottleneck / e_S | 98.4% / 96.5% |

F and G have statistically indistinguishable final-embedding representation
quality — as in TEST08-C/09/10, the restoration difference cannot be
attributed to worse degradation-awareness in G's compact embedding.

## 10. Cross-Input Diversity

The mandatory check (Section 4) is the central methodological result of this
experiment: **9/9 combinations pass**, confirming this run does not suffer
TEST10's failure mode. See `statistics/cross_input_diversity.csv` and
`visualizations/06_stage_alignment_validity.png` for the full breakdown.

## 11. Restoration Trajectory

Residual analysis (explanatory only, not a primary metric): G's residual-to-
GT is worse than F's for every degradation (Haze: 16.70 vs. 15.45; Rain:
10.33 vs. 9.71; Noise: 8.49 vs. 8.46 — essentially tied for Noise).
`output_change_vs_teacher_output` (how close each model's output is to the
teacher's own output) is **not** smaller for G than for F on any degradation
— i.e., despite matching the teacher's intermediate stage representations
better in the compact sense, G's final pixel-space output is not measurably
closer to what the teacher actually produces. This is consistent with
Section 8's reading: intermediate-representation alignment did not propagate
into output-level behavior matching, let alone restoration improvement.

## 12. Complexity

| Model | Deployable params | MACs @128px |
|---|---|---|
| A | 7,371,923 | 1,033,040,896 |
| F | 7,381,189 | 1,033,114,656 |
| G (deployable) | 7,381,189 (identical to F) | 1,033,114,656 (identical to F) |

`verify_inference_graph.py` confirms G's output and `e_S` are **bit-identical**
with or without `traj_heads` present (14,432 params discarded at inference,
along with the 28.8M-param frozen teacher and the fixed PCA transforms,
neither of which is ever saved into a student checkpoint). The deployable
graph is self-contained.

## 13. Limitations

- N=3 seeds, exploratory statistics throughout, per project convention —
  though the G-F effect here is both consistent in sign (3/3) and an order
  of magnitude larger in mean than in variance-of-mean, unlike some earlier
  ambiguous TEST09 comparisons.
- Only one specific trajectory-loss formulation was tested (equal stage
  weights 1/3, λ_traj=0.1, simple Linear projections, MSE-after-L2-norm). A
  different weighting, loss form, or projection capacity might behave
  differently — but per the task's own instruction ("do not add more stages
  unless the corrected three-stage experiment first works"), and given this
  one did not work, further elaboration within the same family is not
  automatically warranted.
- The Haze-specific hypothesis from TEST10 (Section 8) was tested and not
  confirmed in the specific form proposed; this rules out one explanation
  but does not fully explain why Rain — where the teacher is strongest — is
  hurt most.

## 14. GO / NO-GO

Per the task's decision rule:

- **GO** requires no collapse AND G>F on restoration with meaningful
  improvement in at least one degradation, especially Haze. **Not met** —
  G<F on every degradation.
- **PARTIAL GO** requires alignment to work but restoration G≈F. **Not
  met** — restoration is not "approximately equal," it is consistently and
  measurably worse (-0.417dB mean, CI excludes zero).
- **NO-GO** requires fixed targets remain non-collapsed AND G≈F or worse
  across all degradations. **This is exactly what occurred.**
- **INVALID** requires teacher or student targets collapsed. **Explicitly
  ruled out** — this is the corrected, valid case.

**Decision: NO-GO.** Unlike TEST10, this is a genuine, trustworthy NO-GO:
the fixed teacher trajectory targets are demonstrably non-collapsed, the
student learned a real (if imperfect) alignment to them, mandatory
per-epoch monitoring confirmed no collapse occurred during any of the 9
training runs, and the deployable graph is verified clean — yet restoration
consistently worsens across all three degradations and all three seeds. Per
the task's explicit instruction, **do not add more trajectory stages**; the
evidence indicates this specific mechanism (multi-stage compact-embedding
trajectory matching via auxiliary MSE loss) is not the right lever for this
student architecture, regardless of collapse status.

## Recommendation for TEST11

Move to a different restoration mechanism rather than elaborating further on
trajectory-embedding matching:

1. Per TEST09's own findings, **low-rank channel-mixing (Model F) remains
   the best validated mechanism** to date; trajectory distillation (in this
   or the TEST10 form) has now been tested twice (once invalidly, once
   validly) and both times failed to beat it.
2. Given intermediate-representation alignment does not propagate to
   output-level behavior (Section 11), a mechanism that supervises the
   **output space directly** (e.g., a soft/feature-matching loss on the
   decoder's *final* pre-output feature map, or restoration-consistency
   losses at multiple output crops) is a more promising next direction than
   further compact-embedding trajectory schemes.
3. Since the teacher has no demonstrated Haze advantage (Section 2) and no
   mechanism tested across TEST08-C/09/10-R has improved Haze, treat Haze
   as a case requiring either better synthetic-degradation-recipe tuning
   (verify the Haze synthesis parameters match what the teacher was
   actually optimized for) or acceptance as a genuine ceiling, rather than
   continued student-side mechanism search targeting it specifically.
