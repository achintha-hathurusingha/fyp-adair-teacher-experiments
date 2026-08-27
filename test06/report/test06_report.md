# TEST06 — Resolution-Dependent Frequency Influence

## 1. Research Question

"When AdaIR's frequency pathway actually has enough spatial resolution to become non-degenerate, does it begin to matter for restoration, and is that influence degradation-specific?"

This tests the falsifiable hypothesis H_RES: the practical influence of AdaIR's adaptive frequency mechanism increases when the feature-map resolution is large enough for the adaptive frequency boundary to become non-degenerate. H_RES has two parts: (A) where does the mask become active (06-A), and (B) once active, does changing the frequency-path content affect restoration (06-E). These are the primary decision gate.

## 2. Relationship to the AdaIR Paper

The paper's Table 7 reports benefits from FMiM/FMoM variants under **training** ablations (a different architecture is trained from scratch and compared). TEST05.5 tested frequency-path removal on an **already-trained, frozen checkpoint** at benchmark resolution — a different causal question, and not necessarily in conflict with the paper's trained-ablation result. TEST06 does not resolve the training-time question (that would require retraining, explicitly out of scope), but it does test a narrower, directly answerable question: does the frequency path matter at inference time, specifically at a resolution where it is confirmed mathematically non-degenerate (unlike TEST05.5's degenerate benchmark-resolution setup)?

## 3. Relationship to TEST01–05.5

TEST01 established the mask is degenerate at benchmark resolution and found activation "at ≥640-1024px" via a coarse resolution sweep. TEST05.5's frequency-path ablation (T0-T3) and cross-image swap found no causal effect — but entirely within the degenerate-mask regime, leaving open whether resolution was the confound. TEST06 closes this gap directly: source-code analysis (Phase 0) derives the exact activation arithmetic per AFLB, an expanded resolution sweep (Phase 1, 432 configurations across 12 resolutions × 5 aspect ratios) locates the precise activation threshold, and a dedicated causal experiment (06-E) is run specifically at a confirmed-active resolution.

## 4. Forward-Path Audit

Read directly from `AdaIR/net/model.py`'s `FreModule` and `AdaIR.forward()` (unmodified, read-only). Confirmed the mask-activation arithmetic is exact and resolution-dependent: `h_ = int((H_feat // 128) * threshold)`, requiring `H_feat // 128 >= 2` (i.e. `H_feat >= 256`) as a hard mathematical floor, with the actual threshold value (a learned sigmoid output, always <1) determining whether activation occurs above that floor. Because each AFLB's feature map is `x` resized to match its own decoder-stage spatial resolution — `input/8` for AFLB1, `input/4` for AFLB2, `input/2` for AFLB3 — the three AFLBs require very different **input** resolutions to reach even the mathematical floor: AFLB1 ≥2048px, AFLB2 ≥1024px, AFLB3 ≥512px.

TEST05.5's intervention scripts (`frequency_variants.py`, `model_variants.py`) were verified line-by-line against this source: `_fft_released` reproduces the mask/fft computation exactly, and the downstream FMiM/FMoM/`channel_cross_agg` architecture is left completely untouched in every T0-T3 variant. **Audit passed** — TEST05.5's earlier null result was not an artifact of a miscoded intervention.

## 5. Resolution Sweep (06-A)

432 successfully-run configurations (100 native CBSD68/Rain100L images at 481×321, plus a controlled grid: 8 DIV2K validation images × 12 resolutions [128–1536px] × 5 aspect ratios [1:1, 4:3, 3:2, 16:9, 2:1], center-cropped, no upscaling). 68 configurations were skipped (source image too small for a given crop) and 16 hit CUDA OOM at the largest resolutions (1280px+) — both handled explicitly, not silently treated as valid.

**AFLB1**: never activates in the tested range (feature resolution tops out at 192px at input=1536, short of the 256px floor).
**AFLB2**: never confirmed active — right at the edge. Observed β≈0.4966–0.4969 consistently across resolutions, just under the 0.5 threshold needed at feature resolution 256–320px (input 1024–1280px). Would very likely activate just above the tested range (~1536–2048px input).
**AFLB3**: activates starting at **input=768px** (feature resolution 384px), with 116/412 grid configurations (28%) showing genuine activation.

## 6. First Non-Degenerate Resolution

R_first(AFLB3) = 768px input / 384px feature resolution. At first activation: `mask_active_fraction = 0.000027` (a 1-pixel half-width box) but `raw_low_energy_fraction = 0.822` — the tiny mask carries 82% of the low+high energy split (natural images concentrate energy near DC), confirming this is a genuine, non-trivial activation rather than numerical noise. An earlier draft of the activation-detection script used an inappropriately strict threshold (`mask_active_fraction > 1e-4`) that masked this real signal; corrected to `> 0` (justified because `h_`/`w_` are integer-valued by construction, with no floating-point ambiguity near zero).

## 7. Restoration Sanity

No NaN or Inf observed in any of the 432 successful configurations. PSNR/SSIM remained in normal restoration ranges throughout (native benchmark images: 17.5–43.7dB). The 16 OOM cases were skipped and excluded from all downstream analysis, not imputed or ignored silently.

## 8. Frequency Activation Validation (Phase 6)

Before any causal swap, 10 scenes × 3 degradations (30 pairs) were independently verified at the 06-E dataset's resolution (1024×1024): `raw_low != 0`, `raw_high != 0`, and `mask_active_fraction > 0` for **all 30 pairs**. FFT magnitude / mask / raw_low / raw_high visualizations saved for 3 example scenes (`results/frequency_intervention/frequency_activation_examples/`). Phase 6 passed cleanly — proceeding to the causal experiment was justified by direct evidence, not assumption.

## 9. Same-Scene Frequency Intervention (06-E)

25 scenes (native DIV2K images 8–32, disjoint from the resolution-sweep images 0–7, avoiding any data leakage), 1024×1024, parameter-randomized Rain/Haze/Noise synthesis per scene. Baseline inference cached for all 75 (scene × degradation) combinations. Self-swap control (Phase 8, donor==recipient): **max L2 = 0.00000000 (exact)** — the intervention mechanism swaps only the intended `(raw_high, raw_low)` tensors and reproduces the original output bit-for-bit when nothing actually changes, confirming correctness before any causal claim was made.

**Phase 9 — the primary causal test**: 150 same-scene cross-degradation swaps (all 6 ordered recipient/donor pairs × 25 scenes). Mean normalized L2 (relative to the recipient's own normal output) = **0.000013**.

## 10. Control Experiments (Phase 10)

| Control | Normalized L2 |
|---|---|
| Same-scene cross-degradation (primary) | 0.000013 |
| Cross-scene, same-degradation | 0.000011 |
| Random, distribution-matched | 0.000012 |
| Zero tensor | 0.000012 |
| Mean tensor | 0.000012 |

The primary effect does not exceed any control, including the "arbitrary perturbation" controls (zero/random/mean at 0.000012) — a real causal effect would show a clear separation, as TEST04/TEST05.5's spatial-tensor interventions did at other points in the network. Here, swapping in a genuinely different degradation's frequency content is statistically indistinguishable from swapping in meaningless noise.

## 11. Donor-Behavior Analysis (Phase 11)

For each of the 150 swaps, the swapped output was compared against both the recipient's own normal output and the donor's normal output. In **0.0%** of swaps did the swapped output move closer to the donor's behavior than to the recipient's own — the frequency-path content, even when genuinely swapped between degradations, never pulls the restoration output toward the donor degradation's characteristic behavior.

## 12. Frequency-Band Sensitivity

**NOT RUN.** Explicitly gated on Phase 9 showing a non-trivial frequency-path effect. Since Phase 9's primary result did not exceed controls, running this phase would have produced numbers with no causal grounding to interpret — correctly skipped per the task's explicit instruction, not omitted by oversight.

## 13. Degradation-Specific Frequency Response

**NOT RUN.** Same gate as §12.

## 14. Compact Frequency Signature

**NOT RUN.** Gated on §13 producing a clear degradation-specific effect, which did not occur.

## 15. Interpretation

Resolution was the candidate confound TEST06 was built to test, and it has now been tested directly. AFLB3's frequency mask is confirmed genuinely non-degenerate at 1024px (Phase 6), yet a controlled causal swap there produces an effect indistinguishable from noise (Phase 9-10), and never pulls output toward donor behavior (Phase 11). This is a materially different and stronger test than TEST05.5's, which operated entirely within the degenerate-mask regime. The convergence of both results — null at degenerate resolution (TEST05.5) and null at confirmed-non-degenerate resolution (TEST06) — closes the resolution-dependence loophole that motivated this experiment.

## 16. Limitations

- AFLB1 and AFLB2 were never confirmed to activate within the tested resolution range (128–1536px input). AFLB2 is close (β≈0.497 vs. 0.5 needed) and would very plausibly activate at 1536–2048px input; AFLB1 needs ≥2048px. Their resolution-dependent behavior is genuinely untested, not refuted by this experiment.
- The intervention resolution (1024×1024) exceeds AdaIR's 128×128 training-crop size — while this is standard practice for restoration-model evaluation (not out-of-distribution in the usual sense, since full-image inference after patch-based training is the norm in this literature), it remains true that the checkpoint's behavior at 1024px reflects the trained model's generalization, not a resolution it was directly optimized for.
- Only AFLB3 was tested causally (06-E); a full test would repeat this at AFLB2's activation threshold if/when confirmed.
- The "mean tensor" control (Phase 10E) used the recipient's own per-scene spatial mean as a documented proxy for a true dataset-wide mean (computing the latter would require a full additional pass); this is noted as a simplification, not hidden.

## 17. GO / NO-GO

**NO-GO** for a frequency-derived spatial-kernel student signal. See `report/GO_NO_GO.md` for the full 15-question answer set and case classification (Case B: mask activates, 06-E is null).

## 18. Implications for the NAFNet Student

The `q_F` (frequency-response embedding) bridge proposed as a more "principled," frequency-preserving alternative to plain compact-latent distillation was the specific hypothesis this experiment was built to test — and it did not survive the test, even under the most favorable conditions available (a confirmed non-degenerate resolution, a clean causal-swap methodology with passing self-swap and multiple negative controls). Combined with TEST05.5, the evidence across two independent methodologies and the full tested resolution range (128–1536px) does not support routing the student's degradation-adaptive signal through AdaIR's frequency-specific computation. The recommended path for TEST07+ remains TEST05.5's validated compact-embedding distillation (`z_T → e_D`), not a frequency-response-derived signal.
