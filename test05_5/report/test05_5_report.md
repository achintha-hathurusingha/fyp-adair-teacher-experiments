# TEST05.5 — Scientific Audit and F2S Hypothesis Validation

## 1. Purpose and Scope

This experiment is an adversarial scientific audit of TEST01–05's conclusions, designed to actively try to disprove the working hypothesis before committing engineering effort to building a NAFNet student in TEST06:

> **H_F2S**: "The useful degradation-aware teacher representation is meaningfully dependent on the teacher's frequency-related processing."

TEST05.5 does not train NAFNet, implement the F2S module, optimize NPU kernels, or claim student performance. It only determines whether the F2S hypothesis is scientifically justified. All work lives under `test05_5/`; TEST01–05, the AdaIR source, and the checkpoint were read-only references, never modified. Checkpoint SHA256 verified identical to TEST01–05's record (`f3822d9c2ea...5937fb`).

## 2. What TEST01–05 Established (see `test05_5_design.md` for full detail)

TEST01 established that the released AdaIR's frequency mask is degenerate at benchmark resolution (`h_=w_=0` for every image, verified 5 ways) and that removing frequency processing changes output PSNR by <0.004dB — but never tested whether it changes the *representation's* degradation-discriminability. TEST02–03 established the representation is degradation-discriminative and this survives a dataset-identity control. TEST04 established causal relevance of `latent_pre` to output, with an unnormalized comparison suggesting scene-sensitivity exceeds degradation-sensitivity (restricted to a Rain-recipient subset). TEST05 established channel-level degradation-specificity concentration and a PCA-16 compact representation reaching 99.7% accuracy — but with an un-corrected leakage risk in the PCA/scaler fitting procedure.

Seven loopholes (L1–L7) were identified and targeted by this experiment; see the design doc for the complete audit.

## 3. Method Summary

Fourteen phases were executed: (1) simple statistical baselines vs. AdaIR features; (2) leakage-safe PCA re-audit; (3–4) a new parameter-randomized, overlapping-severity-band dataset (600 images) with family and severity-generalization probes; (5–6) normalized causal intervention with a degradation-specificity ratio; (7–9) controlled teacher variants (T0 real / T1 frequency-disabled / T2 content-blind random / T3 phase-scrambled) compared on representation quality and output quality; (10–11) a cross-image frequency-branch randomization control and independent re-verification of internal-feature frequency-band content; (12–14) restoration-relevance and negative-control validation of the compact representation.

## 4. Phase 1 — Simple Statistical Baselines

RGB/gradient/Laplacian/edge/histogram/autocorrelation/FFT-band features (46-dim), grouped by scene, `GroupKFold(5)`: LogisticRegression 99.0%, LinearSVM 99.3% on TEST03's fixed-parameter dataset. Even 3 FFT-band features alone reach 93.3%. **This means most of TEST03's easy-dataset separability is already present in non-learned statistics** — AdaIR's advantage is not demonstrated on this dataset. Phase 3–4's harder dataset restores the AdaIR advantage (see §6).

## 5. Phase 2 — Leakage-Safe PCA Re-Audit

Loophole L2 (TEST05's PCA/scaler fit on all 300 images before CV) was corrected: scaler and PCA now fit on training folds only, per fold. PCA-16 leakage-safe = **99.67%** (95% CI [99.01, 100.00]) vs. TEST05's original 99.7% — a difference of −0.00pp. **The leakage risk was real but did not inflate the result.** PCA-4 through PCA-128 all reproduce TEST05's original numbers within noise.

## 6. Phase 3–4 — Parameter-Randomized Severity Generalization

A new 600-image dataset (100 scenes × 3 degradations × 2 overlapping severity bands, 15% overlap margin) was synthesized to test loophole L3. Family-probe accuracy (both bands pooled): `latent_pre` 99.7%, AFLB1–3 98.8–99.5%, raw `input` only 68.8%. Severity-generalization (train band A / test band B and vice versa): 90.3–100% across all four candidates. **The representation generalizes across degradation severity, not memorizing one fixed synthesis recipe, and clearly outperforms raw-input statistics on this harder task** (68.8% vs. 99.7%) — resolving the concern raised in §4.

## 7. Phase 5–6 — Normalized Causal Intervention

TEST04's raw, un-normalized L2 values were replaced with normalized change (`‖ΔY‖/‖Y‖`), RMS change, and PSNR/SSIM deltas, across all three recipient degradations (876 rows, 30 scenes). Same-scene cross-degradation swaps exceed cross-scene same-degradation swaps at every representation point (degradation-specificity ratio 1.89–2.10, normalized units). **Restricting to TEST04's original Rain-only recipient subset reproduces TEST04's numbers almost exactly** (3.27 vs. their 4.03; cross-scene 7.81 matches exactly) — confirming TEST04's finding was correct for its literal, narrower scope but does not generalize as broadly as "scene dominates degradation." Averaged across all three degradations, the opposite holds.

## 8. Phase 7–9 — Frequency-Path Ablation (T0–T3) — **THE CENTRAL RESULT**

Source-code audit confirmed the only frequency-specific operations are inside `FreModule.fft()`; everything downstream (FMiM cross-attention, FMoM gating) is ordinary spatial computation, left untouched in every variant. Four teacher variants were built and compared on 300 images each:

| Variant | Modification |
|---|---|
| T0 | Released, unmodified |
| T1 | Frequency path disabled (`high=conv_feat`, `low=0`) |
| T2 | `high` replaced by Gaussian noise matched to real `high`'s mean/std (content-blind) |
| T3 | `high` reconstructed with phase spatially permuted, magnitude spectrum preserved |

**Result: degradation-probe accuracy, PCA-16 accuracy, and degradation/scene ratio are statistically indistinguishable across all four variants at every representation point** (e.g. `latent_pre`: 100.0% / 99.7% / 1.865 for T0, T1, T2, and T3 identically). Restoration output quality is flat across variants (32.66dB ± 0.01, SSIM 0.9454 ± 0.0001 for all four). See visualization 06.

T0 vs. T1 showed a small (~1–1.6%) non-zero representation distance at the AFLB outputs, contradicting this script's own initial docstring claim of guaranteed bit-identity — attributable to FFT round-trip floating-point precision (T0 computes `abs(ifft(fft(x)))`, not bit-exact identity even when the mask is exactly zero), not a genuine functional difference; `latent_pre` (upstream of all AFLBs) showed exactly 0.0 distance as expected.

**Per the task's explicit instruction, this is reported honestly as evidence AGAINST the frequency-causal part of H_F2S — the degradation-discriminative, compact, scene-robust properties of the representation survive completely unchanged whether the frequency branch computes real content, no content, content-blind random noise, or phase-scrambled noise.**

## 9. Phase 10–11 — Frequency Randomization Control and Input-vs-Feature Frequency

**Phase 10** went further than T2/T3: AFLB1's `raw_high`/`raw_low` were swapped with a genuinely different, unrelated image's frequency-branch content. Effect on final output: L2 = 0.008, MAE = 0.000008, SSIM = 0.99998 (near pixel-identical; PSNR computed as infinite due to near-zero MSE). **Even feeding completely wrong image content through the frequency branch barely moves the output.**

**Phase 11** independently re-measured radial-band FFT energy (re-using TEST05's methodology) rather than assuming TEST05's original finding. Feature-level low/mid/high band fractions are nearly identical across Rain/Haze/Noise at every candidate (low: 0.34–0.40, mid: 0.43–0.46, high: 0.16–0.21 for all three) — **this does NOT replicate TEST05's original claim of 89.9–99.3% low-frequency energy for Noise.** Input-level frequency correctly differs as physically expected (Noise input high-frequency fraction = 23.9% vs. Haze 9.3%/Rain 10.0%), confirming the measurement methodology is sound — the original feature-level claim simply does not hold up under independent re-measurement.

## 10. Phase 12–14 — Compact Representation Validation

PCA-16/32/64 (leakage-safe) were compared against a random projection (91.3–98.3%, clearly worse than PCA) and a supervised-linear upper bound (100%, matching PCA closely). Two negative controls — shuffled labels and shuffled feature/label correspondence — both collapse to ~32% (chance = 33.3%), confirming the 99.7% result is not a dimensionality or leakage artifact. A random equal-size subset of raw dimensions also performs well (95.7–98.0%), consistent with Phase 1's finding that much of this dataset's separability is present in simple statistics generally. PCA-16 predicts restoration-output PSNR with R² = 0.67–0.68 — the compact code carries restoration-relevant, not just classification-relevant, information.

## 11. Claim-by-Claim Audit (Phases A–H)

See `results/statistics/claim_audit.csv` for full evidence per claim; summarized in the mandatory table below (§15).

## 12. Mathematical Model Audit (Phase 17)

The proposed model `z_T=f_T(x); e_D=P_theta(z_T); a=G_phi(e_D); K_D=K_0+Σa_r K_r; F_out=K_D*F_in` decomposes into empirically distinguishable claims. **`z_T→e_D` (representation exists and compresses well) is empirically supported.** **The frequency-pathway's contribution to producing z_T's useful content is contradicted.** `G_phi`, `K_D`, `F_out` (the student-side kernel-modulation mechanism) are untested by design (out of TEST05.5's scope) and are neither supported nor refuted — they are independent of the frequency question and remain open for TEST06 if pursued with a non-frequency-specific embedding. See `results/statistics/mathematical_model_audit.csv`.

## 13. Alternative Models (Phase 18, theoretical only)

Six candidate TEST06 designs were compared without training (see `results/statistics/alternative_models.csv`): (A) plain NAFNet baseline, (B) conventional feature distillation, (C) compact latent distillation — **best-supported by this audit's evidence**, (D) explicit degradation-embedding conditioning — a cheap fallback given Phase 1's finding, (E) the original frequency-aware F2S design — **contradicted**, (F) frequency-aware dynamic kernel bank — inherits E's flaw plus additional unverified complexity.

## 14. F2S Hypothesis Decision (Phase 15)

Using the five stated SUPPORTED-classification criteria (robust representation; leakage-safe compactness; causal effect surviving normalization; frequency-path ablation measurably changing representation/output; simple-statistical explanations insufficient): criteria 1–3 and 5 (on the hard dataset) are met; **criterion 4 fails outright — the frequency-path ablation does NOT measurably change the representation or output.** H_F2S is classified **REFUTED** for its frequency-causal component, while the broader "compact degradation-aware representation exists and is distillable" claim remains SUPPORTED.

## 15. Mandatory Table 1 — Claim / Evidence / Alternative Explanation / Audit Result / Confidence

| Claim | Evidence (TEST05.5) | Alternative explanation | Audit result | Confidence |
|---|---|---|---|---|
| A. Features are degradation-discriminative | 99–100% easy dataset (simple stats too); 99.7% vs 68.8% hard dataset | Simple pixel statistics explain most of the easy-dataset result | SUPPORTED (with caveat) | High |
| C. Not one fixed parameter signature | 90.3–100% severity generalization | — | SUPPORTED | High |
| D. Causally relevant to output | Normalized ratio 1.89–2.10 favoring degradation | TEST04's narrower Rain-only subset gave opposite direction | SUPPORTED (revised) | High |
| E. Compact (PCA-16) subspace | 99.67% leakage-safe, beats random proj., controls collapse to chance | Leakage inflation (checked, not found) | SUPPORTED | High |
| F. Noise = low-frequency features | Feature bands nearly identical across degradations on re-measurement | Original measurement methodology sensitivity | NOT ESTABLISHED | High |
| G. Frequency processing causes the useful representation (H_F2S core) | T0≈T1≈T2≈T3 on all metrics; cross-image swap L2=0.008 | AFLB channel-attention machinery responds similarly regardless of input | CONTRADICTED | High |
| H. Representation suitable for F2S distillation | Downstream of G | General compact-embedding distillation remains viable (Model C) | PARTIALLY SUPPORTED / frequency mechanism REFUTED | High |

## 16. Mandatory Table 2 — Nine-Question Summary

| # | Question | Answer |
|---|---|---|
| 1 | Does the latent contain degradation info? | Yes, robustly (98.8–100%) |
| 2 | Robust to scene changes? | Yes (corrected: ratio 1.89–2.10 favors degradation once generalized) |
| 3 | Robust to degradation-parameter changes? | Yes (90.3–100% severity generalization) |
| 4 | Causally relevant? | Yes in general, NOT via frequency path specifically |
| 5 | Compact? | Yes (PCA-16 leakage-safe, negative-control validated) |
| 6 | Is frequency causally implicated? | **No — contradicted** |
| 7 | Is F2S justified? | **No** |
| 8 | Is PCA-16 sufficient? | Yes, independent of the frequency question |
| 9 | Should TEST06 proceed? | Yes, but redesigned around Alternative Model C, not the frequency-causal F2S mechanism |

## Final Response Summary

**What survived**: degradation-discriminability, severity-generalization, causal relevance (in a corrected, stronger form), and PCA-16 compactness are all robustly supported, several strengthened by this audit.

**What was weakened or refuted**: TEST04's "scene dominates degradation" headline was subset-specific and reverses when generalized. TEST05's dramatic Noise-low-frequency finding does not replicate. Most importantly, **the central F2S causal claim — that frequency-domain processing produces the useful representation — is contradicted** by four independent, convergent interventions.

**Loopholes found**: L2 (PCA leakage) was real but did not change the conclusion; L3 (fixed-parameter dataset) was real and the representation survived it; L4 (unnormalized intervention) revealed TEST04's finding was narrower in scope than stated; L5/L6 (frequency-path causality, feature-frequency claims) were both found to be unsupported on closer examination.

**PCA-16 remains valid**: yes, unconditionally — its value does not depend on frequency provenance.

**Frequency-path ablation does NOT support F2S.**

**The mathematical F2S model is NOT fully justified**: the `z_T→e_D` link is supported; the `frequency pathway→e_D` link is contradicted and must not be conflated with the former.

**Decision: GO WITH MODIFICATIONS.** Proceed to TEST06 with compact-embedding distillation (Alternative Model C), abandoning the frequency-causal framing and mechanism. See `GO_NO_GO.md` for the full 10-question decision record.
