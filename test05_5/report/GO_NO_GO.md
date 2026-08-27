# TEST05.5 — GO / NO-GO Decision

## 1. Does the teacher's internal representation contain degradation-relevant information?

**Yes, robustly.** `latent_pre` and every AFLB output classify Rain/Haze/Noise at 98.8-100% under grouped (scene-disjoint) cross-validation on both TEST03's original dataset and TEST05.5's new parameter-randomized, overlapping-severity dataset. This survives a strict leakage-safe re-audit (Phase 2) and generalizes across degradation severity (Phase 3-4, 90-100% train-on-one-band/test-on-other).

## 2. Is this robust to scene/content changes (not just a scene-identity shortcut)?

**Yes, with a correction to a prior claim.** TEST04 originally reported the representation is MORE sensitive to scene change than degradation change, but that number was restricted to a Rain-recipient subset. TEST05.5's normalized, full-3-degradation reproduction (Phase 5-6) finds the OPPOSITE holds when generalized: degradation-specificity ratio 1.89-2.10 (>1) at every representation point — same-scene cross-degradation swaps move the output more than cross-scene same-degradation swaps do. TEST04's original number was correct for its literal scope but should not have been generalized as broadly as it was.

## 3. Is this robust to degradation-parameter/severity changes (not one fixed synthesis recipe)?

**Yes.** Phase 3-4's parameter-randomized, overlapping-severity-band dataset (loophole L3) shows family-probe accuracy of 98.8-99.7% and severity-generalization of 90.3-100% across all four representation candidates. Not a fixed-parameter-signature artifact.

## 4. Is the representation causally relevant to the restoration output (not just correlated)?

**Yes, in a general sense — but not via the frequency pathway specifically.** TEST04/TEST05.5 causal swaps confirm `latent_pre` and AFLB outputs causally shape the output (normalized intervention exceeds random/zero/mean controls). However, Phase 7-9/10 show this causal relevance does NOT depend on the frequency-specific computation inside `FreModule.fft()` — disabling it, randomizing it, phase-scrambling it, or feeding it a different image's content all leave the output and representation quality essentially unchanged.

## 5. Is the representation compact (does a small subspace suffice)?

**Yes, robustly.** PCA-16 (leakage-safe) reaches 99.67% degradation accuracy, matches TEST05's original (leaky) number almost exactly (no inflation found), beats random projection (91.3%), matches a supervised-linear upper bound (100%), and both negative controls (shuffled labels, shuffled correspondence) collapse to chance (~32%). R²=0.67-0.68 for predicting restoration-output PSNR from the 16-dim code.

## 6. Is the useful representation causally implicated by the teacher's frequency-aware processing? (H_F2S's central claim)

**No — contradicted.** This is the central, most important finding of TEST05.5. Four independent interventions converge on the same answer:
- **T0 vs T1** (frequency path fully disabled): degradation accuracy, PCA-16 accuracy, and degradation/scene ratio are identical to T0 at every representation point; restoration PSNR/SSIM changes by <0.01dB.
- **T0 vs T2** (frequency content replaced by content-blind matched-random noise): same — all metrics unchanged.
- **T0 vs T3** (phase-scrambled, magnitude-preserved): same — all metrics unchanged.
- **Phase 10** (frequency branch fed a completely different, unrelated image's content): final output changes by L2=0.008, SSIM=0.99999 — essentially no effect.

This corroborates and substantially extends TEST01's original finding (<0.004dB output-quality change from disabling frequency processing) to the representation level. Additionally, Phase 11's independent re-measurement does not replicate TEST05's original claim that internal features show a dramatic degradation-dependent frequency-band pattern (89.9-99.3% low-frequency for Noise) — feature-level frequency-band fractions are nearly identical across all three degradations. The mechanism producing AdaIR's degradation-aware representation is almost certainly the AFLB's channel-attention/gating machinery (FMiM/FMoM) responding to whatever tensor it receives, not genuine frequency-domain reasoning — at least at this benchmark resolution, where the learned mask is degenerate (h_=w_=0, established in TEST01 and reconfirmed independently 6+ times across TEST01-05.5).

## 7. Is Frequency-to-Spatial (F2S) distillation, as originally conceived, scientifically justified?

**No.** The causal link that gives F2S its name — "frequency-domain processing produces the useful degradation embedding" — is directly contradicted by Phase 7-10. Building explicit machinery to distill "frequency-derived" information that is not, in fact, meaningfully frequency-derived would be a misleading and wasted engineering effort.

## 8. Does the PCA-16 compact representation remain valid as a distillation target?

**Yes**, independent of the frequency question. Its validity (leakage-safe, negative-control-tested, restoration-relevant) does not depend on how `latent_pre` was produced internally — only on the fact that it IS degradation-informative and compact, which Phase 2/12-14 confirm robustly.

## 9. Should TEST06 proceed, and if so, on what basis?

**Not as originally specified.** TEST06 should NOT build a frequency-causal F2S module. It SHOULD proceed with a compact-embedding-based distillation approach (Alternative Model C: NAFNet + compact latent distillation), which is well-supported by claims A, C, D, and E of the claim audit, without any dependency on or claim about frequency-domain provenance. Concretely, TEST06 should:
1. Distill a small (≈16-dim) learned embedding of AdaIR's `latent_pre` (or an AFLB output) into a NAFNet student, trained end-to-end rather than via post-hoc PCA.
2. NOT frame this as "frequency-to-spatial" — rename/re-scope the mechanism to reflect what is actually being distilled (a general degradation-aware compact code).
3. Test whether the learned compact code remains leakage-safe, scene-robust, and severity-generalizing once trained end-to-end (Phase 2/3-4's post-hoc findings may not automatically transfer to an end-to-end-learned projection).
4. If frequency-domain processing is still of interest, it should only be revisited at a resolution regime where AdaIR's learned mask is NOT degenerate (TEST01's resolution sweep found the mechanism activates at ≥640-1024px) — a fundamentally different, higher-resolution experimental setup, not a continuation of the current benchmark-resolution pipeline.

## 10. Final Decision

**GO WITH MODIFICATIONS.**

- **GO** on distilling AdaIR's compact internal representation into a NAFNet student — this is well-supported across 5 independent claims (A, C, D, E, and the general causal-relevance finding in D).
- **MODIFICATION REQUIRED**: abandon the frequency-causal "F2S" framing and mechanism (Model E/F in Alternative_Models). Proceed instead with Alternative Model C (compact latent distillation) as the primary TEST06 design, informed by Model D (explicit degradation embedding) as a cheaper fallback/baseline given Phase 1's finding that simple statistics alone are highly discriminative on easy degradation sets.
- The mathematical model's `z_T → e_D` link is empirically supported; its `frequency pathway → e_D` justification is not, and must not be conflated with the former in any TEST06 design document.
