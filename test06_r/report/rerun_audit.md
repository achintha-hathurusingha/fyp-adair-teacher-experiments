# TEST06-R Phase 0 — Verification of Original TEST06

Read (read-only, not modified): `test06/report/test06_report.md`,
`test06/report/forward_path_audit.md`, `test06/report/GO_NO_GO.md`,
`test06/results/environment.txt`, `test06/src/causal_intervention_06e.py`,
`test06/src/resolution_sweep.py`.

## Checklist

1. **Checkpoint SHA256**: `f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb`, matching the value recorded for TEST01–05.5. TEST06-R will re-verify this on its own `write_environment.py` run and confirm identity.
2. **AdaIR source git SHA**: `ccb8b98e49614e07badd0641e5163fa7635c2f02`. Not touched by TEST06-R.
3. **Intervention point**: `causal_intervention_06e.py`'s `_forward_with_aflb3_override()` swaps only the `(raw_high, raw_low)` tensors returned by the unmodified `_fft_released()` (imported read-only from `test01/scripts/model_variants.py`), computed on `net.fre3` (AFLB3). Everything downstream — `channel_cross_l`, `channel_cross_h`, `frequency_refine` (FMoM), `channel_cross_agg`, the `para1`/`para2` residual mix — is the real, untouched module method, called exactly as `FreModule.forward()` does. This matches the source audit performed in the original `test06/report/forward_path_audit.md`. **Confirmed correct.**
4. **Resolution**: 1024×1024, confirmed non-degenerate at AFLB3 (feature resolution 512×512, well above the 384px first-activation point found in 06-A). Matches.
5. **25 scenes**: `test06/results/frequency_intervention/scene_manifest.csv`, built from DIV2K validation images 8–32 (disjoint from the 0–7 used in the resolution sweep). TEST06-R reuses this exact manifest and these exact image files — **not regenerated**.
6. **Degradation synthesis**: parameter-randomized Rain/Haze/Noise via `test06/src/degradation_synthesis.py`, seeded deterministically per `(scene_id, degradation)`. TEST06-R reads the manifest's already-synthesized `rain_path`/`haze_path`/`noise_path` files directly — the same PNGs on disk, not re-synthesized.

## Implementation Correctness Review

The original intervention is verified correct by its own internal evidence: **Phase 8's self-swap control (donor==recipient) produced an exact 0.0 output difference**, which is only possible if the override mechanism is applying the intended tensor at the intended point and nothing else changed. This is preserved and re-verified independently in TEST06-R's own Phase 4.

The two weaknesses identified for this corrected re-run are methodological (unbalanced N between primary and controls; mean-only comparison with no paired/bootstrap statistics; no internal-layer tracing) — **not implementation bugs**. No code defect was found that would invalidate the original 432-configuration resolution sweep, so per the task's explicit instruction, **the resolution sweep is NOT re-run**.

## Data-Quality Finding (discovered during TEST06-R execution)

While computing Control D's global-mean tensor (which requires stacking all 75 real `raw_high`/`raw_low` tensors across every scene), a shape mismatch was found: **`scene_021`'s clean/rain/haze/noise images are all `1024×104`, not `1024×1024`.** This traces to the original `test06/src/build_06e_dataset.py`'s `center_crop()` function, which slices `img[top:top+size, left:left+size]` without first checking the source DIV2K image is actually ≥1024px in both dimensions — unlike `resolution_sweep.py`'s `center_crop_or_pad()`, which explicitly returns `None` (and is skipped) when the source is too small. The source DIV2K image for scene_021 (validation index ~29) is evidently narrower than 1024px on one axis, and the crop silently returned a truncated 104px-tall image instead of failing loudly.

**Why the original TEST06 never surfaced this**: its Phase 8 self-swap and Phase 9 primary cross-degradation swap only ever compare tensors *within* the same scene (all four of scene_021's degraded variants share the same malformed 1024×104 shape, so within-scene operations proceeded consistently, just on an oddly-shaped image). Its Phase 10 cross-scene control explicitly guarded with `if d["raw_high"].shape == r["raw_high"].shape: ... else skip` — silently excluding any cross-scene pairing involving scene_021 without flagging it. TEST06-R's global-mean computation (new in this re-run) has no such guard and surfaced the issue immediately.

**Resolution for TEST06-R**: per the explicit instruction not to modify or regenerate `test06/`'s dataset, **scene_021 is excluded from all TEST06-R balanced analysis** (N=24 scenes, not 25). This is a data-quality exclusion, documented here, not a silent workaround — and it does not require touching any file under `test06/`. All totals in this re-run (primary swaps, controls, propagation traces) are scaled accordingly: 24 scenes × 3 recipients × 2 donors = 144 primary swaps (not 150), 24×3=72 self-swap checks, 24×3=72 unique control computations ×4 = 288 control forward passes, expanded to 144 rows per control type.

## Conclusion

The original TEST06 intervention implementation is **valid and correctly scoped**. TEST06-R proceeds to tighten the statistical inference (balanced N, paired tests, bootstrap CIs) and add internal-propagation tracing (Phases 10–13, new in this re-run) — it does not re-derive facts already established by TEST06's resolution sweep or re-litigate whether the intervention mechanism itself works.
