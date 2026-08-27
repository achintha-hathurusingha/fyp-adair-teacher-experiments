# TEST06 — GO / NO-GO Decision

## Central Question

> "When AdaIR's frequency pathway actually has enough spatial resolution to become non-degenerate, does it begin to matter for restoration, and is that influence degradation-specific?"

## Answer, in one line

The mask becomes genuinely non-degenerate (AFLB3, input ≥768px) — but even there, a controlled causal swap of the frequency-path tensor produces an effect **indistinguishable from swapping in zero, random noise, or the per-tensor mean**. Resolution was not the missing variable.

## Actual numbers

### 06-A — mask activation vs. resolution

| AFLB | Feature/input ratio | First activation (input px) | Feature res at first activation | Status |
|---|---|---|---|---|
| AFLB1 | 1/8 | not reached (needs ≥2048px) | — | NEVER ACTIVATES in 128–1536px range |
| AFLB2 | 1/4 | not reached (β≈0.497 just misses 0.5 threshold needed at feat=256–320) | — | NEVER ACTIVATES in tested range |
| AFLB3 | 1/2 | **768px** | 384px | **ACTIVATES** — 116/412 grid configs (28%) |

At AFLB3's first activation: `mask_active_fraction = 0.000027` (a 1-pixel half-width box), yet `raw_low_energy_fraction = 0.822` — a tiny mask captures 82% of the low+high energy split, confirming this is a real, non-trivial activation of the branch, not numerical noise.

### Restoration sanity (Phase 2)

No NaN/Inf observed across the 432 successfully-run sweep configurations. 16 of 448 attempted grid configs (3.6%) hit CUDA OOM at the largest resolutions (1280px+) and were skipped, not silently treated as valid. PSNR/SSIM remained in normal ranges throughout (native benchmark images: 17.5–43.7dB; controlled grid: broadly consistent).

### 06-E — the primary causal test (25 scenes, 1024×1024, AFLB3 confirmed active)

- **Phase 8 self-swap control**: max L2 = 0.00000000 — the intervention mechanism is verified correct (recipient==donor produces bit-identical output).
- **Phase 9 same-scene cross-degradation swap** (150 swaps, the primary causal test): mean normalized L2 = **0.000013**
- **Phase 10 controls**:
  - cross-scene, same-degradation: 0.000011
  - random, distribution-matched: 0.000012
  - zero tensor: 0.000012
  - mean tensor: 0.000012
- **Phase 11 donor-behavior**: swapped output moved closer to the donor's own normal output than to the recipient's normal output in **0.0%** of 150 swaps.

The primary effect (0.000013) is **not distinguishable from any control**, including the "arbitrary perturbation" controls (zero/random/mean at 0.000012). A real causal effect would show the primary condition clearly exceeding all controls, as TEST04/TEST05.5's spatial-tensor interventions did. Here it does not.

## Gate Decisions Applied

- **Phase 4 gate** (06-A → 06-E): activation found at AFLB3 → PROCEED to 06-E. ✅ Applied correctly.
- **Phase 17 gate** (06-E → 06-B/C/D): 06-E is null → **DO NOT proceed** to frequency-band sensitivity (06-B), degradation-specific sensitivity (06-C), or compact frequency signature (06-D). Per the task's explicit instruction, these phases were **not run** — this is Case B from the decision matrix, not an oversight.

## Case Classification (per task's Phase 17 framework)

**CASE B — MASK ACTIVATES BUT 06-E IS NULL.**

> Frequency becomes computationally non-degenerate but does not measurably affect restoration under the intervention. Do NOT proceed with frequency-conditioned student design. Use compact latent distillation instead.

## Claim Discipline Check

- OBSERVATION: the AFLB3 mask activates at input ≥768px (confirmed, 116/412 configs).
- OBSERVATION: the frequency intervention at that resolution changes output by 0.000013 normalized L2, statistically indistinguishable from zero/random/mean controls (0.000012).
- INFERENCE: the frequency path is NOT functionally relevant at this resolution, despite being computationally non-degenerate.
- We do NOT claim "frequency is useless" (per the task's mandatory claim discipline) — only that, at the resolutions tested (up to 1536px) and for AFLB3 specifically (the only AFLB confirmed active), no causal effect above control level was detected. AFLB1/AFLB2 were never confirmed active in this range, so their resolution-dependent behavior remains genuinely untested, not refuted.

## Answers to the 15 Final-Response Questions

1. First mask activation resolution at each AFLB: AFLB1 not reached; AFLB2 not reached (very close, β≈0.497 vs 0.5 needed); AFLB3 = 768px input.
2. Corresponding feature-map resolutions: AFLB3 = 384×384 at first activation.
3. Mask activation fraction: 0.000027 at first activation (AFLB3), growing to ~0.000039 by 1280px.
4. Low/high spectral energy: raw_low_energy_fraction = 0.82 at first activation (AFLB3) — tiny mask, large energy share.
5. Restoration PSNR/SSIM at tested resolutions: no NaN/Inf; PSNR/SSIM stayed in normal restoration ranges throughout (see restoration_sanity.csv).
6. Self-swap error: 0.00000000 (exact).
7. Cross-degradation frequency-swap effect: normalized L2 = 0.000013 (mean, 150 swaps).
8. Cross-scene control effect: normalized L2 = 0.000011.
9. Random/zero/mean controls: 0.000012 / 0.000012 / 0.000012.
10. Normalized causal effect: 0.000013 — not distinguishable from controls (differences are within the same order of magnitude, effectively noise-level).
11. Donor-behavior similarity: 0% of swaps moved output toward donor's behavior.
12. Does frequency have measurable causal influence at the non-degenerate resolution? **No.**
13. Is that influence degradation-specific? **N/A — no influence to be degradation-specific.**
14. Should 06-B/C/D be run? **No**, per the Phase 4/17 gate — correctly skipped.
15. GO/NO-GO for a frequency-derived student signal: **NO-GO.**

## Final Decision

**NO-GO for a frequency-derived spatial-kernel student signal.**

Resolution was the hypothesis TEST06 was built to test, and it has now been tested directly, not assumed away: even where the frequency-adaptive mask is confirmed mathematically non-degenerate, the frequency path remains causally inert relative to controls. Combined with TEST05.5's findings at the (degenerate) benchmark resolution, this closes the loop: across every resolution regime tested from 128px to 1536px input, and across two independent teacher-checkpoint-based methodologies (T0–T3 ablation at benchmark resolution in TEST05.5; direct causal swap at confirmed-active resolution in TEST06), no causal role for AdaIR's frequency-specific computation has been found.

**Recommendation for TEST07+/student design**: proceed with compact-embedding-based distillation (`z_T → e_D`, as validated in TEST05.5), not a frequency-response-derived signal (`F → q_F`). The `q_F` bridge proposed as the more "principled" alternative was the entire point of building TEST06 to test — and it did not survive the test.
