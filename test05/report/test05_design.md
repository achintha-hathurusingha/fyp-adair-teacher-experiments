# TEST05 Design — Degradation-Specific Representation Discovery

Read (read-only, not modified): `test01/docs/adair_source_audit.md`,
`test01/docs/adair_ablation_report.md`, `test02/report/{source_audit,
test02_report}.md`, `test03/report/{test03_design,test03_report,
synthetic_data_validation}.md`, `test04/report/{forward_graph_audit,
test04_report}.md`, plus their underlying CSV results.

## 1. What TEST01 established

The released AdaIR mask (`FreModule.fft()`) computes its low-frequency
box half-width as `int((h//128)*alpha)` — floor-dividing the feature map
size by a hardcoded `n=128` before multiplying by the learned `alpha`.
At every benchmark resolution tested, `h//128` floors to 0 (or 1, which
then still truncates to 0 given `alpha≈0.5`), so **`raw_low` is exactly
zero** for every image, every AFLB — verified by a from-scratch
forward-pass trace with 5 independent checks (mask complement, energy
conservation, manual reconstruction, `mask.unique()`, floor-division
arithmetic), all passing exactly. A resolution sweep confirmed the
mechanism *can* activate at ≥640-1024px inputs, far above what AdaIR is
benchmarked at. **Design consequence for TEST05**: `raw_low` is included
only as a negative control (Phase 3); no effort is spent making it useful.

## 2. What TEST02 established

Linear probes on 41 pooled (GAP+GMP) intermediate representations,
extracted via non-intrusive forward hooks, classify Rain/Haze/Noise at
71.7% (input) rising to 100.0% (latent) and collapsing to 54.7% (output).
**Confound**: Rain=Rain100L, Haze=SOTS-outdoor, Noise=BSD68 — three
disjoint source datasets, so this trajectory could partly reflect
dataset-domain identity rather than degradation type per se.

## 3. What TEST03 established

The same 41-feature trajectory, reproduced on 100 clean scenes each
synthesized into Rain/Haze/Noise (deterministic, documented streak/haze/
noise models), with `GroupKFold(scene_id)` eliminating any same-scene
leakage. Trajectory survives and strengthens: 66.7% (input) → 100.0%
(most internal stages, zero variance) → 37.0% (output). Same-scene
cross-degradation distances exceed cross-scene same-degradation distances
by up to 2.6× internally (but the reverse — 0.30× — at output).
`raw_low` independently reconfirmed exactly zero (4th confirmation
counting TEST01/02/03/04). **Design consequence**: TEST05 reuses this
exact 100-scene, 300-image dataset (`test03/results/manifest/
scene_manifest.csv`, `test03/data/`) — no new synthesis.

## 4. What TEST04 established

A manual, verified-bit-exact replica of `AdaIR.forward()`
(`test04/src/intervention.py::manual_forward`) allows any intermediate
tensor to be substituted mid-forward-pass. Self-swap gate passed exactly
(0.0 diff, 300/300 scenes). Cross-degradation swaps at `latent_pre`/
`aflb1/2/3_out` produce real, structured output changes (mean L2 14.17 →
53.94, increasing with depth) that exceed random/zero/mean-tensor
controls (L2 1.2-2.2) — ruling out "any perturbation causes this."
**But** cross-scene same-degradation swaps (L2 7.81) exceed the primary
same-scene cross-degradation swaps (L2 4.03, matched subset) — so the
representation's causal influence on output is *not* proven to be
specific to degradation over scene/content. Classified as "MODERATE"
causal evidence, not "STRONG." Skip-connection closure (latent+all skips)
amplified the effect 9× (138.6 vs 15.2), confirming the forward-graph
audit's prediction that skip connections carry substantial un-intervened
recipient identity.

## 5. Why TEST05 is necessary

TEST02/03 measure *discriminability* (can a probe tell Rain from Haze
from Noise). TEST04 measures *causal effect* at the level of whole
tensors, and found that effect is entangled with scene/content — a whole
representation is not "purely" a degradation signal. Neither prior test
asks the operationally critical question for a distillation project:
**is there a *smaller part* of these representations — specific
channels, a compact projection — that is more specifically
degradation-related and less scene-entangled than the full tensor?**
That sub-representation, if it exists, is the actual object the teacher
should transfer to a NAFNet student — full tensors are both wasteful
(NPU cost) and, per TEST04, not cleanly degradation-specific to begin
with. TEST05 searches for it directly, at the channel level, using both
correlational (probe accuracy, distance ratios) and causal
(TEST04-style intervention restricted to the candidate subset) evidence,
plus an independent frequency-domain characterization of the candidate
tensors themselves (not the broken AdaIR mask).

## 6. Working definition: "degradation-specific representation"

A representation (full tensor, channel subset, or compact projection)
`F_deg` is *more degradation-specific* than a baseline `F_full` to the
extent that, relative to `F_full`, it simultaneously:

1. **Retains** high degradation discriminability (grouped-CV linear probe
   accuracy comparable to `F_full`'s, not necessarily identical).
2. **Reduces** the scene/content signal, measured two ways: (a) a lower
   `D_scene` (same-degradation, cross-scene distance) relative to
   `D_degradation`, i.e. a *higher* `degradation_scene_ratio`; (b) when
   swapped causally (TEST04-style), a same-scene cross-degradation swap
   producing a *larger* output change than a matched cross-scene
   same-degradation swap — the one relationship the *full* latent tensor
   failed to satisfy in TEST04.
3. **Retains** a measurable fraction of the full tensor's causal effect
   on output when swapped (not necessarily 100% — a useful compact signal
   trades some effect size for compactness and specificity).
4. Is **smaller** than the full tensor (fewer channels, lower
   dimensionality, or both) — the explicit compactness requirement for
   NPU deployment.

No single number captures this; TEST05 reports every criterion
separately and only combines them into an explicit, sensitivity-tested
composite score at the very end (Phase 18), never allowing raw
classification accuracy alone to decide the ranking — per the task's
explicit instruction and the lesson learned from TEST04 (a
high-classification-accuracy representation, the full latent, turned out
not to be cleanly degradation-specific under causal testing).
