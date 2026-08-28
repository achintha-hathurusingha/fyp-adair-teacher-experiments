# TEST20 — do the AFLB mask values actually change with degradation type, on Himeth's REPAIRED checkpoint?

## Why the released checkpoint can't answer this

TEST01/06/18 found the released `adair3d.ckpt`'s mask is degenerate: active
fraction = 0.0 for every degradation, at every practical resolution
(`h_ = (h // 128 * threshold).int()` floors to 0 whenever the feature map is
under 256px, which is every AFLB in this project's own testing). There is
nothing to "track by degradation" on a mask that never activates at all.

Himeth's `finetune/freq_fix.py` independently found the same root cause
(confirmed from his own docstring — the resolution floor, plus a second,
separate defect: the mask is built via a non-differentiable hard index-write,
so `rate_conv` never received gradient and stayed at its random
initialisation regardless of input) and repaired it: `alpha, beta` are now
read as a *fraction* of the half-spectrum (resolution-independent by
construction) and the mask edge is a soft sigmoid, differentiable in
alpha/beta. His own mask-ablation (`runs/finetune/results/mask_with_without/`)
already showed the repaired mask does real, statistically significant work,
concentrated in dehaze. This asks the natural follow-up: does it actually
vary *by degradation type*, the thing its name promises?

## Method

`test20/scripts/values_by_degradation.py`. Loads the stock `AdaIR` class,
monkey-patches it with Himeth's own `apply_freq_fix(mode="soft", tau=0.05)`
(matching `C_full_soft`'s own training config exactly — no re-derivation of
the repair), loads `runs/finetune/C_full_soft/final.pt` (0 missing, 0
unexpected keys). His patched `fft()` already stores `{alpha, beta,
coverage}` on every forward (`_freqfix_last`) — this script just reads that,
plus `para1`/`para2` (the residual-gate weights) directly off each FreModule,
for Rain/Haze/Noise inputs, at all 3 AFLB positions.

## Results

| AFLB | Rain coverage | Haze coverage | Noise coverage | spread |
|---|---:|---:|---:|---:|
| AFLB1 (deepest) | 0.381 | **0.461** | 0.418 | 0.080 |
| AFLB2 | **0.207** | 0.179 | 0.193 | 0.028 |
| AFLB3 (shallowest) | 0.275 | 0.291 | 0.285 | 0.016 |

(full alpha/beta/coverage/para1/para2 per AFLB x degradation in
`results/values_by_degradation.csv`)

**The mask genuinely varies by degradation now** — coverage ranges from 0.18
to 0.46 across conditions, nowhere near the released checkpoint's flat 0.0.
The repair fixed what TEST01/06/18 found broken.

**But the pattern is not simple, and not always "haze-favoring":**
- **AFLB1**: Haze gets the widest mask (0.461), matching Himeth's own
  mask-ablation finding that dehaze is where the frequency split earns its
  keep physically (haze is a smooth low-frequency veil).
- **AFLB2**: the ranking *inverts* — Rain gets the widest mask (0.207), Haze
  the narrowest (0.179).
- **AFLB3**: the three degradations are nearly indistinguishable (spread
  0.016, an order of magnitude tighter than AFLB1).

**para1/para2** (the residual gate deciding how much of the frequency-refined
output vs. the bypass to keep) are fixed per-AFLB weights, not
input-dependent — but their relative magnitude is itself a finding: `fre1`
gives para2/para1 ~= -1.96 (mean para1=-0.0033, para2=0.0065), a roughly
2x imbalance. TEST06-R measured this same ratio on the released checkpoint
at ~200x (para1 mean=-0.000155, para2 mean=0.0297). The repair did not just
fix the mask — it substantially rebalanced how much the network trusts the
frequency branch relative to the bypass.

## Consequence

Contrary to the released checkpoint, degradation-dependent mask behaviour is
real once the resolution/gradient defects are fixed — this is genuine new
evidence that AFLB's *design intent* is achievable, just not realised in the
released weights. It does not overturn TEST05.5/TEST19's separate finding
that degradation identity is already strongly separable upstream at
`latent_pre` (99.0% leave-scene-out accuracy, TEST19) — that separation
happens before any AFLB runs at all. What this adds: even *given* a working
frequency-adaptive mechanism downstream, its behaviour is small in magnitude
(Himeth's own ablation: +0.056dB overall) and its degradation-dependence is
uneven across depth (strong and sensible at AFLB1, inverted at AFLB2, nearly
absent at AFLB3) — not the clean, physically-obvious pattern a "frequency
mining and modulation" story would predict at every stage.
