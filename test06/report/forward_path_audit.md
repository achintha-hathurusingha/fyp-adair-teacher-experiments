# TEST06 Phase 0 — Forward-Path Source Audit

Read directly from `AdaIR/net/model.py` (unmodified, read-only) on devon,
`FreModule` (lines 289–370) and `AdaIR.forward()` (lines 372–470+).

## 1. Tensor trace, one AFLB instance (`self.fft()` / `FreModule.forward()`)

| # | Name | Shape (dim=48, AFLB2 example, input H×W=1024×1024) | Produced by |
|---|---|---|---|
| 1 | `inp_img` | (B, 3, 1024, 1024) | model input, unchanged, re-supplied at every AFLB call |
| 2 | `x` (resized) | (B, 3, H_feat, W_feat) — `F.interpolate(inp_img, (H,W))`, H,W taken from `y` | `FreModule.forward()`, bilinear resize of the ORIGINAL image to the current decoder stage's spatial size |
| 3 | `x` (post-conv1) | (B, dim_AFLB, H_feat, W_feat) | `self.conv1(x)`, a 3×3 conv, `in_dim=3 → dim_AFLB` |
| 4 | `threshold` (α,β) | (B, 2, 1, 1) | `rate_conv(adaptive_avg_pool2d(x,1)).sigmoid()` — two learned scalars per image, both in (0,1) |
| 5 | `mask` | (B, dim_AFLB, H_feat, W_feat), binary | box of half-height `h_=int((H_feat//128)*threshold[:,0])`, half-width `w_=int((W_feat//128)*threshold[:,1])`, centered |
| 6 | `fft` (shifted) | (B, dim_AFLB, H_feat, W_feat), complex64 | `fftshift(fft2(x, norm='forward'))` |
| 7 | `fft_high` | same shape, complex64 | `fft * (1-mask)` |
| 8 | `raw_high` | (B, dim_AFLB, H_feat, W_feat), real | `abs(ifft2(unshift(fft_high)))` |
| 9 | `fft_low` | same shape, complex64 | `fft * mask` |
| 10 | `raw_low` | (B, dim_AFLB, H_feat, W_feat), real | `abs(ifft2(unshift(fft_low)))` |
| 11 | `high_feature` (mined) | (B, dim_AFLB, H_feat, W_feat) | `channel_cross_l(raw_high, y)` — FMiM cross-attention vs. spatial branch `y` |
| 12 | `low_feature` (mined) | (B, dim_AFLB, H_feat, W_feat) | `channel_cross_h(raw_low, y)` |
| 13 | `agg` | (B, dim_AFLB, H_feat, W_feat) | `frequency_refine(low_feature, high_feature)` — FMoM (`FreRefine`: H-L spatial gate + L-H channel gate) |
| 14 | `out` (cross-agg) | (B, dim_AFLB, H_feat, W_feat) | `channel_cross_agg(y, agg)` |
| 15 | AFLB output | (B, dim_AFLB, H_feat, W_feat) | `out*para1 + y*para2` — residual mix, `para1` init 0, `para2` init 1 |

## 2. Mask-activation arithmetic (exact, from source)

```
h_ = int((H_feat // 128) * threshold_alpha)   # threshold_alpha in (0,1), strictly < 1
w_ = int((W_feat // 128) * threshold_beta)
```

`int()` truncates toward zero. Since `threshold < 1` strictly (sigmoid never
reaches 1), `h_ >= 1` requires `(H_feat // 128) * threshold >= 1`, which is
**mathematically impossible unless `H_feat // 128 >= 2`**, i.e. **`H_feat >=
256`** — and even then requires `threshold >= 0.5`. `H_feat < 256` makes
`h_ = 0` **guaranteed**, independent of the learned threshold value. This
reconfirms TEST01's degenerate-mask finding as a mathematical certainty
(not just an empirical observation) for any feature map narrower than 256px
on the relevant axis.

## 3. Per-AFLB feature resolution vs. input resolution

`AFLB.forward(x, y)` resizes `x` (the raw image) to match `y`'s spatial size
— `y` is the SPATIAL branch at that specific decoder depth, not the input
image. From `AdaIR.forward()`'s downsampling structure (`down1_2`,
`down2_3`, `down3_4`, each halving H,W once):

| AFLB | Spatial branch (`y`) | Feature H,W (relative to input) | dim_AFLB (base dim=48) | Min input size for `H_feat>=256` |
|---|---|---|---|---|
| AFLB1 (`fre1`) | `latent` (bottleneck, after 3 downsamples) | input / 8 | 384 | input >= 2048 |
| AFLB2 (`fre2`) | `out_dec_level3` (after 1 upsample from bottleneck) | input / 4 | 192 | input >= 1024 |
| AFLB3 (`fre3`) | `out_dec_level2` (after 2 upsamples) | input / 2 | 96 | input >= 512 |

**This means the three AFLBs require very different input resolutions to
activate**, and the requested 06-A sweep (128–1536px) is expected to reach
AFLB3's activation threshold and plausibly AFLB2's, but very likely NOT
AFLB1's (which needs input >= 2048px, beyond the requested sweep's upper
bound of 1536). This is a testable prediction, not an assumption — 06-A's
empirical sweep will confirm or refute it per-AFLB, and the sweep range
should be treated as potentially insufficient for AFLB1 specifically.

## 4. Verification that TEST05.5's intervention was scoped correctly

TEST05.5's `frequency_variants.py` / `test01/model_variants.py` `_fft_released`,
`_fft_no_frequency`, T2 (`_fft_t2_matched_random`), T3 (`_fft_t3_phase_shuffle`)
all replace ONLY the body of `fft()` (tensor items 3–10 above). Comparing
line-by-line against the source read here:

- `_fft_released` reproduces items 3–10 exactly (conv1 → mask via
  `(h//n)*threshold` → fft/shift → high/low split → ifft/unshift → abs) —
  **verified identical**, including the `int()` truncation order.
- `_instrumented_forward` (used by all T0–T3 variants) calls
  `channel_cross_l`, `channel_cross_h`, `frequency_refine` (FreRefine/FMoM),
  and `channel_cross_agg` UNCHANGED from the released module methods (items
  11–15) — **confirmed no accidental modification of downstream FMiM/FMoM
  architecture**. Only the `(high, low)` tensors handed to
  `channel_cross_l`/`channel_cross_h` differ between variants.

**Audit passes.** TEST05.5's intervention was correctly scoped to the
frequency-specific computation only, as intended.

## 5. Implication for TEST06 design

Given §3, 06-A's resolution sweep must report per-AFLB activation
separately (not a single "the mask activates at resolution R" number) —
the three AFLBs are expected to activate at substantially different input
resolutions, and 06-E's same-scene intervention dataset should be sized to
guarantee activation at whichever AFLB(s) the sweep confirms are reachable
within a practical resolution/memory budget (almost certainly AFLB3, likely
AFLB2, probably not AFLB1 without a >2048px dataset).
