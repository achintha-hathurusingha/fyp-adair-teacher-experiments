# AdaIR Source-Code Audit

Scope: everything needed to design a faithful, isolated ablation of the AFLB frequency
mechanism. Written from direct inspection of the vendored source
(`teacher-experiments/AdaIR`, `git rev-parse HEAD` = `ccb8b98e49614e07badd0641e5163fa7635c2f02`,
cloned from `https://github.com/c-yn/AdaIR`) and the paper (`2403.14614v1.pdf`, arXiv:2403.14614,
ICLR 2025). Nothing below is inferred beyond what is directly visible in the source or the paper text.

## 1. Relevant source files

| File | Contents |
|---|---|
| `net/model.py` | Entire model: `LayerNorm`, `FeedForward`, `Attention` (MDTA), `TransformerBlock`, `Chanel_Cross_Attention` (CA), `SpatialGate` (H-L unit), `ChannelGate` (L-H unit), `FreRefine` (FMoM), `FreModule` (AFLB), `AdaIR` (top-level) |
| `test.py` | Evaluation entry point, PSNR/SSIM computation, checkpoint loading |
| `utils/dataset_utils.py` | `DenoiseTestDataset`, `DerainDehazeDataset` — image loading, `crop_img(base=16)`, Gaussian noise synthesis |
| `utils/val_utils.py` | `compute_psnr_ssim` |
| `utils/image_utils.py` | `crop_img` |
| `options.py` | CLI args (`de_type`, `patch_size`, etc.) — not used directly here since we drive the model ourselves |

## 2. Class/function map

```
AdaIR(nn.Module)
 ├─ patch_embed: OverlapPatchEmbed (3x3 conv, in=3 -> dim=48)
 ├─ encoder_level1..3, latent, decoder_level1..3, refinement: nn.Sequential[TransformerBlock]
 ├─ down1_2/down2_3/down3_4: Downsample (conv + PixelUnshuffle)
 ├─ up4_3/up3_2/up2_1: Upsample (conv + PixelShuffle)
 ├─ reduce_chan_level2/3: 1x1 conv (skip-connection channel reduction)
 ├─ fre1, fre2, fre3: FreModule   <-- present only if decoder=True (default)
 └─ output: 3x3 conv, dim*2 -> 3, with global residual (+ inp_img)

FreModule(nn.Module)                          # == AFLB in the paper
 ├─ conv, conv1: 3x3 conv, in_dim=3 -> dim     # conv is UNUSED in forward() (see 4.1)
 ├─ score_gen: 7x7 conv, 2->2                  # UNUSED in forward() (see 4.1)
 ├─ para1, para2: learnable scalars (per-channel), residual gate
 ├─ channel_cross_l, channel_cross_h, channel_cross_agg: Chanel_Cross_Attention  # == CA / Eq.(2)-(3)
 ├─ frequency_refine: FreRefine                # == FMoM
 ├─ rate_conv: Sequential[1x1 conv(dim->dim/8), GELU, 1x1 conv(dim/8->2)]  # == MGB, Eq.(1)
 ├─ .fft(x, n=128): domain transform + MGB + mask + FFT split             # == FMiM steps 1-2
 └─ .forward(x, y): FMiM step 3 (CA mining) + FMoM + final CA + residual gate

FreRefine(nn.Module)                          # == FMoM
 ├─ SpatialGate: 7x7 conv(2->1) + sigmoid on [max,mean] over channel      # == H-L unit, Eq.(5)-(6)
 ├─ ChannelGate: MLP(avgpool+maxpool, dim->dim/16->dim) + sigmoid          # == L-H unit
 └─ proj: 1x1 conv(dim->dim)

Chanel_Cross_Attention(nn.Module)             # == CA, Eq.(2)-(3)
 └─ forward(x, y): x -> Q, y -> K,V ; multi-head channel attention (attn matrix is CxC, not spatial)
```

## 3. Exact equations implemented (paper vs. code)

### 3.1 MGB / mask generation — **the equation under test**

Paper, Eq. (1) (p.6):

```
[alpha, beta] = sigmoid( W2^(1x1) ( GELU( W1^(1x1) ( GAP_s(P) ) ) ) )

M_l[ H/2 - alpha*H/k : H/2 + alpha*H/k,  W/2 - beta*W/k : W/2 + beta*W/k ] = 1
M_h = 1 - M_l                                     (k = 128)
```

i.e. the box half-height is **`alpha * H / k`** — multiply `alpha` by `H` first, then
divide by `k`.

Code, `net/model.py`, `FreModule.fft()`:

```python
threshold = F.adaptive_avg_pool2d(x, 1)
threshold = self.rate_conv(threshold).sigmoid()          # == alpha, beta  (matches Eq. 1's rate_conv)

for i in range(mask.shape[0]):
    h_ = (h//n * threshold[i,0,:,:]).int()                # ***(h // n) * alpha***, THEN int()
    w_ = (w//n * threshold[i,1,:,:]).int()
    mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
```

**Finding**: the code computes `int( (H // k) * alpha )`, i.e. it floor-divides `H` by
`k` *before* multiplying by `alpha`, whereas the paper's Eq. (1) specifies
`alpha * H / k` (multiply first, divide by `k`, as a real number, discretized only once
when building the integer slice). `(H // k) * alpha` and `(alpha * H) / k` are **not**
algebraically equivalent under integer flooring — e.g. `H=250, alpha=0.9, k=128`:
paper form `int(0.9*250/128) = int(1.758) = 1` (non-zero), code form
`int((250//128)*0.9) = int(1*0.9) = int(0.9) = 0` (zero). At the resolutions used by
every AdaIR benchmark (`H < 128`, so `H // 128 == 0` outright), both forms are usually
zero — the two formulations only diverge once `H` grows past ~128px, and even then not
identically. This is an **implementation-level deviation from the published equation**,
not merely a "large enough resolution" issue — a correctly-ordered evaluation of Eq. (1)
would produce non-zero masks at somewhat smaller resolutions than the code's
floor-then-multiply order does, at borderline `alpha`/`beta` values.

We do not have evidence either way on *why* the code is written this order (author
implementation choice, a translation error from the paper's formula, or a deliberate
simplification) — that is not directly observable from the source and is not claimed here.

### 3.2 FMiM cross-attention (Eq. 2-4, p.7) — CA / `Chanel_Cross_Attention`

```
X_* = softmax(Q K^T / alpha) V,   * in {l, h}
Q = DW1(W3^1x1(F_*)),  K = DW2(W4^1x1(X)),  V = DW3(W5^1x1(X))
F_* = IFFT(M_* ⊙ F)
```

Code matches: `channel_cross_l(high_feature, y)` and `channel_cross_h(low_feature, y)`
implement exactly this — `x` (first arg) generates Q, `y` (second arg, the decoder
feature) generates K/V. `Chanel_Cross_Attention.temperature` is the learnable `alpha`
scaling factor in Eq. (2). Attention matrix shape is `(B, heads, C, C)` — **channel**
attention, not spatial — so it is not vulnerable to the same H/W discretization issue
as the mask.

### 3.3 FMoM (Eq. 5-6, p.7) — `FreRefine`

```
X_l_hat = X_l ⊙ A_{H-L},    A_{H-L} = sigmoid(W6^7x7([GAP_c(X_h), GMP_c(X_h)]))
```

Code: `FreRefine.forward(low, high)`:
`spatial_weight = SpatialGate(high)` (H-L, computed from **high**-freq features via
max+mean channel pooling + 7x7 conv + sigmoid, matches Eq. 6 exactly — `SpatialGate`
uses `torch.max`/`torch.mean` over the channel dim, equivalent to GMP_c/GAP_c);
`channel_weight = ChannelGate(low)` (L-H, from **low**-freq features via
avgpool+maxpool+MLP+sigmoid); then `low_w = low*spatial_weight`,
`high_w = high*channel_weight`, `agg = proj(low_w + high_w)`. Matches the paper's stated
cross-complementation (high informs the low branch's spatial map; low informs the high
branch's channel map).

## 4. Tensor shapes at every AFLB

For a representative Rain100L image (480x320 after `crop_img(base=16)`):

| AFLB | Applied to | dim | feature H×W | conv_feat/mask shape |
|---|---|---:|---|---|
| AFLB1 (`fre1`) | `latent` (deepest) | 384 (`dim*2^3`) | 40×60 | (1, 384, 40, 60) |
| AFLB2 (`fre2`) | `decoder_level3` | 192 (`dim*2^2`) | 80×120 | (1, 192, 80, 120) |
| AFLB3 (`fre3`) | `decoder_level2` | 96 (`dim*2^1`) | 160×240 | (1, 96, 160, 240) |

Downsample factors from input: AFLB1 = H/8, AFLB2 = H/4, AFLB3 = H/2 (three `Downsample`
stages, each `PixelUnshuffle(2)`, precede the latent).

### 4.1 Dead code (present in checkpoint, never called)

`FreModule.__init__` defines `self.conv` (3x3 conv, in_dim=3->dim) and `self.score_gen`
(7x7 conv, 2->2). Neither is referenced anywhere in `FreModule.forward()` or `.fft()`.
The released checkpoint (`adair3d.ckpt`) loads with **zero missing / zero unexpected
keys** into the full `AdaIR(decoder=True)` module — i.e. these submodules do hold
trained weights, they are simply never exercised by the forward pass. This is stated
here as an observed fact (loader assertion + `grep` over `forward()`), not a claim about
author intent.

## 5. Checkpoint loading logic

`test.py` (released): wraps `AdaIR` in a `pytorch_lightning.LightningModule`
(`AdaIRModel`) and calls `AdaIRModel.load_from_checkpoint(ckpt_path)`, i.e. it expects a
Lightning-format checkpoint with a `state_dict` keyed under `net.*` and does not
independently validate key coverage.

Our loader (`scripts/instrument.py: load_adair`) is a plain-PyTorch equivalent, used
because we do not want a Lightning dependency for inference-only instrumentation: loads
the raw checkpoint dict, extracts `state_dict`/`params`/`model`/`net` (whichever key is
present), strips a `net.`/`module.`/`model.` prefix, and calls
`model.load_state_dict(sd, strict=False)` followed by an explicit assertion that
`missing_keys == [] and unexpected_keys == []` — i.e. we get Lightning's implicit
strict-load guarantee explicitly, without the Lightning dependency. Verified: loads
`adair3d.ckpt` into `AdaIR(decoder=True)` with 28,784,824 parameters (matches the
paper's stated ~28.8M) and 0 missing/0 unexpected keys.

## 6. Evaluation logic

`utils/val_utils.py: compute_psnr_ssim` (released) — operates on a batch of restored/clean
tensors already in `[0,1]`, converts to `uint8` (`*255`, round), computes
`skimage.metrics.peak_signal_noise_ratio` and `structural_similarity` per image in the
batch (multichannel), returns the batch mean and count. Our `scripts/stats_utils.py:
psnr_ssim` reimplements this exactly (uint8 round-trip, `data_range=255`,
`channel_axis=2`) for a single (1,3,H,W) pair, to avoid the DataLoader/Lightning
plumbing `test.py` otherwise requires.

`utils/dataset_utils.py: crop_img(image, base=16)` crops (does **not** pad) each image
so H and W are multiples of 16 — trims from both edges symmetrically
(`crop_h//2 : h-crop_h+crop_h//2`). `DenoiseTestDataset._add_gaussian_noise`: additive
`np.random.randn(*shape) * sigma` on the `[0,255]` uint8 image, clipped and re-cast to
uint8, *before* `ToTensor()`. Our `scripts/run_inference.py` reimplements both exactly
(verified against source, see prior conversation turns).

## 7. Datasets / images used

| Degradation | Source | Images used | Note |
|---|---|---:|---|
| Derain | Rain100L test split (`.../rain100L_test/Rain100L/{norain-*,rainy/rain-*}.png`) | 100/100 | full test split, no sampling needed |
| Dehaze | SOTS-outdoor (`.../RESIDE/SOTS/outdoor/{input,target}`) | 100 of 492 unique scenes | de-duplicated by scene id (input has some scenes with >1 haze render), first 100 sorted |
| Denoise | BSD68 (`.../test/denoise/bsd68`, 68 unique clean images) | 100 instances | AdaIR's own protocol synthesises noise at test time (no native noisy files); 68 images @ sigma=25 + 16 extra @ sigma=15 + 16 extra @ sigma=50 = 100 instances, sigma recorded per row |

## 8. Checkpoint used

`weights/adair3d.ckpt` — the **3-degradation** all-in-one teacher (denoise+derain+dehaze),
as distinct from `adair5d.ckpt` (5-degradation, adds deblur+enhance) and the
single-degradation specialist checkpoints also present in
`/home/minura/FYP/Workspace/Himeth/weights/`. 28,784,824 parameters, loads with 0
missing/0 unexpected keys.

## 9. Environment

- Remote host: devon (192.248.10.68), user `minura`
- GPU: NVIDIA RTX 4090, 24564 MiB VRAM
- **Known hardware issue**: logical CPUs 8-11 on this host intermittently and silently
  corrupt in-process data (observed as a nonsensical `PIL.PngImagePlugin` TypeError
  mid-run, and separately as an `openpyxl` XML-parse crash on a freshly-written file).
  Every Python invocation on this host in this project is pinned with
  `taskset -c 0-7,12-31` to avoid those cores. Excel files are never written on this
  host — CSVs are exported here and the `.xlsx` is rendered on a separate (trusted)
  local machine.
- conda env: `adair-distill` (Python 3.11.15, torch 2.5.1+cu, torchvision 0.20.1,
  onnx 1.17.0, onnxruntime 1.20.1, scikit-image 0.24.0, scikit-learn 1.9.0,
  matplotlib 3.11.1, pandas 2.3.3)
- AdaIR source: `git rev-parse HEAD` = `ccb8b98e49614e07badd0641e5163fa7635c2f02`
  (`github.com/c-yn/AdaIR`, fresh clone, not the copy vendored in the older
  `fyp-adair-distill` repo)
