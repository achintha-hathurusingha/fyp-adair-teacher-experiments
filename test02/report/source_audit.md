# TEST02 Source-Code Audit — Degradation Conditioning Mechanisms

Scope: does the released AdaIR implementation contain ANY explicit
degradation label, embedding, classifier, prompt, routing mechanism, or
task ID? Answered by direct inspection of `teacher-experiments/AdaIR`
(`git rev-parse HEAD` = `ccb8b98e49614e07badd0641e5163fa7635c2f02`), not
inferred from the paper. Extends `test01/docs/adair_source_audit.md`
(read-only reference, not modified) with a specific focus on this question.

## 1. Model entry point

`net/model.py`, class `AdaIR(nn.Module)`, method:

```python
def forward(self, inp_img, noise_emb=None):
```

**Finding**: `noise_emb` is the *only* parameter besides the image itself.
`grep -n noise_emb net/model.py` returns exactly **one** match — this
signature line. It is never read, never passed to any submodule, never
referenced anywhere else in the 475-line file. **There is no code path by
which any external label, embedding, or task ID reaches the model.** This
is a vestigial parameter slot (plausibly left over from an earlier/other
variant of the architecture that did take a conditioning embedding),
consistent with the two other vestigial submodules found in `test01`'s
audit (`FreModule.conv`, `FreModule.score_gen` — defined, checkpoint-loaded,
never called).

**Conclusion**: AdaIR performs **blind** all-in-one restoration — it infers
what to do purely from the pixel content of `inp_img`. Any degradation-type
information the network uses must be *implicitly derived from the image
itself* somewhere in the forward pass, not supplied externally. This
directly matches the paper's claim of operating "without requiring the
prior information of the input degradation type" — and TEST02 exists to
find *where*, if anywhere, that implicit information becomes explicit
enough to be linearly read out.

## 2. Full module inventory relevant to this question

| # | Source file / class / function | Purpose | Tensor shape (480x320 input) | Could carry degradation info? |
|---|---|---|---|---|
| 1 | `AdaIR.forward` | entry point | in: (1,3,H,W) | N/A (routing only) |
| 2 | `OverlapPatchEmbed.forward` (`self.patch_embed`) | shallow feature extraction, 3x3 conv 3->48ch | (1,48,H,W) | Yes — first learned representation |
| 3 | `encoder_level1` (`nn.Sequential[TransformerBlock]` x4) | level-1 encoder | (1,48,H,W) | Yes |
| 4 | `Downsample` (`down1_2`) | conv+PixelUnshuffle | (1,96,H/2,W/2) | pass-through, not separately probed |
| 5 | `encoder_level2` (x6 blocks) | level-2 encoder | (1,96,H/2,W/2) | Yes |
| 6 | `down2_3` | downsample | (1,192,H/4,W/4) | pass-through |
| 7 | `encoder_level3` (x6 blocks) | level-3 encoder | (1,192,H/4,W/4) | Yes |
| 8 | `down3_4` | downsample | (1,384,H/8,W/8) | pass-through |
| 9 | `latent` (x8 blocks) | bottleneck | (1,384,H/8,W/8) | Yes — deepest, most compressed |
| 10 | `fre1` (`FreModule`, AFLB1) | frequency learning block on `latent` | (1,384,H/8,W/8) | Yes — see below |
| 11 | `up4_3`, `reduce_chan_level3`, `decoder_level3` (x6) | decoder level 3 | (1,192,H/4,W/4) | Yes |
| 12 | `fre2` (AFLB2) | on `decoder_level3` output | (1,192,H/4,W/4) | Yes |
| 13 | `up3_2`, `reduce_chan_level2`, `decoder_level2` (x6) | decoder level 2 | (1,96,H/2,W/2) | Yes |
| 14 | `fre3` (AFLB3) | on `decoder_level2` output | (1,96,H/2,W/2) | Yes |
| 15 | `up2_1`, `decoder_level1` (x4) | decoder level 1 | (1,96,H,W) | Yes |
| 16 | `refinement` (x4 blocks) | final refinement | (1,96,H,W) | Yes |
| 17 | `output` (3x3 conv, 96->3, + residual `+inp_img`) | reconstruction | (1,3,H,W) | Output only |

### Inside every `FreModule` (AFLB) — `net/model.py`

| # | Function/attribute | Purpose | Shape (AFLB1 example, dim=384) | Notes |
|---|---|---|---|---|
| 18 | `FreModule.rate_conv` | MGB: produces `[alpha,beta]` | (1,2,1,1) | Section 3.1 of test01 audit: this is the *only* place a scalar summary of the image's spectral content is compressed to 2 numbers per image |
| 19 | `FreModule.fft()` -> `raw_high`, `raw_low` | FMiM raw split | (1,384,H/8,W/8) each | `raw_low` is exactly zero at benchmark resolution (test01 finding) — see Phase 12/13 here for whether it's *still* informative in aggregate across 300 images despite being exactly zero per-image (trivially: a constant zero vector carries zero information by construction, but this is verified empirically, not assumed) |
| 20 | `channel_cross_l`/`channel_cross_h` (`Chanel_Cross_Attention`) -> `mined_high`, `mined_low` | FMiM cross-attention mining | (1,384,H/8,W/8) each | Q from raw_high/raw_low, K/V from `y` (decoder feature) — this is where the (possibly degenerate) frequency split gets mixed back with real spatial content |
| 21 | `FreRefine.SpatialGate` -> `hl_spatial_weight` | FMoM H-L unit | (1,1,H/8,W/8) | spatial attention map |
| 22 | `FreRefine.ChannelGate` -> `lh_channel_weight` | FMoM L-H unit | (1,384,1,1) | channel attention vector — notably **already a compact per-channel descriptor**, a natural distillation-target candidate purely on dimensionality grounds |
| 23 | `channel_cross_agg` -> `cross_agg_out` | final CA merge | (1,384,H/8,W/8) | |
| 24 | `aflb_out = out*para1 + y*para2` | AFLB residual output | (1,384,H/8,W/8) | what actually propagates to the next decoder stage |

## 3. Explicit degradation mechanism search — result

Searched for: class-embedding tables (`nn.Embedding`), prompt banks (as in
PromptIR, a different paper's architecture — AdaIR does not use this
pattern), one-hot task vectors, `if degradation_type ==` branching,
auxiliary classifier heads, task-routing/mixture-of-experts gates.

**None found.** The only per-image scalar bottleneck anywhere in the
architecture is `FreModule.rate_conv`'s `[alpha, beta]` output (item 18) —
and that is a *learned function of the image*, not a label; TEST02 Phase 13
tests empirically whether it happens to carry usable degradation
information as a side effect, not by assumption.

## 4. Checkpoint loading (unchanged from test01)

Same strict loader as `test01` (`scripts/instrument.py: load_adair`),
`adair3d.ckpt`, 28,784,824 parameters, 0 missing / 0 unexpected keys.
**Not modified for TEST02** — verified once at extraction time and
asserted before any feature extraction runs.

## 5. Test dataset handling (unchanged from test01)

Same `crop_img(base=16)` cropping, same Gaussian-noise synthesis for the
Noise degradation (now using the per-image deterministic seeding fix
established in `test01`), same 300-image manifest content (Rain100L /
SOTS-outdoor / BSD68, identical selection logic) — re-exported as
`test02/results/dataset_manifest.csv` per the TEST02 spec's required
schema, not regenerated from scratch.
