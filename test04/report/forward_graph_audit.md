# TEST04 Phase 0 — AdaIR Forward Graph Audit

Source: `net/model.py`, `AdaIR.forward()` (lines 426-475), verified by direct
`sed` extraction from the vendored repo (`git rev-parse HEAD` =
`ccb8b98e49614e07badd0641e5163fa7635c2f02`), not memory or assumption.
Read-only — this file was not modified.

## 1-2. Tensor flow and module table

| # | Tensor | Producer | Consumer(s) | Shape (480x320 input) | Skip connection? |
|---|---|---|---|---|---|
| 1 | `inp_enc_level1` | `patch_embed(inp_img)` | `encoder_level1` | (1,48,H,W) | no |
| 2 | `out_enc_level1` | `encoder_level1` | `down1_2`; **`torch.cat` at decoder_level1 input** | (1,48,H,W) | **YES -- skip to decoder L1** |
| 3 | `inp_enc_level2` | `down1_2(out_enc_level1)` | `encoder_level2` | (1,96,H/2,W/2) | no |
| 4 | `out_enc_level2` | `encoder_level2` | `down2_3`; **`torch.cat` at decoder_level2 input** | (1,96,H/2,W/2) | **YES -- skip to decoder L2** |
| 5 | `inp_enc_level3` | `down2_3(out_enc_level2)` | `encoder_level3` | (1,192,H/4,W/4) | no |
| 6 | `out_enc_level3` | `encoder_level3` | `down3_4`; **`torch.cat` at decoder_level3 input** | (1,192,H/4,W/4) | **YES -- skip to decoder L3** |
| 7 | `inp_enc_level4` | `down3_4(out_enc_level3)` | `latent` | (1,384,H/8,W/8) | no |
| 8 | `latent` (pre-AFLB1) | `self.latent(inp_enc_level4)` | `fre1` (AFLB1) | (1,384,H/8,W/8) | no |
| 9 | `latent` (post-AFLB1, **variable reassigned**) | `fre1(inp_img, latent)` | `up4_3` | (1,384,H/8,W/8) | no |
| 10 | `inp_dec_level3` | `up4_3(latent)` then `cat([·, out_enc_level3])` then `reduce_chan_level3` | `decoder_level3` | (1,192,H/4,W/4) | consumes skip #6 |
| 11 | `out_dec_level3` (pre-AFLB2) | `decoder_level3` | `fre2` (AFLB2) | (1,192,H/4,W/4) | no |
| 12 | `out_dec_level3` (post-AFLB2, reassigned) | `fre2(inp_img, out_dec_level3)` | `up3_2` | (1,192,H/4,W/4) | no |
| 13 | `inp_dec_level2` | `up3_2(·)` then `cat([·, out_enc_level2])` then `reduce_chan_level2` | `decoder_level2` | (1,96,H/2,W/2) | consumes skip #4 |
| 14 | `out_dec_level2` (pre-AFLB3) | `decoder_level2` | `fre3` (AFLB3) | (1,96,H/2,W/2) | no |
| 15 | `out_dec_level2` (post-AFLB3, reassigned) | `fre3(inp_img, out_dec_level2)` | `up2_1` | (1,96,H/2,W/2) | no |
| 16 | `inp_dec_level1` | `up2_1(·)` then `cat([·, out_enc_level1])` | `decoder_level1` | (1,96,H,W) | consumes skip #2 |
| 17 | `out_dec_level1` | `decoder_level1` -> `refinement` -> `output` conv | **`+ inp_img`** (final residual) | (1,3,H,W) | -- |

**Critical finding #1 -- `inp_img` is re-injected FOUR times, independently
of latent/AFLB swapping**: once into `fre1`, once into `fre2`, once into
`fre3` (each `FreModule.forward(x, y)` call takes `inp_img` as its first
argument `x`, used to compute `conv_feat`/the FMiM query branch), and once
as the final global residual (`output = ... + inp_img`). **Swapping
`latent` or any AFLB output does NOT remove the recipient's own degraded
image from the computation** -- the recipient's `inp_img` keeps flowing
into every AFLB's FMiM branch and into the final residual regardless of
what internal representation is substituted. This means even a "complete"
internal-representation swap cannot fully impersonate the donor; the
recipient's raw input remains partially present throughout. This is
stated as an architectural fact, not a flaw, and directly motivates
Phase 8's progressive skip-connection intervention design.

**Critical finding #2 -- three independent skip connections bypass
latent/AFLB entirely**: `out_enc_level1/2/3` are consumed via
`torch.cat` directly at each decoder level's input, in parallel with
whatever comes down from latent/AFLB. **A latent-only or AFLB-output-only
swap leaves all three encoder skip connections carrying the RECIPIENT's
own (unswapped) encoder features into every decoder stage.** This is the
single most important fact for interpreting intervention results: if a
latent swap produces only a partial output change, that is expected
given the architecture, not necessarily evidence the representation is
weakly causal -- the skip paths provide an alternate, un-intervened
channel for recipient-specific information to keep influencing the
output.

## 3. Concatenation/fusion tensors

Three `torch.cat` + 1x1-conv (`reduce_chan_level2/3`) fusions, exactly as
listed in rows 10, 13, 16 above. `decoder_level1`'s input is `cat([·,
out_enc_level1])` directly (48+48=96 channels matches `decoder_level1`'s
expected `dim*2^1`), no separate reduce-channel conv at that level
(channel count already matches after concatenation).

## 4-5. Tensors entering each AFLB / produced by FMiM/FMoM

Each `FreModule.forward(x, y)` (see `test01/docs/adair_source_audit.md`
section 3 for the full FMiM/FMoM equation trace, unchanged here) receives:
`x = inp_img` (recipient's raw degraded image, NOT swappable without
swapping the top-level input, which is out of scope -- that would just be
running the donor image directly) and `y` = the tensor being processed at
that depth (`latent` for AFLB1, `out_dec_level3` for AFLB2, `out_dec_level2`
for AFLB3). `y` is exactly the tensor we intervene on for the "AFLB
output" and "latent" experiments (as the INPUT `y` to the FreModule, whose
OUTPUT then replaces it for the downstream computation).

## 6-7. Decoder dependency on encoder features / latent replaceability

Yes -- confirmed by rows 10, 13, 16 (critical finding #2). Latent
(row 8/9) **can be safely replaced from an engineering standpoint**: it is
a plain tensor output of `self.latent(...)`, consumed only by `fre1`, no
hidden state, no in-place mutation, no batch-norm running stats (the
network uses `LayerNorm`, which is instance-wise, not batch-wise -- safe
under single-image batch=1 substitution). Shape/channel count are fixed
by the architecture (384, H/8, W/8) regardless of input content, so a
donor latent from a DIFFERENT degraded version of the SAME scene has
IDENTICAL shape to the recipient's (same crop dimensions, same scene) --
confirmed safe to swap.

## 8. AFLB output replaceability

Same conclusion: `fre1/2/3` outputs are plain tensors (no hidden state),
consumed by the next `up*_*` module. Safe to swap between same-shape
donor/recipient tensors.

## 9. Shape/channel matching

Since TEST03's Rain/Haze/Noise variants of a scene are all cropped from
the *same* clean image with the *same* `crop_img(base=16)` call, every
donor/recipient pair for a given scene has bit-identical spatial
dimensions and (by architecture) identical channel counts at every
intervention point. Verified programmatically before every swap (hard
assertion in `src/intervention.py`), not just assumed.

## 10. Normalization/hidden state that could make swapping invalid

`LayerNorm` (`BiasFree_LayerNorm`/`WithBias_LayerNorm`) is computed
per-sample, per-position, over the channel dimension only -- no running
statistics, no cross-sample state. `nn.Conv2d` layers are stateless at
inference. **No BatchNorm anywhere in AdaIR** (confirmed by `grep -n
BatchNorm net/model.py` -> no matches). **Conclusion: there is no hidden
state anywhere in the model that would make a same-shape tensor
substitution invalid.** The model is a pure function of its inputs at
each stage; substituting an intermediate tensor with another
identically-shaped tensor and continuing the same sequence of module
calls is a well-defined, valid operation.

## Verdict

**Latent and AFLB-output swapping are both technically valid**, subject to
the two critical findings above (skip connections and `inp_img`
re-injection provide un-intervened channels that a pure latent/AFLB swap
cannot close). This motivates the Phase 8 progressive design: Condition A
(latent only) isolates the bottleneck's causal contribution in the
presence of these un-intervened channels; Condition B/C (latent + skips)
test whether closing those channels strengthens the effect, without
assuming more swapping is automatically better (per the task's explicit
caution).

**Terminology used throughout TEST04, per task instruction**: there is
**one shared decoder** (not per-degradation decoders). We refer to
"recipient computation" (whose `inp_img`, skip connections, and downstream
weights are used) and "donor representation" (the internal tensor
substituted in from a different degraded version of the same scene).
