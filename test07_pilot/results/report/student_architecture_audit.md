# TEST07-Pilot: Student Architecture Audit

Per the task's explicit instruction: NAFNet code already exists (in
`fyp-adair-distill`, read-only reference, NOT modified). Read, not
reimplemented from scratch.

## Source

`fyp-adair-distill/src/models/nafnet.py` (clean-room reimplementation, no
`basicsr` dependency), `src/models/norms.py`, `configs/model/nafnet_locked.yaml`.
Git history (`08f01b7`, `2a1c791`, `0150003`, `e191b47`) shows this is an
actively engineered, validated architecture — NOT a stub.

## 1. Bottleneck tensor

`self.middle_blks` output — the last tensor before the decoder's first
upsample. For the **LOCKED "M" arm** (`width=16, enc_blk_nums=[2,2,4,8],
middle_blk_num=12, dec_blk_nums=[2,2,2,2]`): 4 encoder downsample stages,
each halving H,W and doubling channels, so bottleneck shape =
`(B, 16*2^4, H/16, W/16) = (B, 256, H/16, W/16)`. For a 128×128 training
patch: `(B, 256, 8, 8)`.

## 2. Normalization layers

`build_norm(norm_type, ...)` from `norms.py`, 7 supported variants
(`NORM_TYPES`). The LOCKED config uses **N-F+clamp**: `layernorm2d`
(standard channel-wise LayerNorm, mean+var) everywhere EXCEPT the two
full-resolution stages (encoder level 0 and its decoder counterpart), which
use `affine_clamp` (`AffineClampNorm2d`, no statistics, hard magnitude
clamp at ±8.0 per `nafnet_locked.yaml`, chosen from measured divergence
data — see `findings.md` F1/F6/F9). This choice trades 0.005dB quality for
a documented 1.59× NPU latency improvement (F1: normalization, not
convolution, dominates INT8 Hexagon latency).

## 3. Parameter count

M arm: **7.37M params, 4.13 GMACs** (per `family_reselection.md`,
`w16_sidd` config). S/L alternatives also defined (`family:` block in
`nafnet_locked.yaml`) — S=smaller/faster, L=larger. TEST07-Pilot uses the
**M arm** (the team's current locked default) for all four models, per the
task's requirement that all models share identical architecture except for
the proposed mechanism.

## 4. Operation support

All ops (`Conv2d`, `AdaptiveAvgPool2d`, `PixelShuffle`, the custom norm
variants, `SimpleGate`) are standard PyTorch 2.5.1 — the exact version
already validated in the `adair-distill` conda environment used throughout
TEST01–06-R. No additional dependencies required. `ChannelGate`
(`gate.py`) exists but is disabled by default (`gate.enabled: false` in
the locked config) — TEST07-Pilot leaves it disabled, matching the locked
default, since gating is an orthogonal Phase-02 concern per the source
docstring.

## 5. Forward signature

`NAFNet.forward(inp) -> output`, `(B,3,H,W) -> (B,3,H,W)`, global residual
applied (`x = x + inp`), auto-pads to a multiple of `2^n_encoder_stages=16`.
Straightforward to hook: TEST07-Pilot subclasses/wraps this exact class,
inserting a bottleneck tap after `self.middle_blks` and, for Models C/D,
a modulation step applied to that same tensor before the decoder loop
begins — the LOCKED class is used unmodified via composition (a thin
wrapper), not by editing `fyp-adair-distill`'s file.

## 6. Conclusion

NAFNet code exists, is mature, and is directly reusable. TEST07-Pilot
imports `fyp-adair-distill/src/models/nafnet.py` and `norms.py` **read-only**
(via `sys.path` insertion, matching the established read-only-import
convention used throughout TEST05.5/TEST06 for `test01`/`test04` imports),
and builds Models A–D as thin wrapper classes in `test07_pilot/src/` that
compose the unmodified `NAFNet`/`NAFBlock` classes rather than editing them.
