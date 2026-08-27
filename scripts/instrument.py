"""Non-invasive instrumentation of AdaIR's FreModule (the paper's AFLB).

We do NOT edit the vendored AdaIR/net/model.py. Instead we monkey-patch a
*replica* of FreModule.forward/.fft onto each of the three fre1/fre2/fre3
instances at runtime (types.MethodType), calling the exact same submodules
(self.conv1, self.rate_conv, self.channel_cross_l/h/agg,
self.frequency_refine.SpatialGate/ChannelGate/proj, self.para1/2) in the exact
same order as the original code. Every intermediate the analysis plan wants is
captured into a Recorder along the way.

Code -> plan-vocabulary mapping (see README sheet in the workbook for the
full writeup):

  FreModule           == AFLB (Adaptive Frequency Learning Block), x3 in decoder
  FreModule.fft()      -> "MGB" (mask/boundary generation) + FMiM raw split
  channel_cross_l/h     -> FMiM cross-attention (mines high/low against y)
  FreRefine (frequency_refine) -> FMoM
    .SpatialGate (on high) -> "H-L unit"  (spatial attention, applied to low)
    .ChannelGate (on low)  -> "L-H unit"  (channel attention, applied to high)
  channel_cross_agg     -> final cross-attention merge with y
  out*para1 + y*para2   -> AFLB residual output

Two submodules exist in every FreModule (self.conv, self.score_gen) but are
NEVER called in the original forward() -- they hold trained weights (the
checkpoint loads them with zero missing/unexpected keys) yet are dead code.
We reproduce that faithfully: we do not invoke them either.

AFLB depth ordering: fre1 runs first (on `latent`, the deepest/smallest
feature map, dim*8 channels) -> fre2 (decoder_level3, dim*4) -> fre3
(decoder_level2, dim*2, shallowest/largest). So "AFLB index 1" = deepest.
"""
from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn.functional as F


class Recorder:
    """Collects tensors for the AFLB currently being instrumented.

    `store[aflb_name][key] = tensor` (detached, still on-device). Call
    `start()` before a forward pass and `snapshot()` to pull a plain dict out
    (moved to CPU) afterwards.
    """

    def __init__(self):
        self.enabled = False
        self._store: dict[str, dict[str, torch.Tensor]] = {}

    def start(self):
        self._store = {}
        self.enabled = True

    def put(self, aflb_name: str, key: str, tensor: torch.Tensor):
        if not self.enabled:
            return
        self._store.setdefault(aflb_name, {})[key] = tensor.detach()

    def snapshot_cpu(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            aflb: {k: v.cpu() for k, v in d.items()}
            for aflb, d in self._store.items()
        }


def _instrumented_fft(self, x, recorder: Recorder, aflb_name: str, n=128):
    """Replica of FreModule.fft — identical math, with capture points."""
    conv_feat = self.conv1(x)
    recorder.put(aflb_name, "conv_feat", conv_feat)

    mask = torch.zeros(conv_feat.shape).to(conv_feat.device)
    h, w = conv_feat.shape[-2:]
    threshold = F.adaptive_avg_pool2d(conv_feat, 1)
    threshold = self.rate_conv(threshold).sigmoid()
    recorder.put(aflb_name, "threshold_alpha_beta", threshold)  # (B,2,1,1)

    for i in range(mask.shape[0]):
        h_ = (h // n * threshold[i, 0, :, :]).int()
        w_ = (w // n * threshold[i, 1, :, :]).int()
        mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
    recorder.put(aflb_name, "mask", mask)

    fft = torch.fft.fft2(conv_feat, norm="forward", dim=(-2, -1))
    fft = self.shift(fft)
    recorder.put(aflb_name, "fft_shifted", fft)

    fft_high = fft * (1 - mask)
    high = self.unshift(fft_high)
    high = torch.fft.ifft2(high, norm="forward", dim=(-2, -1))
    high = torch.abs(high)
    recorder.put(aflb_name, "raw_high", high)

    fft_low = fft * mask
    low = self.unshift(fft_low)
    low = torch.fft.ifft2(low, norm="forward", dim=(-2, -1))
    low = torch.abs(low)
    recorder.put(aflb_name, "raw_low", low)

    return high, low


def _instrumented_forward(self, x, y, recorder: Recorder, aflb_name: str):
    """Replica of FreModule.forward — identical math, with capture points."""
    _, _, H, W = y.size()
    x = F.interpolate(x, (H, W), mode="bilinear")
    recorder.put(aflb_name, "x_resized", x)

    high_feature, low_feature = self._instrumented_fft(x, recorder, aflb_name)

    high_feature = self.channel_cross_l(high_feature, y)
    low_feature = self.channel_cross_h(low_feature, y)
    recorder.put(aflb_name, "mined_high", high_feature)
    recorder.put(aflb_name, "mined_low", low_feature)

    # -- FreRefine (FMoM), inlined so we can capture H-L / L-H separately --
    spatial_weight = self.frequency_refine.SpatialGate(high_feature)   # H-L unit
    channel_weight = self.frequency_refine.ChannelGate(low_feature)    # L-H unit
    recorder.put(aflb_name, "hl_spatial_weight", spatial_weight)
    recorder.put(aflb_name, "lh_channel_weight", channel_weight)

    high_w = high_feature * channel_weight
    low_w = low_feature * spatial_weight
    agg = low_w + high_w
    agg = self.frequency_refine.proj(agg)
    recorder.put(aflb_name, "fmom_agg", agg)

    out = self.channel_cross_agg(y, agg)
    recorder.put(aflb_name, "cross_agg_out", out)

    aflb_out = out * self.para1 + y * self.para2
    recorder.put(aflb_name, "aflb_out", aflb_out)
    recorder.put(aflb_name, "y_in", y)

    return aflb_out


def attach_instrumentation(model, recorder: Recorder):
    """Monkey-patch fre1/fre2/fre3 with the capturing replicas above.

    fre1 -> AFLB1 (deepest, on `latent`)
    fre2 -> AFLB2 (decoder_level3)
    fre3 -> AFLB3 (decoder_level2, shallowest)
    """
    net = model.net if hasattr(model, "net") else model
    name_map = {"fre1": "AFLB1", "fre2": "AFLB2", "fre3": "AFLB3"}
    for attr, aflb_name in name_map.items():
        fre = getattr(net, attr)
        fre._instrumented_fft = types.MethodType(
            lambda self, x, recorder=recorder, aflb_name=aflb_name, n=128: _instrumented_fft(
                self, x, recorder, aflb_name, n),
            fre,
        )
        fre.forward = types.MethodType(
            lambda self, x, y, recorder=recorder, aflb_name=aflb_name: _instrumented_forward(
                self, x, y, recorder, aflb_name),
            fre,
        )
    return net


TRANSFORMER_STAGES = [
    "encoder_level1", "encoder_level2", "encoder_level3", "latent",
    "decoder_level3", "decoder_level2", "decoder_level1", "refinement",
]


def attach_stage_hooks(net, recorder: Recorder):
    """Forward hooks on the plain nn.Sequential encoder/decoder stages.

    These need no replication -- a forward hook sees the real output tensor
    directly.
    """
    handles = []
    for stage_name in TRANSFORMER_STAGES:
        stage = getattr(net, stage_name)

        def _hook(module, inputs, output, stage_name=stage_name):
            recorder.put("_stages", stage_name, output)

        handles.append(stage.register_forward_hook(_hook))
    return handles


def load_adair(adair_repo: Path, ckpt_path: Path, device: str):
    import sys
    sys.path.insert(0, str(adair_repo))
    from net.model import AdaIR

    model = AdaIR(decoder=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = None
    for key in ("state_dict", "params", "model", "net"):
        if isinstance(ckpt.get(key), dict):
            sd = ckpt[key]
            break
    if sd is None:
        sd = ckpt
    sd = {(k[len("net."):] if k.startswith("net.") else k): v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=False)
    assert not result.missing_keys, f"missing keys: {result.missing_keys}"
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model
