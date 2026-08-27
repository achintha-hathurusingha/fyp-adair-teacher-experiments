"""TEST18: parameterized reimplementation of AdaIR's FreModule (the AFLB)
supporting the exact ablation axes in the paper's own Table 7 (Section
4.4), plus a full AdaIR variant that swaps fre1/fre2/fre3 for the
ablatable version. Read-only reuse of everything else in AdaIR's
net/model.py (imported directly, never edited in place).

Ablation axes (mask_mode, use_lh, use_hl) -> Table 7 rows:
    A_baseline:      mask_mode=None                          (no AFLB at all)
    B_fixed_mask:    mask_mode="fixed",   use_lh=False, use_hl=False
    C_learned_mask:  mask_mode="learned", use_lh=False, use_hl=False
    D_plus_lh:       mask_mode="learned", use_lh=True,  use_hl=False
    E_full:          mask_mode="learned", use_lh=True,  use_hl=True  (== released AdaIR)

Design decision (documented, not specified by the paper): when both
use_lh and use_hl are False, FreRefine (FMoM) is skipped entirely and
low_feature + high_feature are summed directly before channel_cross_agg,
since Table 7 does not specify the exact fallback aggregation for that
intermediate state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TEACHER_EXP = Path(__file__).resolve().parent.parent.parent
_CANDIDATES = [TEACHER_EXP.parent / "AdaIR", TEACHER_EXP / "AdaIR"]  # local vs devon layout
ADAIR_ROOT = next((p for p in _CANDIDATES if (p / "net" / "model.py").exists()), _CANDIDATES[0])
sys.path.insert(0, str(ADAIR_ROOT))

from net.model import (  # noqa: E402 (read-only reuse)
    AdaIR, OverlapPatchEmbed, TransformerBlock, Downsample, Upsample,
    Chanel_Cross_Attention, SpatialGate, ChannelGate, FreRefine,
)

VARIANTS = {
    "A_baseline":     dict(mask_mode=None,     use_lh=False, use_hl=False),
    "B_fixed_mask":   dict(mask_mode="fixed",   use_lh=False, use_hl=False),
    "C_learned_mask": dict(mask_mode="learned", use_lh=False, use_hl=False),
    "D_plus_lh":      dict(mask_mode="learned", use_lh=True,  use_hl=False),
    "E_full":         dict(mask_mode="learned", use_lh=True,  use_hl=True),
}
FIXED_MASK_HALF_SIDE = 10  # paper's Table 7 caption: "Fixed uses a fixed square mask with sides of 10"


class AblatableFreModule(nn.Module):
    """Reimplementation of FreModule (net/model.py's AFLB) with the mask
    strategy and FMoM gates individually switchable. Numerically identical
    to the released FreModule when mask_mode="learned", use_lh=True,
    use_hl=True (E_full) -- verified by weight-for-weight architectural
    parity, not just shape matching (see test18/scripts/verify_parity.py).
    """

    def __init__(self, dim, num_heads, bias, in_dim=3,
                 mask_mode: str = "learned", use_lh: bool = True, use_hl: bool = True):
        super().__init__()
        assert mask_mode in (None, "fixed", "learned")
        self.mask_mode = mask_mode
        self.use_lh = use_lh
        self.use_hl = use_hl

        if mask_mode is None:
            return  # A_baseline: no parameters, forward() is identity

        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        if use_lh or use_hl:
            self.frequency_refine = FreRefine(dim)
            if not use_hl:
                self.frequency_refine.SpatialGate = _IdentityGate()
            if not use_lh:
                self.frequency_refine.ChannelGate = _IdentityGate()
        else:
            self.agg_proj = nn.Conv2d(dim, dim, kernel_size=1)  # documented fallback aggregation

        if mask_mode == "learned":
            self.rate_conv = nn.Sequential(
                nn.Conv2d(dim, dim // 8, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 8, 2, 1, bias=False),
            )

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

    def shift(self, x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x, n=128):
        x = self.conv1(x)
        h, w = x.shape[-2:]
        mask = torch.zeros(x.shape, device=x.device)

        if self.mask_mode == "fixed":
            hs = ws = FIXED_MASK_HALF_SIDE
            mask[:, :, h // 2 - hs:h // 2 + hs, w // 2 - ws:w // 2 + ws] = 1
        else:  # "learned"
            threshold = F.adaptive_avg_pool2d(x, 1)
            threshold = self.rate_conv(threshold).sigmoid()
            for i in range(mask.shape[0]):
                h_ = (h // n * threshold[i, 0, :, :]).int()
                w_ = (w // n * threshold[i, 1, :, :]).int()
                mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm="forward", dim=(-2, -1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)
        high = torch.abs(torch.fft.ifft2(self.unshift(fft_high), norm="forward", dim=(-2, -1)))

        fft_low = fft * mask
        low = torch.abs(torch.fft.ifft2(self.unshift(fft_low), norm="forward", dim=(-2, -1)))

        return high, low, mask

    def forward(self, x, y, return_diagnostics: bool = False):
        if self.mask_mode is None:
            if return_diagnostics:
                return y, {"active": False}
            return y

        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode="bilinear")

        high_feature, low_feature, mask = self.fft(x)
        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        if self.use_lh or self.use_hl:
            agg = self.frequency_refine(low_feature, high_feature)
        else:
            agg = self.agg_proj(low_feature + high_feature)

        out = self.channel_cross_agg(y, agg)
        result = out * self.para1 + y * self.para2

        if return_diagnostics:
            diag = {"active": True, "mask": mask.detach(), "high_feature": high_feature.detach(),
                    "low_feature": low_feature.detach(), "agg": agg.detach(), "output": result.detach()}
            return result, diag
        return result


class _IdentityGate(nn.Module):
    """Replaces SpatialGate/ChannelGate when a FMoM gate is ablated away --
    returns a scale of 1 (no gating), so FreRefine's low*weight / high*weight
    become plain passthrough for the ablated branch."""

    def forward(self, x):
        return torch.ones(x.shape[0], x.shape[1] if isinstance(self, ChannelGate) else 1, 1, 1, device=x.device)


class AdaIRAblatable(AdaIR):
    """AdaIR with fre1/fre2/fre3 replaced by AblatableFreModule instances
    per the requested variant. Everything else (encoder/decoder/patch
    embed/refinement) is inherited from AdaIR unchanged."""

    def __init__(self, variant: str, dim=48, heads=(1, 2, 4, 8), bias=False, **kwargs):
        cfg = VARIANTS[variant]
        super().__init__(dim=dim, heads=list(heads), bias=bias, decoder=True, **kwargs)
        self.variant = variant
        self.fre1 = AblatableFreModule(dim * 2 ** 3, num_heads=heads[2], bias=bias, **cfg)
        self.fre2 = AblatableFreModule(dim * 2 ** 2, num_heads=heads[2], bias=bias, **cfg)
        self.fre3 = AblatableFreModule(dim * 2 ** 1, num_heads=heads[2], bias=bias, **cfg)


def build_variant(name: str) -> AdaIRAblatable:
    return AdaIRAblatable(name)


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    for name in VARIANTS:
        m = build_variant(name)
        with torch.no_grad():
            out = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name}: out={tuple(out.shape)} finite={torch.isfinite(out).all().item()} params={n_params:,}")
