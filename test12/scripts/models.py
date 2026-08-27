"""TEST12: Models A (baseline), F2 (TEST11's rank-2 reproduced, a=G(e_D)),
T12 (feature-conditioned rank-2, a=G([e_D; phi(F)])). Built on the exact
locked NAFNet M-arm, read-only imported from fyp-adair-distill. Rank is
fixed at 2 for both F2 and T12 -- content-conditioning is the sole
manipulated variable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST12 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST12.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"
sys.path.insert(0, str(FYP_ADAIR_DISTILL))
from src.models.nafnet import NAFNet  # noqa: E402 (read-only reuse)

LOCKED_CFG = dict(
    img_channels=3, width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
    dec_blk_nums=[2, 2, 2, 2], use_gate=False, norm_type="layernorm2d",
    full_res_norm_type="affine_clamp", clamp_bound=8.0,
)
BOTTLENECK_CHAN = LOCKED_CFG["width"] * (2 ** len(LOCKED_CFG["enc_blk_nums"]))  # 256
POOLED_DIM = BOTTLENECK_CHAN * 2  # GAP+GMP = 512 (== phi(F) dim)
PCA_DIM = 16
RANK = 2


def build_base_nafnet() -> NAFNet:
    return NAFNet(
        img_channels=LOCKED_CFG["img_channels"], width=LOCKED_CFG["width"],
        enc_blk_nums=LOCKED_CFG["enc_blk_nums"], middle_blk_num=LOCKED_CFG["middle_blk_num"],
        dec_blk_nums=LOCKED_CFG["dec_blk_nums"], use_gate=LOCKED_CFG["use_gate"],
        norm_type=LOCKED_CFG["norm_type"], full_res_norm_type=LOCKED_CFG["full_res_norm_type"],
        clamp_bound=LOCKED_CFG["clamp_bound"],
    )


def pooled_gap_gmp(x: torch.Tensor) -> torch.Tensor:
    gap = x.mean(dim=(2, 3))
    gmp = x.amax(dim=(2, 3))
    return torch.cat([gap, gmp], dim=1)


def zero_init_linear(layer: nn.Linear):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class PilotNAFNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = build_base_nafnet()

    def _encode_to_bottleneck(self, inp: torch.Tensor):
        net = self.net
        x_in = net._pad(inp)
        x = net.intro(x_in)
        if net.gate is not None:
            x = net.gate(x)
        skips = []
        for enc, down in zip(net.encoders, net.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = net.middle_blks(x)
        return x, skips, x_in

    def _decode_from_bottleneck(self, x, skips, x_in, h, w):
        net = self.net
        for dec, up, skip in zip(net.decoders, net.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = net.ending(x)
        x = x + x_in
        return x[:, :, :h, :w]

    def bottleneck_pooled(self, inp: torch.Tensor) -> torch.Tensor:
        x, _, _ = self._encode_to_bottleneck(inp)
        return pooled_gap_gmp(x)


class ModelA(PilotNAFNetBase):
    def forward(self, inp: torch.Tensor):
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        out = self._decode_from_bottleneck(x, skips, x_in, h, w)
        return out, None


class ModelF2(PilotNAFNetBase):
    """TEST11's rank-2 mechanism reproduced exactly: a=G(e_D), G a single
    Linear(16,2). No content conditioning."""

    def __init__(self, rank: int = RANK):
        super().__init__()
        self.rank = rank
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.U = nn.Parameter(torch.randn(BOTTLENECK_CHAN, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(BOTTLENECK_CHAN, rank) * 0.02)
        self.a_head = nn.Linear(PCA_DIM, rank)
        zero_init_linear(self.a_head)

    def _condition(self, F_in, e_d):
        a = self.a_head(e_d)
        Vt_F = torch.einsum("cr,bchw->brhw", self.V, F_in)
        scaled = Vt_F * a[:, :, None, None]
        delta = torch.einsum("cr,brhw->bchw", self.U, scaled)
        return F_in + delta, a

    def forward(self, inp: torch.Tensor):
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_d = self.proj(pooled_gap_gmp(x))
        x, _ = self._condition(x, e_d)
        out = self._decode_from_bottleneck(x, skips, x_in, h, w)
        return out, e_d

    def forward_diagnostics(self, inp: torch.Tensor):
        """Returns out, e_d, a -- for cross-scene coefficient-variance
        comparison against T12 (does a degradation-only head already vary
        across scenes within one degradation, or is it flat?)."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_d = self.proj(pooled_gap_gmp(x))
        x_cond, a = self._condition(x, e_d)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, e_d, a


class ModelT12(PilotNAFNetBase):
    """Feature-conditioned rank-2: a = G([e_D; phi(F)]), phi(F)=GAP+GMP(F)
    (512-dim, == pooled_gap_gmp(bottleneck)). G is a small MLP
    528 -> 32 -> rank, final layer zero-initialized so a=0 (identity) at
    init. Same low-rank operator (U,V, rank=2) as F2 -- content
    conditioning is the sole manipulated variable."""

    def __init__(self, rank: int = RANK):
        super().__init__()
        self.rank = rank
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.U = nn.Parameter(torch.randn(BOTTLENECK_CHAN, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(BOTTLENECK_CHAN, rank) * 0.02)
        self.coeff_head = nn.Sequential(
            nn.Linear(PCA_DIM + POOLED_DIM, 32), nn.ReLU(inplace=True), nn.Linear(32, rank))
        zero_init_linear(self.coeff_head[2])

    def _condition(self, F_in, e_d, phi):
        q = torch.cat([e_d, phi], dim=1)
        a = self.coeff_head(q)
        Vt_F = torch.einsum("cr,bchw->brhw", self.V, F_in)
        scaled = Vt_F * a[:, :, None, None]
        delta = torch.einsum("cr,brhw->bchw", self.U, scaled)
        return F_in + delta, a

    def forward(self, inp: torch.Tensor):
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        phi = pooled_gap_gmp(x)
        e_d = self.proj(phi)
        x, _ = self._condition(x, e_d, phi)
        out = self._decode_from_bottleneck(x, skips, x_in, h, w)
        return out, e_d

    def forward_diagnostics(self, inp: torch.Tensor):
        """Returns out, e_d, a, phi, F_pre, F_cond."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        phi = pooled_gap_gmp(x)
        e_d = self.proj(phi)
        F_pre = x
        x_cond, a = self._condition(x, e_d, phi)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, e_d, a, phi, F_pre, x_cond

    def forward_with_override(self, inp: torch.Tensor, e_d_override=None, phi_override=None):
        """Normal encode, but condition using overridden e_d and/or phi --
        for the mandatory causal controls (degradation-only, content-only,
        shuffled-content)."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        phi_normal = pooled_gap_gmp(x)
        e_d_normal = self.proj(phi_normal)
        e_d_used = e_d_override if e_d_override is not None else e_d_normal
        phi_used = phi_override if phi_override is not None else phi_normal
        x_cond, a = self._condition(x, e_d_used, phi_used)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, a


MODELS = {"A": lambda: ModelA(), "F2": lambda: ModelF2(), "T12": lambda: ModelT12()}
