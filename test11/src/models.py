"""TEST11: Models A (baseline) and F{2,4,8,16} -- TEST09's compact-latent-KD
+ low-rank channel-mixing mechanism, with ONLY the rank R varied. Built on
the exact locked NAFNet M-arm, read-only imported from fyp-adair-distill.
No FiLM, no trajectory distillation, no extra decoder conditioning -- rank
is the sole manipulated variable, isolating conditional-operator capacity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST11 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST11.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"
sys.path.insert(0, str(FYP_ADAIR_DISTILL))
from src.models.nafnet import NAFNet  # noqa: E402 (read-only reuse)

LOCKED_CFG = dict(
    img_channels=3, width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
    dec_blk_nums=[2, 2, 2, 2], use_gate=False, norm_type="layernorm2d",
    full_res_norm_type="affine_clamp", clamp_bound=8.0,
)
BOTTLENECK_CHAN = LOCKED_CFG["width"] * (2 ** len(LOCKED_CFG["enc_blk_nums"]))  # 256
POOLED_DIM = BOTTLENECK_CHAN * 2  # GAP+GMP = 512
PCA_DIM = 16
RANKS = [2, 4, 8, 16]


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


class LowRankChannelMix(nn.Module):
    """F' = F + U diag(a(e_S)) V^T F, rank R, U/V shared+learned, a(e_S) from
    a zero-initialized linear head (adaptive contribution starts at zero)."""

    def __init__(self, channels: int, pca_dim: int = PCA_DIM, rank: int = 2):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.a_head = nn.Linear(pca_dim, rank)
        zero_init_linear(self.a_head)

    def forward(self, F_in: torch.Tensor, e_s: torch.Tensor):
        Vt_F = torch.einsum("cr,bchw->brhw", self.V, F_in)
        a = self.a_head(e_s)
        scaled = Vt_F * a[:, :, None, None]
        delta = torch.einsum("cr,brhw->bchw", self.U, scaled)
        return F_in + delta, a


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

    def bottleneck_pooled(self, inp: torch.Tensor) -> torch.Tensor:
        x, _, _ = self._encode_to_bottleneck(inp)
        return pooled_gap_gmp(x)


class ModelA(PilotNAFNetBase):
    def _project_embedding(self, bottleneck):
        return None

    def forward(self, inp: torch.Tensor):
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        for dec, up, skip in zip(net.decoders, net.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s


class ModelF(PilotNAFNetBase):
    """Compact latent KD + low-rank channel-mixing conditioning at the
    bottleneck. Rank R is the only manipulated variable across TEST11."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.bn_lowrank = LowRankChannelMix(BOTTLENECK_CHAN, PCA_DIM, rank)

    def _project_embedding(self, bottleneck):
        pooled = pooled_gap_gmp(bottleneck)
        return self.proj(pooled)

    def forward(self, inp: torch.Tensor):
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        x, _ = self.bn_lowrank(x, e_s)
        for dec, up, skip in zip(net.decoders, net.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s

    def forward_diagnostics(self, inp: torch.Tensor):
        """Returns out, e_s, a (coefficients), F_pre, F_cond -- for
        modulation/coefficient/effective-rank analysis."""
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        F_pre = x
        x, a = self.bn_lowrank(x, e_s)
        F_cond = x
        for dec, up, skip in zip(net.decoders, net.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s, a, F_pre, F_cond


MODELS = {
    "A": lambda: ModelA(),
    "F2": lambda: ModelF(rank=2),
    "F4": lambda: ModelF(rank=4),
    "F8": lambda: ModelF(rank=8),
    "F16": lambda: ModelF(rank=16),
}
RANK_OF = {"F2": 2, "F4": 4, "F8": 8, "F16": 16}
