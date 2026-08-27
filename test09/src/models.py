"""TEST09: Models A/B/C/D/E/F -- multi-depth FiLM conditioning and low-rank
channel-mixing, built on the exact TEST07-B/08-C locked NAFNet M-arm, read-
only imported from fyp-adair-distill.

Decoder stage/level convention follows fyp-adair-distill's own naming
(NAFNet.stage_norm's `level = n_stages-1-stage_idx` for decoder stages):
  stage_idx 0 (decoders[0], 128ch) = "decoder level 3" (closest to bottleneck)
  stage_idx 1 (decoders[1], 64ch)  = "decoder level 2"
  stage_idx 2 (decoders[2], 32ch)  = "decoder level 1"
  stage_idx 3 (decoders[3], 16ch)  = "decoder level 0" (full resolution)

All conditioned models share ONE e_S (bottleneck -> GAP+GMP(512) ->
Linear(512,16)) for both the KD loss and every conditioning signal, per the
same design isolation used in TEST08-C.

  A: baseline, no KD, no conditioning.
  B: KD only (= validated TEST07-B config).
  C: KD + FiLM at bottleneck only (= validated TEST08-C config).
  D: KD + FiLM at bottleneck + decoder level 3.
  E: KD + FiLM at bottleneck + decoder level 3 + decoder level 2.
  F: KD + low-rank (rank 4) channel-mixing conditioning at bottleneck ONLY
     (no FiLM at the bottleneck for F -- this isolates operator expressivity
     from conditioning depth).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST09 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST09.parent
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
LOWRANK_R = 4

# stage_idx (order of net.decoders/net.ups) -> (level, channel count)
DEC_STAGE_INFO = {0: (3, 128), 1: (2, 64), 2: (1, 32), 3: (0, 16)}


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
    a zero-initialized linear head so the adaptive contribution starts at
    exactly zero (U/V themselves are randomly initialized but contribute
    nothing until a(e_S) becomes nonzero through training)."""

    def __init__(self, channels: int, pca_dim: int = PCA_DIM, rank: int = LOWRANK_R):
        super().__init__()
        self.U = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.a_head = nn.Linear(pca_dim, rank)
        zero_init_linear(self.a_head)

    def forward(self, F_in: torch.Tensor, e_s: torch.Tensor):
        Vt_F = torch.einsum("cr,bchw->brhw", self.V, F_in)
        a = self.a_head(e_s)  # (B, R)
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


class FlexModel(PilotNAFNetBase):
    def __init__(self, kd: bool = False, condition_bottleneck: bool = False,
                 condition_decoder_levels: tuple = (), lowrank_bottleneck: bool = False,
                 lowrank_rank: int = LOWRANK_R):
        super().__init__()
        self.kd = kd
        self.condition_bottleneck = condition_bottleneck
        self.condition_decoder_levels = set(condition_decoder_levels)
        self.lowrank_bottleneck = lowrank_bottleneck

        if kd:
            self.proj = nn.Linear(POOLED_DIM, PCA_DIM)

        if condition_bottleneck:
            self.bn_gamma = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
            self.bn_beta = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
            zero_init_linear(self.bn_gamma)
            zero_init_linear(self.bn_beta)

        if lowrank_bottleneck:
            self.bn_lowrank = LowRankChannelMix(BOTTLENECK_CHAN, PCA_DIM, lowrank_rank)

        self.dec_gamma = nn.ModuleDict()
        self.dec_beta = nn.ModuleDict()
        for stage_idx, (level, c) in DEC_STAGE_INFO.items():
            if level in self.condition_decoder_levels:
                g, b = nn.Linear(PCA_DIM, c), nn.Linear(PCA_DIM, c)
                zero_init_linear(g)
                zero_init_linear(b)
                self.dec_gamma[str(stage_idx)] = g
                self.dec_beta[str(stage_idx)] = b

    def _project_embedding(self, bottleneck):
        if not self.kd:
            return None
        pooled = pooled_gap_gmp(bottleneck)
        return self.proj(pooled)

    def _apply_film(self, x, gamma_head, beta_head, e_s):
        gamma = 1.0 + gamma_head(e_s)
        beta = beta_head(e_s)
        return gamma[:, :, None, None] * x + beta[:, :, None, None], gamma, beta

    def forward(self, inp: torch.Tensor):
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)

        if self.condition_bottleneck:
            x, _, _ = self._apply_film(x, self.bn_gamma, self.bn_beta, e_s)
        elif self.lowrank_bottleneck:
            x, _ = self.bn_lowrank(x, e_s)

        for stage_idx, (dec, up, skip) in enumerate(zip(net.decoders, net.ups, reversed(skips))):
            level, _ = DEC_STAGE_INFO[stage_idx]
            x = up(x)
            x = x + skip
            if level in self.condition_decoder_levels:
                x, _, _ = self._apply_film(x, self.dec_gamma[str(stage_idx)], self.dec_beta[str(stage_idx)], e_s)
            x = dec(x)

        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s

    def forward_diagnostics(self, inp: torch.Tensor):
        """Returns out, e_s, stage_diag: {stage_name: dict(F_pre, F_cond, gamma?, beta?, a?)}."""
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        stage_diag = {}

        if self.condition_bottleneck:
            F_pre = x
            x, gamma, beta = self._apply_film(x, self.bn_gamma, self.bn_beta, e_s)
            stage_diag["bottleneck"] = {"F_pre": F_pre, "F_cond": x, "gamma": gamma, "beta": beta}
        elif self.lowrank_bottleneck:
            F_pre = x
            x, a = self.bn_lowrank(x, e_s)
            stage_diag["bottleneck"] = {"F_pre": F_pre, "F_cond": x, "a": a}

        for stage_idx, (dec, up, skip) in enumerate(zip(net.decoders, net.ups, reversed(skips))):
            level, _ = DEC_STAGE_INFO[stage_idx]
            x = up(x)
            x = x + skip
            if level in self.condition_decoder_levels:
                F_pre = x
                x, gamma, beta = self._apply_film(x, self.dec_gamma[str(stage_idx)], self.dec_beta[str(stage_idx)], e_s)
                stage_diag[f"decoder_level{level}"] = {"F_pre": F_pre, "F_cond": x, "gamma": gamma, "beta": beta}
            x = dec(x)

        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s, stage_diag


MODELS = {
    "A": lambda: FlexModel(kd=False),
    "B": lambda: FlexModel(kd=True),
    "C": lambda: FlexModel(kd=True, condition_bottleneck=True),
    "D": lambda: FlexModel(kd=True, condition_bottleneck=True, condition_decoder_levels=(3,)),
    "E": lambda: FlexModel(kd=True, condition_bottleneck=True, condition_decoder_levels=(3, 2)),
    "F": lambda: FlexModel(kd=True, lowrank_bottleneck=True),
}
