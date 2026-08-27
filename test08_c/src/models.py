"""TEST08-C: Models A (baseline), B (compact latent KD, = validated TEST07-B
config), C (KD + bottleneck FiLM-style spatial conditioning from e_S). Same
locked NAFNet M-arm as TEST07-B, imported read-only from fyp-adair-distill,
composed via the same faithful forward-pass replica pattern.

Model C conditions ONLY the deepest bottleneck (per spec): F_cond =
(1+gamma)*F + beta, gamma/beta produced by two Linear heads from e_S,
zero-initialized so the conditioning path starts as an identity transform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST08C = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST08C.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"  # sibling of teacher-experiments
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
    return torch.cat([gap, gmp], dim=1)  # (B, 2C)


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

    def _project_embedding(self, bottleneck):
        return None

    def forward(self, inp: torch.Tensor):
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        out = self._decode_from_bottleneck(x, skips, x_in, h, w)
        return out, e_s

    def bottleneck_pooled(self, inp: torch.Tensor) -> torch.Tensor:
        """Expose GAP+GMP bottleneck (512-dim) for the representation probe,
        available on ALL models regardless of projection head."""
        x, _, _ = self._encode_to_bottleneck(inp)
        return pooled_gap_gmp(x)


class ModelA(PilotNAFNetBase):
    """Baseline: plain NAFNet, no distillation, no conditioning."""
    pass


class ModelB(PilotNAFNetBase):
    """Validated TEST07-B config: bottleneck -> GAP+GMP(512) -> Linear(512,16),
    trained with L_restore + 0.1 * MSE(e_S, e_T). No spatial use of e_S."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)

    def _project_embedding(self, bottleneck):
        pooled = pooled_gap_gmp(bottleneck)  # (B, 512)
        return self.proj(pooled)  # (B, 16)


class ModelC(PilotNAFNetBase):
    """B + bottleneck FiLM-style spatial conditioning from the SAME e_S used
    for the KD loss (no separate conditioning embedding, per spec -- this
    isolates whether one learned state can both represent and control).

    F_cond = (1 + gamma) * F + beta, gamma/beta from two Linear(16->256)
    heads, zero-initialized so conditioning starts as identity."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.gamma_head = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
        self.beta_head = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def _project_embedding(self, bottleneck):
        pooled = pooled_gap_gmp(bottleneck)
        return self.proj(pooled)

    def _condition(self, x, e_s):
        gamma = 1.0 + self.gamma_head(e_s)
        beta = self.beta_head(e_s)
        x_cond = gamma[:, :, None, None] * x + beta[:, :, None, None]
        return x_cond, gamma, beta

    def forward(self, inp: torch.Tensor):
        """Standard interface (out, e_s) -- used by the shared training loop,
        identical signature to A/B."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        x_cond, _, _ = self._condition(x, e_s)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, e_s

    def forward_diagnostics(self, inp: torch.Tensor):
        """Extended interface for internal analysis: returns out, e_s, gamma,
        beta, F (pre-conditioning bottleneck), F_cond (post-conditioning)."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        x_cond, gamma, beta = self._condition(x, e_s)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, e_s, gamma, beta, x, x_cond

    def forward_with_override_embedding(self, inp: torch.Tensor, e_override: torch.Tensor):
        """Runs the normal encode pipeline, but conditions with e_override
        instead of the model's own computed e_s. Used for controls
        (random/zero/shuffled) and the donor-recipient embedding
        intervention. Returns out, gamma, beta."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        x_cond, gamma, beta = self._condition(x, e_override)
        out = self._decode_from_bottleneck(x_cond, skips, x_in, h, w)
        return out, gamma, beta


MODELS = {"A": ModelA, "B": ModelB, "C": ModelC}
