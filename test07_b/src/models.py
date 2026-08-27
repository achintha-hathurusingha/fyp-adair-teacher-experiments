"""TEST07-B: Models A and B ONLY. Same locked NAFNet M-arm as TEST07-Pilot,
imported read-only from fyp-adair-distill (unmodified), composed via a
faithful forward-pass replica.

CORRECTION FROM PILOT: student bottleneck pooling is now GAP+GMP (512-dim),
matching the teacher's GAP+GMP (768-dim) pooling structure, then
Linear(512, 16) -- the pilot's GAP-only asymmetry is fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST07B = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07B.parent
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

    def bottleneck_pooled(self, inp: torch.Tensor) -> torch.Tensor:
        """Expose GAP+GMP bottleneck (512-dim) for the representation probe,
        available on BOTH models (Model A has no projection head, but this
        method works regardless)."""
        x, _, _ = self._encode_to_bottleneck(inp)
        return pooled_gap_gmp(x)


class ModelA(PilotNAFNetBase):
    """Baseline: plain NAFNet, no distillation, no projection head."""
    pass


class ModelB(PilotNAFNetBase):
    """+ compact latent distillation: bottleneck -> GAP+GMP(512) -> Linear(512,16)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)

    def _project_embedding(self, bottleneck):
        pooled = pooled_gap_gmp(bottleneck)  # (B, 512)
        return self.proj(pooled)  # (B, 16)


MODELS = {"A": ModelA, "B": ModelB}
