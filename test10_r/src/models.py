"""TEST10-R: Models A/F/G, built on the exact TEST07-B/08-C/09/10 locked
NAFNet M-arm, read-only imported from fyp-adair-distill.

  A: baseline, no KD, no conditioning.
  F: TEST09's best mechanism reproduced -- compact latent KD + low-rank
     (rank=2) channel-mixing conditioning at the bottleneck.
  G: F + CORRECTED restoration-trajectory distillation. Student-side stage
     projections are SIMPLE LINEAR ONLY (no MLP, no BatchNorm, no
     trainable normalization -- per this task's explicit design
     constraint). Teacher-side targets are FIXED (precomputed, leakage-safe
     PCA-32 per stage, fit on training crops only -- see
     build_teacher_targets.py) and never touched by backprop; they are
     looked up from a cache during training, exactly like the existing
     16-dim KD embedding, NOT computed by an online/jointly-trained
     teacher-side head (that joint-training design is what collapsed in
     TEST10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
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
LOWRANK_R = 2
TRAJ_DIM = 32

# student decoder stage_idx -> channel count (decoders[0]=16x16, [1]=32x32, [2]=64x64)
STUDENT_TRAJ_STAGES = {0: 128, 1: 64, 2: 32}


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

    def __init__(self, channels: int, pca_dim: int = PCA_DIM, rank: int = LOWRANK_R):
        super().__init__()
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
    def __init__(self, lowrank_rank: int = LOWRANK_R):
        super().__init__()
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.bn_lowrank = LowRankChannelMix(BOTTLENECK_CHAN, PCA_DIM, lowrank_rank)

    def _project_embedding(self, bottleneck):
        pooled = pooled_gap_gmp(bottleneck)
        return self.proj(pooled)

    def _decode_with_stages(self, x, skips, x_in, h, w, collect_stages=False):
        net = self.net
        stage_feats = {}
        for stage_idx, (dec, up, skip) in enumerate(zip(net.decoders, net.ups, reversed(skips))):
            x = up(x)
            x = x + skip
            if collect_stages and stage_idx in STUDENT_TRAJ_STAGES:
                stage_feats[stage_idx] = x
            x = dec(x)
        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, stage_feats

    def forward(self, inp: torch.Tensor):
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        x, _ = self.bn_lowrank(x, e_s)
        out, _ = self._decode_with_stages(x, skips, x_in, h, w, collect_stages=False)
        return out, e_s


class ModelG(ModelF):
    """F + CORRECTED trajectory distillation. Student-side stage
    projections are simple Linear(pooled_dim -> 32) heads ONLY -- no MLP,
    no BatchNorm. Compared against FIXED (precomputed, frozen) teacher PCA
    targets, looked up in train.py -- this class has no knowledge of the
    teacher at all."""

    def __init__(self, lowrank_rank: int = LOWRANK_R, traj_dim: int = TRAJ_DIM):
        super().__init__(lowrank_rank)
        self.traj_heads = nn.ModuleDict({
            str(stage_idx): nn.Linear(c * 2, traj_dim)  # GAP+GMP -> 2*c
            for stage_idx, c in STUDENT_TRAJ_STAGES.items()
        })

    def forward_trajectory(self, inp: torch.Tensor):
        """Returns out, e_s, {stage_idx: e_s_traj (B,32)} -- TRAINING ONLY."""
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        e_s = self._project_embedding(x)
        x, _ = self.bn_lowrank(x, e_s)
        out, stage_feats = self._decode_with_stages(x, skips, x_in, h, w, collect_stages=True)
        e_s_traj = {idx: self.traj_heads[str(idx)](pooled_gap_gmp(feat))
                    for idx, feat in stage_feats.items()}
        return out, e_s, e_s_traj


MODELS = {
    "A": lambda: ModelA(),
    "F": lambda: ModelF(),
    "G": lambda: ModelG(),
}
