"""TEST07-Pilot: Models A-D. All four wrap the SAME locked NAFNet (M arm,
imported READ-ONLY from fyp-adair-distill, never modified) via a faithful
forward-pass replica that taps the bottleneck tensor -- the same
"manual_forward" pattern established in test04/test05_5/test06 for AdaIR,
applied here to NAFNet since fyp-adair-distill's forward() does not expose
intermediate tensors and must not be edited.

Model A: plain NAFNet, no distillation.
Model B: + compact latent distillation (bottleneck -> 16-dim projection,
          MSE against teacher's PCA-16 embedding).
Model C: B + lightweight FiLM-style affine conditioning of the bottleneck,
          initialized near-identity (gamma~1, beta~0).
Model D: B + low-rank (R=2) dynamic spatial kernel applied at ONE location
          (the bottleneck), K(e) = K_0 + sum_r a_r(e) K_r.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TEST07 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"  # sibling of teacher-experiments, not inside it
sys.path.insert(0, str(FYP_ADAIR_DISTILL))
from src.models.nafnet import NAFNet, NAFBlock  # noqa: E402 (read-only reuse)

LOCKED_CFG = dict(
    img_channels=3, width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
    dec_blk_nums=[2, 2, 2, 2], use_gate=False, norm_type="layernorm2d",
    full_res_norm_type="affine_clamp", clamp_bound=8.0,
)
BOTTLENECK_CHAN = LOCKED_CFG["width"] * (2 ** len(LOCKED_CFG["enc_blk_nums"]))  # 16*16=256
PCA_DIM = 16


def build_base_nafnet() -> NAFNet:
    return NAFNet(
        img_channels=LOCKED_CFG["img_channels"], width=LOCKED_CFG["width"],
        enc_blk_nums=LOCKED_CFG["enc_blk_nums"], middle_blk_num=LOCKED_CFG["middle_blk_num"],
        dec_blk_nums=LOCKED_CFG["dec_blk_nums"], use_gate=LOCKED_CFG["use_gate"],
        norm_type=LOCKED_CFG["norm_type"], full_res_norm_type=LOCKED_CFG["full_res_norm_type"],
        clamp_bound=LOCKED_CFG["clamp_bound"],
    )


class PilotNAFNetBase(nn.Module):
    """Faithful replica of NAFNet.forward(), composed around an internal
    (unmodified) NAFNet instance, with a tap at the bottleneck. Subclasses
    override `_modulate_bottleneck` to inject Model C/D mechanisms."""

    def __init__(self):
        super().__init__()
        self.net = build_base_nafnet()

    def _modulate_bottleneck(self, x: torch.Tensor, e_s: torch.Tensor | None) -> torch.Tensor:
        return x  # Model A/B: no modulation

    def _project_embedding(self, bottleneck: torch.Tensor) -> torch.Tensor | None:
        return None  # Model A: no embedding

    def _encode_to_bottleneck(self, inp: torch.Tensor):
        """Shared encoder path, exposed separately so callers (e.g. the
        representation probe) can get the raw bottleneck tensor even for
        Model A, which has no projection head."""
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

    def forward(self, inp: torch.Tensor):
        net = self.net
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        bottleneck = x
        e_s = self._project_embedding(bottleneck)
        x = self._modulate_bottleneck(x, e_s)

        for dec, up, skip in zip(net.decoders, net.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = net.ending(x)
        x = x + x_in
        out = x[:, :, :h, :w]
        return out, e_s


class ModelA(PilotNAFNetBase):
    """Baseline: plain NAFNet, no distillation."""
    pass


class ModelB(PilotNAFNetBase):
    """+ compact latent distillation: bottleneck -> GAP -> Linear(256,16)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(BOTTLENECK_CHAN, PCA_DIM)

    def _project_embedding(self, bottleneck):
        pooled = bottleneck.mean(dim=(2, 3))  # GAP, (B, 256)
        return self.proj(pooled)  # (B, 16)


class ModelC(ModelB):
    """B + FiLM-style channel-wise affine conditioning, near-identity init."""

    def __init__(self):
        super().__init__()
        self.gamma_head = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
        self.beta_head = nn.Linear(PCA_DIM, BOTTLENECK_CHAN)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)  # gamma = 1 + 0 = 1 at init
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)   # beta = 0 at init

    def _modulate_bottleneck(self, x, e_s):
        gamma = 1.0 + self.gamma_head(e_s)  # (B, C)
        beta = self.beta_head(e_s)          # (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ModelD(ModelB):
    """B + low-rank (R=2) dynamic spatial kernel at the bottleneck.
    K(e) = K_0 + sum_r a_r(e) K_r, applied as one extra depthwise 3x3 conv
    whose weights are a learned base plus a data-dependent low-rank
    correction -- a single location only, per the task's explicit scope
    limit ("do NOT place dynamic kernels throughout the network")."""

    RANK = 2

    def __init__(self):
        super().__init__()
        c = BOTTLENECK_CHAN
        self.k0 = nn.Parameter(torch.zeros(c, 1, 3, 3))
        nn.init.kaiming_uniform_(self.k0, a=5 ** 0.5)
        self.kr = nn.Parameter(torch.zeros(self.RANK, c, 1, 3, 3))
        nn.init.normal_(self.kr, std=0.01)  # small init: correction starts near-inert
        self.coef_head = nn.Linear(PCA_DIM, self.RANK)
        nn.init.zeros_(self.coef_head.weight)
        nn.init.zeros_(self.coef_head.bias)  # a_r(e)=0 at init -> K(e)=K_0 only

    def _modulate_bottleneck(self, x, e_s):
        b, c, hh, ww = x.shape
        a = self.coef_head(e_s)  # (B, R)
        # Per-sample dynamic kernel: K = k0 + sum_r a_r * kr[r]. Applied via
        # grouped conv per-sample (batched using F.conv2d with groups=B*C).
        k = self.k0.unsqueeze(0) + torch.einsum("br,rcokl->bcokl", a, self.kr)  # (B,C,1,3,3)
        k = k.reshape(b * c, 1, 3, 3)
        x_r = x.reshape(1, b * c, hh, ww)
        out = F.conv2d(x_r, k, padding=1, groups=b * c)
        out = out.reshape(b, c, hh, ww)
        return x + out  # residual: starts as identity (a_r=0 -> extra conv output governed by tiny kr init)


MODELS = {"A": ModelA, "B": ModelB, "C": ModelC, "D": ModelD}
