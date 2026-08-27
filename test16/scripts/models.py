"""TEST16: the four full student models under hardware evaluation.

A  -- TEST12's baseline NAFNet, TRAINED (checkpoint reused read-only from
      test12/results/checkpoints/model_A_seed*.pt).
F2 -- TEST12's validated rank-2 low-rank conditional operator, TRAINED
      (model_F2_seed*.pt). This is Minura's actual mechanism.
N  -- Normalization-surgery baseline: identical to A except the internal
      per-block norm_type is switched from "layernorm2d" to "affine_clamp"
      (fyp-adair-distill/src/models/norms.py's own documented axis). NOT
      trained this pass -- per norms.py's own stated convention, NPU
      latency does not depend on weights, so this is exported/profiled
      untrained; PSNR/SSIM are reported as not-measured for N.
S  -- Static-mixture reinterpretation of Minura's operator (TEST15's
      finding: only runtime-generated CONV WEIGHTS are the NPU risk, not
      runtime scalar coefficients). Three static (compile-time-constant)
      1x1-conv branches at the bottleneck, mixed by scalar coefficients
      alpha = G(e_D, phi(F)) computed at runtime -- structurally the
      TEST15 `static_mixture_op`, dropped into the exact TEST12 shape/
      conditioning convention (content+degradation, T12-style) so it's a
      fair architectural comparison against F2. NOT trained this pass, for
      the same reason as N -- PSNR/SSIM reported as not-measured.

Both N and S are still REAL forward-computable modules (correct shapes,
finite outputs) -- they are just randomly initialized rather than
gradient-trained, which is sufficient for ONNX export, NPU compilation,
and latency/memory profiling (weight values do not change graph structure
or NPU op scheduling), but not sufficient for restoration-quality claims.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

TEST16 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST16.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"
TEST12_SCRIPTS = TEACHER_EXP / "test12" / "scripts"

sys.path.insert(0, str(FYP_ADAIR_DISTILL))

from src.models.nafnet import NAFNet  # noqa: E402 (read-only reuse)

# Read-only reuse of TEST12's exact, validated A and F2 classes + constants.
# Loaded under an explicit module name (not `models`) to avoid colliding with
# this file's own module name when both directories are on sys.path.
_spec = importlib.util.spec_from_file_location("test12_models", TEST12_SCRIPTS / "models.py")
_test12_models = importlib.util.module_from_spec(_spec)
sys.modules["test12_models"] = _test12_models
_spec.loader.exec_module(_test12_models)

ModelA = _test12_models.ModelA
ModelF2 = _test12_models.ModelF2
PilotNAFNetBase = _test12_models.PilotNAFNetBase
LOCKED_CFG = _test12_models.LOCKED_CFG
BOTTLENECK_CHAN = _test12_models.BOTTLENECK_CHAN
POOLED_DIM = _test12_models.POOLED_DIM
PCA_DIM = _test12_models.PCA_DIM
RANK = _test12_models.RANK
pooled_gap_gmp = _test12_models.pooled_gap_gmp
zero_init_linear = _test12_models.zero_init_linear
build_base_nafnet = _test12_models.build_base_nafnet

TEST12_CKPT_DIR = TEACHER_EXP / "test12" / "results" / "checkpoints"

# ---------------------------------------------------------------------------
# Model N: normalization surgery. Internal per-block norm_type switched from
# layernorm2d -> affine_clamp. full_res_norm_type left None: nafnet.py raises
# if full_res_norm_type duplicates an already-affine_clamp norm_type at
# stage 0 (redundant-config guard), so the boundary stages simply inherit
# the (now affine_clamp) block norm_type directly.
# ---------------------------------------------------------------------------
N_CFG = dict(LOCKED_CFG)
N_CFG["norm_type"] = "affine_clamp"
N_CFG["full_res_norm_type"] = None


def build_norm_surgery_nafnet() -> NAFNet:
    return NAFNet(
        img_channels=N_CFG["img_channels"], width=N_CFG["width"],
        enc_blk_nums=N_CFG["enc_blk_nums"], middle_blk_num=N_CFG["middle_blk_num"],
        dec_blk_nums=N_CFG["dec_blk_nums"], use_gate=N_CFG["use_gate"],
        norm_type=N_CFG["norm_type"], full_res_norm_type=N_CFG["full_res_norm_type"],
        clamp_bound=N_CFG["clamp_bound"],
    )


class ModelN(nn.Module):
    """Model N: baseline architecture, affine_clamp norm throughout (no
    LayerNorm2d anywhere in the graph). Untrained -- see module docstring.
    """

    def __init__(self):
        super().__init__()
        self.net = build_norm_surgery_nafnet()

    def forward(self, inp: torch.Tensor):
        return self.net(inp), None


# ---------------------------------------------------------------------------
# Model S: static-mixture operator. Three static (compile-time-constant)
# 1x1 convolutions at the bottleneck; only the scalar mixing coefficients
# are computed at runtime from [e_D, phi(F)] (T12-style conditioning).
# Structurally identical to TEST15's `static_mixture_op`, which measured
# 74ms in isolation -- identical to Minura's actual low-rank operator.
# ---------------------------------------------------------------------------
N_EXPERTS = 3


class ModelS(PilotNAFNetBase):
    def __init__(self, n_experts: int = N_EXPERTS):
        super().__init__()
        self.n_experts = n_experts
        self.proj = nn.Linear(POOLED_DIM, PCA_DIM)
        self.experts = nn.ModuleList(
            [nn.Conv2d(BOTTLENECK_CHAN, BOTTLENECK_CHAN, kernel_size=1) for _ in range(n_experts)]
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(PCA_DIM + POOLED_DIM, 32), nn.ReLU(inplace=True), nn.Linear(32, n_experts)
        )
        zero_init_linear(self.coeff_head[2])

    def _condition(self, F_in, e_d, phi):
        q = torch.cat([e_d, phi], dim=1)
        alpha = self.coeff_head(q)
        out = F_in
        for i, expert in enumerate(self.experts):
            out = out + alpha[:, i, None, None, None] * expert(F_in)
        return out, alpha

    def forward(self, inp: torch.Tensor):
        _, _, h, w = inp.shape
        x, skips, x_in = self._encode_to_bottleneck(inp)
        phi = pooled_gap_gmp(x)
        e_d = self.proj(phi)
        x, _ = self._condition(x, e_d, phi)
        out = self._decode_from_bottleneck(x, skips, x_in, h, w)
        return out, e_d


MODELS = {"A": ModelA, "F2": ModelF2, "N": ModelN, "S": ModelS}
TRAINED_MODELS = {"A", "F2"}  # only these load real checkpoints; N/S are architecture-only this pass


def load_trained(name: str, seed: int = 0, device: str = "cpu"):
    """Load a TRAINED model (A or F2) from TEST12's checkpoints. Raises for
    N/S -- they have no trained weights (see module docstring / TEST16
    Phase-14 scoping decision)."""
    if name not in TRAINED_MODELS:
        raise ValueError(f"{name} has no trained checkpoint this pass (untrained, latency-only).")
    model = MODELS[name]()
    state = torch.load(TEST12_CKPT_DIR / f"model_{name}_seed{seed}.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_untrained(name: str, seed: int = 0, device: str = "cpu"):
    """Build an untrained model (N or S) with a fixed seed for reproducible
    export/profiling."""
    torch.manual_seed(seed)
    model = MODELS[name]()
    model.to(device)
    model.eval()
    return model
