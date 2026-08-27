"""TEST15: isolated-operator and small-combination zoo for the Snapdragon
NPU benchmark. Every module is a minimal nn.Module with a fixed input
shape and an ONNX-exportable forward(), covering:

  - the base operator table (Conv3x3, DWConv, 1x1Conv, Add, Multiply,
    ReLU/clamp, Sigmoid, Softmax, GAP, GMP, Resize, Concat, elementwise
    affine, LayerNorm2D, RMSNorm-like, dynamic conv, FFT)
  - the required small combinations (Conv->Add, Conv->Clamp, Conv->Mul,
    Conv->Add->Clamp, DWConv->Pointwise, 1x1->Add->Clamp, GAP->Linear->Mul)
  - a STATIC-vs-DYNAMIC weight pair for convolution (the crux of the
    "Minura idea" question)
  - Minura's ACTUAL validated operator (TEST09-14's low-rank channel
    mixing, U diag(a(e_D)) V^T) as a direct candidate
  - the proposed NPU-native static-mixture alternative
    (Y = sum_i a_i(e_D) * B_i(F), B_i static 1x1 convs) as the
    hardware-informed redesign candidate

Each entry in OP_ZOO is (name, module_instance, tuple_of_dummy_inputs,
tuple_of_input_names). Multi-input modules (the two Minura-family
operators) take a second input e_D (a small per-sample vector) to
represent the degradation-embedding-driven conditioning signal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ---- representative shapes ----
C_MID = 64     # generic mid-network channel count
HW_MID = 32    # generic mid-network spatial size
C_BOTTLENECK = 256  # matches TEST09-14's NAFNet bottleneck channel count
HW_BOTTLENECK = 8   # matches the 128px-crop bottleneck spatial size
RANK = 2
E_DIM = 16     # degradation embedding dim, matches TEST07-B onward


# =========================================================================
# Base operator table
# =========================================================================

class OpConv3x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class OpDepthwiseConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1, groups=C_MID)

    def forward(self, x):
        return self.conv(x)


class OpConv1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 1)

    def forward(self, x):
        return self.conv(x)


class OpAdd(nn.Module):
    def forward(self, x, y):
        return x + y


class OpMultiply(nn.Module):
    def forward(self, x, y):
        return x * y


class OpReluClamp(nn.Module):
    def forward(self, x):
        return torch.clamp(F.relu(x), max=8.0)


class OpSigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class OpSoftmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class OpGlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class OpGlobalMaxPool(nn.Module):
    def forward(self, x):
        return x.amax(dim=(2, 3), keepdim=True)


class OpResize(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class OpConcatenate(nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=1)


class OpElementwiseAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, C_MID, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, C_MID, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class OpLayerNorm2D(nn.Module):
    """Standard channel-wise LayerNorm over (C,H,W), matching the
    normalization form used throughout the locked NAFNet (read-only
    reused architecture description from fyp-adair-distill, reimplemented
    here minimally for isolated NPU benchmarking -- NOT importing the
    actual module, to keep this export self-contained)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C_MID))
        self.bias = nn.Parameter(torch.zeros(C_MID))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x_n = (x - mu) / torch.sqrt(var + 1e-6)
        return x_n * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class OpRMSNormLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C_MID))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-6)
        return (x / rms) * self.weight.view(1, -1, 1, 1)


class OpDynamicConv(nn.Module):
    """Literal per-sample dynamic-weight convolution: the 3x3 kernel is
    generated at runtime from a small input vector, then applied via
    F.conv2d. This is the specific pattern flagged as likely problematic
    for NPU compilation (rank>=5 weight tensor with a batch dimension,
    runtime-computed weights)."""

    def __init__(self):
        super().__init__()
        self.groups = 8  # in_channels/groups=8, out_channels=C_MID -> weight (C_MID, 8, 3, 3)
        self.in_per_group = C_MID // self.groups
        self.weight_gen = nn.Linear(E_DIM, C_MID * self.in_per_group * 3 * 3)
        nn.init.zeros_(self.weight_gen.bias)

    def forward(self, x, e_d):
        b = x.shape[0]
        w = self.weight_gen(e_d).view(b, C_MID, self.in_per_group, 3, 3)
        # NPU/ONNX export path: batch-dim-in-weight conv is normally done
        # via grouped conv trick (fold batch into groups) -- kept literal
        # here deliberately, to test whether THIS naive per-sample-loop
        # form compiles at all (it is the most direct translation of
        # "generate a conv kernel from e_D and apply it").
        outs = []
        for i in range(b):
            outs.append(F.conv2d(x[i:i + 1], w[i], padding=1, groups=self.groups))
        return torch.cat(outs, dim=0)


class OpFFT(nn.Module):
    def forward(self, x):
        spec = torch.fft.fft2(x)
        return spec.real ** 2 + spec.imag ** 2


# =========================================================================
# Small combinations
# =========================================================================

class ComboConvAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1)

    def forward(self, x, y):
        return self.conv(x) + y


class ComboConvClamp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1)

    def forward(self, x):
        return torch.clamp(self.conv(x), -8.0, 8.0)


class ComboConvMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1)

    def forward(self, x, y):
        return self.conv(x) * y


class ComboConvAddClamp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 3, padding=1)

    def forward(self, x, y):
        return torch.clamp(self.conv(x) + y, -8.0, 8.0)


class ComboDWConvPointwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(C_MID, C_MID, 3, padding=1, groups=C_MID)
        self.pw = nn.Conv2d(C_MID, C_MID, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Combo1x1AddClamp(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(C_MID, C_MID, 1)

    def forward(self, x, y):
        return torch.clamp(self.conv(x) + y, -8.0, 8.0)


class ComboGAPLinearMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C_MID, C_MID)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        gate = torch.sigmoid(self.fc(pooled)).view(x.shape[0], C_MID, 1, 1)
        return x * gate


# =========================================================================
# Minura's ACTUAL operator (TEST09-14, read-only-derived math, reimplemented
# minimally here for isolated export -- NOT importing test-directory code,
# per every prior experiment's directory-isolation rule) vs the proposed
# NPU-native static-mixture redesign
# =========================================================================

class MinuraLowRankOp(nn.Module):
    """F' = F + U diag(a(e_D)) V^T F, exactly TEST09-14's validated
    mechanism -- the actual candidate operator under scrutiny."""

    def __init__(self, channels=C_BOTTLENECK, rank=RANK):
        super().__init__()
        self.U = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(channels, rank) * 0.02)
        self.a_head = nn.Linear(E_DIM, rank)

    def forward(self, F_in, e_d):
        a = self.a_head(e_d)
        Vt_F = torch.einsum("cr,bchw->brhw", self.V, F_in)
        scaled = Vt_F * a[:, :, None, None]
        delta = torch.einsum("cr,brhw->bchw", self.U, scaled)
        return F_in + delta


class StaticMixtureOp(nn.Module):
    """Proposed NPU-native redesign: Y = F + sum_i a_i(e_D) * B_i(F),
    where each B_i is a STATIC (fixed-weight, ordinary) 1x1 conv -- only
    the per-sample scalar/channel mixing coefficients are dynamic, not
    any convolution weight itself."""

    def __init__(self, channels=C_BOTTLENECK, n_experts=RANK + 1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(n_experts)])
        self.coeff_head = nn.Linear(E_DIM, n_experts)

    def forward(self, F_in, e_d):
        a = self.coeff_head(e_d)  # (B, n_experts)
        out = F_in
        for i, expert in enumerate(self.experts):
            out = out + a[:, i, None, None, None] * expert(F_in)
        return out


def _mid_input():
    return torch.randn(1, C_MID, HW_MID, HW_MID)


def _bottleneck_inputs():
    return torch.randn(1, C_BOTTLENECK, HW_BOTTLENECK, HW_BOTTLENECK), torch.randn(1, E_DIM)


# name -> (module, dummy_inputs_tuple, input_names_tuple)
OP_ZOO = {
    # base operator table
    "conv3x3": (OpConv3x3(), (_mid_input(),), ("x",)),
    "depthwise_conv": (OpDepthwiseConv(), (_mid_input(),), ("x",)),
    "conv1x1": (OpConv1x1(), (_mid_input(),), ("x",)),
    "add": (OpAdd(), (_mid_input(), _mid_input()), ("x", "y")),
    "multiply": (OpMultiply(), (_mid_input(), _mid_input()), ("x", "y")),
    "relu_clamp": (OpReluClamp(), (_mid_input(),), ("x",)),
    "sigmoid": (OpSigmoid(), (_mid_input(),), ("x",)),
    "softmax": (OpSoftmax(), (_mid_input(),), ("x",)),
    "global_avg_pool": (OpGlobalAvgPool(), (_mid_input(),), ("x",)),
    "global_max_pool": (OpGlobalMaxPool(), (_mid_input(),), ("x",)),
    "resize": (OpResize(), (_mid_input(),), ("x",)),
    "concatenate": (OpConcatenate(), (_mid_input(), _mid_input()), ("x", "y")),
    "elementwise_affine": (OpElementwiseAffine(), (_mid_input(),), ("x",)),
    "layernorm2d": (OpLayerNorm2D(), (_mid_input(),), ("x",)),
    "rmsnorm_like": (OpRMSNormLike(), (_mid_input(),), ("x",)),
    "dynamic_conv": (OpDynamicConv(), (_mid_input(), torch.randn(1, E_DIM)), ("x", "e_d")),
    "fft": (OpFFT(), (torch.randn(1, 1, HW_MID, HW_MID),), ("x",)),

    # small combinations
    "combo_conv_add": (ComboConvAdd(), (_mid_input(), _mid_input()), ("x", "y")),
    "combo_conv_clamp": (ComboConvClamp(), (_mid_input(),), ("x",)),
    "combo_conv_mul": (ComboConvMul(), (_mid_input(), _mid_input()), ("x", "y")),
    "combo_conv_add_clamp": (ComboConvAddClamp(), (_mid_input(), _mid_input()), ("x", "y")),
    "combo_dwconv_pointwise": (ComboDWConvPointwise(), (_mid_input(),), ("x",)),
    "combo_1x1_add_clamp": (Combo1x1AddClamp(), (_mid_input(), _mid_input()), ("x", "y")),
    "combo_gap_linear_mul": (ComboGAPLinearMul(), (_mid_input(),), ("x",)),

    # Minura candidate operators (the actual research question)
    "minura_lowrank_op": (MinuraLowRankOp(), _bottleneck_inputs(), ("F", "e_d")),
    "static_mixture_op": (StaticMixtureOp(), _bottleneck_inputs(), ("F", "e_d")),
}
