"""TEST05.5 Phase 7-9 support: controlled teacher variants for the
frequency-path ablation, built as NEW code inside test05_5 (checkpoint and
test01 source are read-only; nothing here modifies them).

Source-code audit (Phase 7, done by direct inspection of
AdaIR/net/model.py's FreModule.fft(), also documented already in
test01/scripts/model_variants.py and docs/adair_source_audit.md):
  - The ONLY frequency-specific operations in the whole AdaIR forward pass
    are inside FreModule.fft(): torch.fft.fft2 -> fftshift -> box mask
    (mask=1 near DC = "low", mask=0 elsewhere = "high") -> ifftshift ->
    ifft2 -> abs(), producing (high, low).
  - Everything downstream of fft() (channel_cross_l/h = FMiM cross-attn,
    frequency_refine = FMoM H-L/L-H gating, channel_cross_agg, para1/para2
    residual mix) is ordinary spatial conv/attention -- NOT frequency-
    specific -- and is deliberately left untouched in every variant here,
    per the task's instruction not to remove entire AFLB/Transformer blocks.
  - CRITICAL PRE-ESTABLISHED FACT (TEST01, reconfirmed independently by
    TEST02-05): at benchmark resolution the learned mask half-width h_/w_
    round to 0, i.e. mask is the all-zero tensor. This means, for the
    RELEASED model on THIS dataset, high = abs(ifft(fft*1)) = conv_feat
    (an identity pass) and low = abs(ifft(fft*0)) = 0, ALWAYS. Consequently
    T0 (released) and T1 (no_frequency, from test01/model_variants.py) are
    mathematically GUARANTEED to be bit-identical here -- this is not a
    new finding, it is restated so that a T0==T1 result in this experiment
    is correctly interpreted as "confirms known mask degeneracy," not
    "surprising new evidence against F2S."
  - Because T0/T1 are degenerate, T2 and T3 below are the variants that
    actually inject a NEW frequency-domain intervention, needed for the
    ablation to be informative at all:

T2 -- MATCHED_RANDOM: the "high" tensor is replaced by Gaussian noise
    matched to the REAL high's per-channel mean/std (first+second moment
    matched, but carrying zero information about the actual image). "low"
    stays zero (matches T0's degenerate low). Tests whether the FMiM/FMoM
    machinery is doing something specific to the real conv-domain content,
    or would respond identically to any matched-statistics tensor.

T3 -- PHASE_SHUFFLE: conv_feat -> fft2 -> shift -> split into magnitude and
    phase -> phase is spatially permuted (random shuffle of phase values
    across the frequency grid) -> reconstruct complex spectrum -> unshift
    -> ifft2 -> abs() -> "high". This is a genuine frequency-domain
    intervention (unlike T0/T1, which never differ here): it preserves the
    exact power/magnitude spectrum (per-frequency energy) but destroys
    spatial/phase structure, directly testing Phase 10's question ("useful
    frequency information" vs "generic frequency statistics"). low=0 as
    above.

T4 (phase/magnitude cross-swap) is NOT implemented: since T0's magnitude
    spectrum already reduces to an identity spatial pass (mask=0 => no
    frequency-selective filtering occurs at all), a "swap magnitude from
    image A / phase from image B" experiment would be swapping between two
    already-untouched conv_feat maps' full FFTs, which collapses to a
    content-swap experiment already covered by TEST04's causal
    intervention at other tensor points, not a new frequency-specific
    question. This omission is recorded here rather than silently skipped.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "test01" / "scripts"))
from instrument import Recorder, load_adair, TRANSFORMER_STAGES  # noqa: E402
from model_variants import _fft_released, _fft_no_frequency, _instrumented_forward  # noqa: E402 (read-only reuse)

VARIANTS = ["T0_released", "T1_no_frequency", "T2_matched_random", "T3_phase_shuffle"]


def _fft_t2_matched_random(self, x, recorder: Recorder, aflb_name: str, n=128, seed=0):
    high_real, low_real = _fft_released(self, x, recorder, aflb_name, n=n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    mean = high_real.mean(dim=(2, 3), keepdim=True)
    std = high_real.std(dim=(2, 3), keepdim=True)
    noise = torch.randn(high_real.shape, generator=g).to(high_real.device)
    high = noise * std + mean
    low = torch.zeros_like(low_real)
    recorder.put(aflb_name, "raw_high", high)
    recorder.put(aflb_name, "raw_low", low)
    return high, low


def _fft_t3_phase_shuffle(self, x, recorder: Recorder, aflb_name: str, n=128, seed=0):
    conv_feat = self.conv1(x)
    recorder.put(aflb_name, "conv_feat", conv_feat)
    fft = torch.fft.fft2(conv_feat, norm="forward", dim=(-2, -1))
    fft = self.shift(fft)
    magnitude, phase = torch.abs(fft), torch.angle(fft)

    g = torch.Generator(device="cpu").manual_seed(seed)
    b, c, h, w = phase.shape
    flat = phase.reshape(b, c, h * w)
    perm = torch.stack([torch.randperm(h * w, generator=g) for _ in range(b * c)])
    perm = perm.reshape(b, c, h * w).to(phase.device)
    shuffled_phase = torch.gather(flat, 2, perm).reshape(b, c, h, w)

    shuffled_fft = magnitude * torch.exp(1j * shuffled_phase)
    high = torch.abs(torch.fft.ifft2(self.unshift(shuffled_fft), norm="forward", dim=(-2, -1)))
    low = torch.zeros_like(high)
    recorder.put(aflb_name, "raw_high", high)
    recorder.put(aflb_name, "raw_low", low)
    return high, low


_FFT_IMPLS = {
    "T0_released": _fft_released,
    "T1_no_frequency": _fft_no_frequency,
    "T2_matched_random": _fft_t2_matched_random,
    "T3_phase_shuffle": _fft_t3_phase_shuffle,
}


def load_variant(adair_repo: Path, ckpt_path: Path, device: str, variant: str, seed=0):
    assert variant in VARIANTS, variant
    model = load_adair(adair_repo, ckpt_path, device)
    recorder = Recorder()
    net = model.net if hasattr(model, "net") else model

    fft_fn = _FFT_IMPLS[variant]
    name_map = {"fre1": "AFLB1", "fre2": "AFLB2", "fre3": "AFLB3"}
    for attr, aflb_name in name_map.items():
        fre = getattr(net, attr)
        fre._variant_fft = types.MethodType(
            lambda self, x, recorder=recorder, aflb_name=aflb_name, fn=fft_fn, seed=seed: (
                fn(self, x, recorder, aflb_name, seed=seed) if fn in (_fft_t2_matched_random, _fft_t3_phase_shuffle)
                else fn(self, x, recorder, aflb_name)
            ),
            fre,
        )
        fre.forward = types.MethodType(
            lambda self, x, y, recorder=recorder, aflb_name=aflb_name, variant=variant: _instrumented_forward(
                self, x, y, recorder, aflb_name, variant),
            fre,
        )

    handles = []
    for stage_name in TRANSFORMER_STAGES:
        stage = getattr(net, stage_name)

        def _hook(module, inputs, output, stage_name=stage_name):
            recorder.put("_stages", stage_name, output)

        handles.append(stage.register_forward_hook(_hook))

    return model, recorder
