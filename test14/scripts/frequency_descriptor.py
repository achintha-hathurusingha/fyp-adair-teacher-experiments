"""TEST14: compact 8-band radial frequency descriptor q_F, computed
INDEPENDENTLY from the degraded RGB input -- explicitly NOT reusing any of
AdaIR's internal FFT tensors (raw_low/raw_high/M_l/M_h/FMiM/FMoM), per the
task's explicit prohibition. TEST06-R already showed AdaIR's own frequency
branch is causally irrelevant to its output; this is a deliberately
independent, from-scratch descriptor, testing a different question.

Steps: RGB -> luminance -> |FFT2|^2 -> fftshift -> normalize by total
energy -> sum into 8 radial bands, normalized so band edges are in units
of Nyquist (radius / (N/2), so the horizontal/vertical axis edge = 1.0 =
Nyquist; the four corners extend slightly beyond to sqrt(2), which is
intentionally excluded from all 8 bands -- see note in descriptor_audit.py
about sum(q_F) not always being exactly 1).
"""
from __future__ import annotations

import torch

N_BANDS = 8
BAND_EDGES = [0.0, 1 / 16, 2 / 16, 3 / 16, 4 / 16, 6 / 16, 8 / 16, 12 / 16, 1.0]  # in units of Nyquist


def _build_band_masks(size: int, device) -> torch.Tensor:
    """Returns (N_BANDS, size, size) boolean masks, radius normalized so the
    horizontal/vertical edge = 1.0 (Nyquist)."""
    coords = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx ** 2 + yy ** 2)  # center=0, axis edge=1.0, corner=sqrt(2)
    masks = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        masks.append((radius >= lo) & (radius < hi))
    return torch.stack(masks, dim=0)  # (8, size, size)


_MASK_CACHE = {}


def compute_qF(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1]. Returns q_F: (B, 8), each row sums to
    approximately (not exactly) 1 -- see module docstring."""
    b, c, h, w = x.shape
    assert h == w, "frequency descriptor assumes square crops"
    key = (h, x.device)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _build_band_masks(h, x.device)
    masks = _MASK_CACHE[key]  # (8, H, W)

    y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)
    spec = torch.fft.fft2(y)
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    s = spec.real ** 2 + spec.imag ** 2  # |FFT|^2, (B, H, W)
    s_norm = s / (s.sum(dim=(-2, -1), keepdim=True) + eps)  # (B, H, W)

    # sum s_norm inside each band -> (B, 8)
    q_f = torch.einsum("bhw,khw->bk", s_norm, masks.float())
    return q_f


def compute_log_energy_bands(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Alternative descriptor (analysis-only, per task spec): log-energy
    radial bands, NOT used for training unless the primary descriptor fails
    its validity check."""
    b, c, h, w = x.shape
    key = (h, x.device)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _build_band_masks(h, x.device)
    masks = _MASK_CACHE[key]

    y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    spec = torch.fft.fft2(y)
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    s = spec.real ** 2 + spec.imag ** 2
    log_s = torch.log(s + eps)
    band_sum = torch.einsum("bhw,khw->bk", log_s, masks.float())
    band_count = masks.float().sum(dim=(-2, -1)).clamp(min=1)  # (8,)
    return band_sum / band_count[None, :]  # mean log-energy per band
