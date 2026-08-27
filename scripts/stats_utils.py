"""Small numeric helpers shared by the instrumentation and inference driver."""
from __future__ import annotations

import numpy as np
import torch


def tensor_stats(t: torch.Tensor) -> dict:
    """Numerical fingerprint of a tensor: mean/std/min/max/L1/L2/energy.

    Computed over *all* elements (batch=1 during this experiment, so this is
    effectively "over C,H,W"). Energy = sum(x^2); L2 = sqrt(energy).
    """
    x = t.detach()
    if x.is_complex():
        x = x.abs()
    x = x.float()
    l2_sq = torch.sum(x * x).item()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "l1": x.abs().sum().item(),
        "l2": float(np.sqrt(l2_sq)),
        "energy": l2_sq,
        "numel": x.numel(),
    }


def psnr_ssim(restored: torch.Tensor, clean: torch.Tensor) -> tuple[float, float]:
    """PSNR/SSIM between two (1,3,H,W) tensors in [0,1], matching AdaIR's own
    val_utils.compute_psnr_ssim convention (uint8 round-trip, multichannel SSIM).
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    r = restored.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
    c = clean.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
    r_u8 = (r * 255.0).round().astype(np.uint8)
    c_u8 = (c * 255.0).round().astype(np.uint8)
    psnr = peak_signal_noise_ratio(c_u8, r_u8, data_range=255)
    ssim = structural_similarity(c_u8, r_u8, data_range=255, channel_axis=2)
    return float(psnr), float(ssim)
