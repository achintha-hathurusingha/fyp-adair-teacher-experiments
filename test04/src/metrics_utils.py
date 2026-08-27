"""TEST04 shared metric helpers."""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_np_u8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)


def psnr_ssim_mse(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: (1,3,H,W) tensors in [0,1]. Returns PSNR/SSIM/MSE(a vs b)."""
    a_u8, b_u8 = to_np_u8(a), to_np_u8(b)
    psnr = float(peak_signal_noise_ratio(b_u8, a_u8, data_range=255))
    ssim = float(structural_similarity(b_u8, a_u8, data_range=255, channel_axis=2))
    mse = float(((a.detach().float() - b.detach().float()) ** 2).mean().item())
    return {"psnr": psnr, "ssim": ssim, "mse": mse}


def output_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """L2 / MAE distance between two (1,3,H,W) output tensors."""
    diff = (a.detach().float() - b.detach().float())
    return {
        "l2": float(torch.linalg.vector_norm(diff.reshape(-1)).item()),
        "mae": float(diff.abs().mean().item()),
    }


def pooled_vec(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    gap = x.mean(dim=(2, 3))[0]
    gmp = x.amax(dim=(2, 3))[0]
    return torch.cat([gap, gmp]).cpu().numpy()


def residual_stats(clean: torch.Tensor, output: torch.Tensor) -> dict:
    r = (clean.detach().float() - output.detach().float())
    l2 = torch.linalg.vector_norm(r.reshape(-1)).item()
    return {"residual_mean": r.mean().item(), "residual_std": r.std(unbiased=False).item(),
            "residual_energy": l2 ** 2, "residual_mae": r.abs().mean().item()}
