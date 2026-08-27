"""Phase 10: visualizations.

Per-degradation panels (rain/haze/noise, one representative image each,
R001/H001/N001): original, FFT magnitude, released low/high masks, modified
low/high masks, released/modified low spectra, H-L attention, L-H attention,
restored (released) output, ground truth.

Plus distribution plots: mask-area, PSNR, PSNR-difference (no_frequency -
released), frequency-energy, alpha/beta -- all split by degradation.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python build_visualizations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TEST01 = Path(__file__).resolve().parent.parent
REPO = TEST01.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(TEST01 / "scripts"))
from model_variants import load_variant  # noqa: E402
from run_inference import crop_img, load_rgb  # noqa: E402
import csv as csv_module  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = REPO / "manifest.csv"
CSV_DIR = TEST01 / "csv_export"
OUT_DIR = TEST01 / "outputs" / "visuals"
DEG_COLORS = {"Rain": "#3b7dd8", "Haze": "#d8853b", "Noise": "#3bb273"}
DEG_ORDER = ["Rain", "Haze", "Noise"]
REPRESENTATIVE = {"Rain": "R001", "Haze": "H001", "Noise": "N001"}


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def panel_for_image(image_id: str, deg: str, row: dict, device="cuda"):
    gt_np = crop_img(load_rgb(row["gt_path"]))
    if deg == "Noise":
        rng = np.random.RandomState(abs(hash(image_id)) % (2 ** 31))
        noise = rng.randn(*gt_np.shape) * float(row["noise_sigma"])
        degraded_np = np.clip(gt_np + noise, 0, 255).astype(np.uint8)
    else:
        degraded_np = crop_img(load_rgb(row["input_path"]))
    degraded_t = to_tensor(degraded_np, device)

    snaps = {}
    restored_np = None
    for variant in ["released", "modified_mask"]:
        model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, variant)
        recorder.start()
        with torch.no_grad():
            restored_t = model(degraded_t)
        snaps[variant] = recorder.snapshot_cpu()
        if variant == "released":
            restored_np = (restored_t.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        del model
        torch.cuda.empty_cache()

    d_rel = snaps["released"]["AFLB1"]
    d_mod = snaps["modified_mask"]["AFLB1"]
    fft_mag = d_rel["fft_shifted"][0].abs().mean(0).numpy()
    mask_rel = d_rel["mask"][0, 0].numpy()
    mask_mod = d_mod["mask"][0, 0].numpy()
    low_rel = d_rel["raw_low"][0].mean(0).numpy()
    low_mod = d_mod["raw_low"][0].mean(0).numpy()
    hl = d_rel["hl_spatial_weight"][0, 0].numpy()
    lh_mean = d_rel["lh_channel_weight"][0].mean().item()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes[0, 0].imshow(degraded_np); axes[0, 0].set_title(f"{image_id} ({deg}) -- degraded input")
    axes[0, 1].imshow(np.log1p(fft_mag), cmap="inferno"); axes[0, 1].set_title("FFT magnitude (log1p)")
    axes[0, 2].imshow(restored_np); axes[0, 2].set_title("restored (released)")
    axes[0, 3].imshow(gt_np); axes[0, 3].set_title("ground truth")

    axes[1, 0].imshow(mask_rel, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title(f"RELEASED low mask (sum={mask_rel.sum():.0f})")
    axes[1, 1].imshow(1 - mask_rel, cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title("RELEASED high mask")
    axes[1, 2].imshow(mask_mod, cmap="gray", vmin=0, vmax=1); axes[1, 2].set_title(f"MODIFIED low mask (sum={mask_mod.sum():.0f})")
    axes[1, 3].imshow(1 - mask_mod, cmap="gray", vmin=0, vmax=1); axes[1, 3].set_title("MODIFIED high mask")

    axes[2, 0].imshow(low_rel, cmap="viridis"); axes[2, 0].set_title(f"RELEASED low spectrum\nmax={low_rel.max():.2e}")
    axes[2, 1].imshow(low_mod, cmap="viridis"); axes[2, 1].set_title(f"MODIFIED low spectrum\nmax={low_mod.max():.2e}")
    axes[2, 2].imshow(hl, cmap="magma"); axes[2, 2].set_title("H-L spatial attention (AFLB1)")
    axes[2, 3].text(0.1, 0.5, f"L-H channel attention\n(AFLB1) mean = {lh_mean:.4f}", fontsize=12)
    axes[2, 3].axis("off")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{deg} -- {image_id} -- mechanism panel (AFLB1, deepest)")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{deg.lower()}_{image_id}_panel.png", dpi=110)
    plt.close(fig)
    print(f"saved {deg.lower()}_{image_id}_panel.png")


def distribution_plots():
    per_image = pd.read_csv(CSV_DIR / "20_Per_Image_All_Variants.csv")
    diff_nofreq = pd.read_csv(CSV_DIR / "27_Released_vs_NoFrequency.csv")
    mgb = pd.read_csv(CSV_DIR / "21_Ablation_MGB_Values.csv")
    freq = pd.read_csv(CSV_DIR / "22_Ablation_Frequency_Statistics.csv")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # A. mask-area distribution (representative images, released)
    sub = mgb[mgb.model == "released"]
    data = [sub[sub.Degradation == d]["mask_area_pct"] for d in DEG_ORDER]
    axes[0, 0].boxplot(data, tick_labels=DEG_ORDER)
    axes[0, 0].set_title("A. mask_area_pct (released, representative imgs)")

    # B. PSNR distribution (all 300, released)
    rel = per_image[per_image.model == "released"]
    data = [rel[rel.Degradation == d]["psnr"] for d in DEG_ORDER]
    bp = axes[0, 1].boxplot(data, tick_labels=DEG_ORDER, patch_artist=True)
    for patch, d in zip(bp["boxes"], DEG_ORDER):
        patch.set_facecolor(DEG_COLORS[d]); patch.set_alpha(0.6)
    axes[0, 1].set_title("B. PSNR distribution (released, n=300)")

    # C. PSNR difference distribution (no_frequency - released)
    data = [diff_nofreq[diff_nofreq.Degradation == d]["psnr_diff"] for d in DEG_ORDER]
    axes[0, 2].boxplot(data, tick_labels=DEG_ORDER)
    axes[0, 2].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0, 2].set_title("C. PSNR diff: no_frequency - released (n=300)")

    # D. frequency-energy distribution (low_pct, representative imgs, released)
    fsub = freq[freq.model == "released"]
    data = [fsub[fsub.Degradation == d]["low_pct"] for d in DEG_ORDER]
    axes[1, 0].boxplot(data, tick_labels=DEG_ORDER)
    axes[1, 0].set_title("D. low-frequency energy % (released, representative imgs)")

    # E. alpha/beta distribution
    data_a = [sub[sub.Degradation == d]["alpha"] for d in DEG_ORDER]
    data_b = [sub[sub.Degradation == d]["beta"] for d in DEG_ORDER]
    positions_a = np.arange(len(DEG_ORDER)) * 2.0
    positions_b = positions_a + 0.6
    axes[1, 1].boxplot(data_a, positions=positions_a, widths=0.5, tick_labels=DEG_ORDER)
    axes[1, 1].boxplot(data_b, positions=positions_b, widths=0.5, tick_labels=["" for _ in DEG_ORDER])
    axes[1, 1].set_title("E. alpha (left) / beta (right) per degradation")

    # F. SSIM difference distribution (no_frequency - released) as a stand-in
    # feature-space visualization (PCA/t-SNE already covered in the original
    # 300-image notebook; not duplicated here)
    data = [diff_nofreq[diff_nofreq.Degradation == d]["ssim_diff"] for d in DEG_ORDER]
    axes[1, 2].boxplot(data, tick_labels=DEG_ORDER)
    axes[1, 2].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1, 2].set_title("F. SSIM diff: no_frequency - released (n=300)")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "distributions_A_F.png", dpi=120)
    plt.close(fig)
    print("saved distributions_A_F.png")


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv_module.DictReader(f))
    for deg, image_id in REPRESENTATIVE.items():
        row = next(r for r in rows if r["Image_ID"] == image_id)
        panel_for_image(image_id, deg, row)

    distribution_plots()


if __name__ == "__main__":
    main()
