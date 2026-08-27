"""TEST18 Phase 5: frequency-domain diagnostics -- "draw diagrams in
frequency domain to see actually what happened" at each AFLB position,
for each trained variant, on one representative image per degradation.

For every (variant, AFLB position, image) where the AFLB is active
(mask_mode is not None), saves a composite figure: input feature FFT
magnitude -> the mask (learned box or fixed box) -> low/high split ->
FMoM-modulated output -> final AFLB output FFT magnitude. Also runs the
same diagnostic against the original released adair3d.ckpt as a
reference point (using the unmodified, released FreModule directly).

Usage (devon, adair-distill env):
  python frequency_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

TEST18 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST18 / "scripts"))
from ablatable_model import build_variant, VARIANTS, ADAIR_ROOT  # noqa: E402

sys.path.insert(0, str(ADAIR_ROOT))
from net.model import AdaIR  # noqa: E402 (read-only reuse, released reference)

CKPT_DIR = TEST18 / "results" / "checkpoints"
OUT_DIR = TEST18 / "results" / "frequency_diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RELEASED_CKPT = Path("/home/minura/FYP/Workspace/Himeth/weights/adair3d.ckpt")

REPRESENTATIVE_IMAGES = {
    "Rain": Path("/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L/rainy/rain-050.png"),
    "Haze": Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/input/0001_0.8_0.2.jpg"),
}
# Noise: synthesize on the fly from a real clean SOTS target image (sigma=25)
NOISE_SOURCE = Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/target/0001.png")

AFLB_NAMES = {"fre1": "AFLB1 (latent, deepest)", "fre2": "AFLB2 (decoder_level3)", "fre3": "AFLB3 (decoder_level2)"}


def load_image_tensor(path, device, size=256):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def make_noisy(device, size=256, sigma=25):
    img = Image.open(NOISE_SOURCE).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(arr.shape) * sigma
    noisy = np.clip(arr + noise, 0, 255) / 255.0
    return torch.from_numpy(noisy.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def fft_mag_log(t: torch.Tensor) -> np.ndarray:
    """log-magnitude FFT spectrum, channel-averaged, shifted to center."""
    x = t[0].mean(dim=0)  # average over channels
    f = torch.fft.fftshift(torch.fft.fft2(x))
    mag = torch.log1p(torch.abs(f))
    return mag.cpu().numpy()


def run_variant_diagnostics(variant_name, images, device):
    """Returns dict[(deg, aflb_key)] -> diag dict, or None if AFLB inactive."""
    model = build_variant(variant_name).to(device)
    ckpt_path = CKPT_DIR / f"model_{variant_name}_final.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    results = {}
    with torch.no_grad():
        for deg, x in images.items():
            _, _, H, W = x.shape
            inp_enc1 = model.patch_embed(x)
            out_enc1 = model.encoder_level1(inp_enc1)
            inp_enc2 = model.down1_2(out_enc1)
            out_enc2 = model.encoder_level2(inp_enc2)
            inp_enc3 = model.down2_3(out_enc2)
            out_enc3 = model.encoder_level3(inp_enc3)
            inp_enc4 = model.down3_4(out_enc3)
            latent = model.latent(inp_enc4)

            latent, diag1 = model.fre1(x, latent, return_diagnostics=True)
            results[(deg, "fre1")] = diag1

            inp_dec3 = model.up4_3(latent)
            inp_dec3 = torch.cat([inp_dec3, out_enc3], 1)
            inp_dec3 = model.reduce_chan_level3(inp_dec3)
            out_dec3 = model.decoder_level3(inp_dec3)
            out_dec3, diag2 = model.fre2(x, out_dec3, return_diagnostics=True)
            results[(deg, "fre2")] = diag2

            inp_dec2 = model.up3_2(out_dec3)
            inp_dec2 = torch.cat([inp_dec2, out_enc2], 1)
            inp_dec2 = model.reduce_chan_level2(inp_dec2)
            out_dec2 = model.decoder_level2(inp_dec2)
            out_dec2, diag3 = model.fre3(x, out_dec2, return_diagnostics=True)
            results[(deg, "fre3")] = diag3
    del model
    torch.cuda.empty_cache()
    return results


def plot_composite(deg, aflb_key, variant_name, diag, out_path):
    if not diag.get("active"):
        return False
    mask = diag["mask"][0].mean(dim=0).cpu().numpy()
    high_mag = fft_mag_log(diag["high_feature"])
    low_mag = fft_mag_log(diag["low_feature"])
    agg_mag = fft_mag_log(diag["agg"])
    out_mag = fft_mag_log(diag["output"])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    panels = [("Mask (low-freq region)", mask, "gray"),
              ("High-freq mined (post-FMiM)", high_mag, "inferno"),
              ("Low-freq mined (post-FMiM)", low_mag, "inferno"),
              ("FMoM-aggregated", agg_mag, "inferno"),
              ("AFLB output spectrum", out_mag, "inferno")]
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{variant_name} -- {AFLB_NAMES.get(aflb_key, aflb_key)} -- {deg}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def released_model_diagnostics(images, device):
    """Reference: the actual released architecture, loaded from adair3d.ckpt."""
    model = AdaIR(decoder=True).to(device)
    state = torch.load(RELEASED_CKPT, map_location=device, weights_only=False)
    sd = state.get("params", state.get("state_dict", state))
    sd = {k.replace("module.", "").replace("net.", "", 1) if k.startswith("net.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"released ckpt load: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model.eval()

    results = {}
    with torch.no_grad():
        for deg, x in images.items():
            inp_enc1 = model.patch_embed(x)
            out_enc1 = model.encoder_level1(inp_enc1)
            inp_enc2 = model.down1_2(out_enc1)
            out_enc2 = model.encoder_level2(inp_enc2)
            inp_enc3 = model.down2_3(out_enc2)
            out_enc3 = model.encoder_level3(inp_enc3)
            inp_enc4 = model.down3_4(out_enc3)
            latent = model.latent(inp_enc4)
            latent_out = model.fre1(x, latent)
            results[(deg, "fre1")] = {"active": True, "output": latent_out.detach()}
    return results, model


def main():
    device = "cuda"
    images = {
        "Rain": load_image_tensor(REPRESENTATIVE_IMAGES["Rain"], device),
        "Haze": load_image_tensor(REPRESENTATIVE_IMAGES["Haze"], device),
        "Noise": make_noisy(device),
    }

    n_saved = 0
    for variant in VARIANTS:
        print(f"=== {variant} ===", flush=True)
        results = run_variant_diagnostics(variant, images, device)
        for (deg, aflb_key), diag in results.items():
            out_path = OUT_DIR / f"{variant}_{aflb_key}_{deg}.png"
            if plot_composite(deg, aflb_key, variant, diag, out_path):
                n_saved += 1
                print(f"  saved {out_path.name}", flush=True)
            else:
                print(f"  {variant} {aflb_key} {deg}: AFLB inactive (A_baseline), skipped", flush=True)

    print(f"\nwrote {n_saved} frequency-diagram composites to {OUT_DIR}")


if __name__ == "__main__":
    main()
