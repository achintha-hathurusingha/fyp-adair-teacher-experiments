"""Full pipeline walkthrough of the ORIGINAL released AdaIR checkpoint
(adair3d.ckpt), for one representative image per degradation type
(Rain/Haze/Noise). Captures, at every stage:

  - the actual feature map (channel-mean heatmap) at patch_embed, each
    encoder level, latent, each decoder level, and the final output image
  - full frequency diagnostics (mask, mined high/low, aggregated,
    AFLB output spectrum) at all 3 AFLBs

Diagnostics are captured via non-invasive `register_forward_hook` calls
on the REAL FreModule's own submodules (rate_conv, channel_cross_l,
channel_cross_h, frequency_refine) during a completely normal,
unmodified forward pass -- NOT by reimplementing any of AdaIR's logic,
so there is zero risk of diverging from the released model's actual
behavior. The box mask itself is reconstructed post-hoc from the
captured threshold tensor using the exact formula independently audited
from source in TEST01 (`h_ = int((h // n) * threshold)`, n=128).

This is documentation/visualization only -- the released checkpoint and
AdaIR source are read-only, never modified.

Usage (devon, adair-distill env):
  python original_checkpoint_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

TEST18 = Path(__file__).resolve().parent.parent
ADAIR_ROOT = Path("/home/minura/teacher-experiments/AdaIR")
sys.path.insert(0, str(ADAIR_ROOT))
from net.model import AdaIR  # noqa: E402 (read-only reuse)

RELEASED_CKPT = Path("/home/minura/FYP/Workspace/Himeth/weights/adair3d.ckpt")
OUT_DIR = TEST18 / "results" / "original_checkpoint_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = {
    "Rain": Path("/home/minura/FYP/Workspace/Himeth/data/rain100L/rain100L_test/Rain100L/rainy/rain-050.png"),
    "Haze": Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/input/0001_0.8_0.2.jpg"),
}
NOISE_SOURCE = Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/target/0001.png")


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


def save_rgb(t: torch.Tensor, path: Path):
    arr = t[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


def save_feature_heatmap(t: torch.Tensor, path: Path, title: str):
    fmap = t[0].mean(dim=0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(fmap, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fft_mag_log(t: torch.Tensor) -> np.ndarray:
    x = t[0].mean(dim=0)
    f = torch.fft.fftshift(torch.fft.fft2(x))
    return torch.log1p(torch.abs(f)).detach().cpu().numpy()


def attach_diagnostic_hooks(fre_module):
    """Non-invasive forward hooks -- observes real intermediate
    activations, reimplements nothing. See module docstring."""
    store = {}

    def save(key):
        def hook(module, inp, out):
            store[key] = out.detach() if torch.is_tensor(out) else out
        return hook

    fre_module.rate_conv.register_forward_hook(save("threshold"))
    fre_module.channel_cross_l.register_forward_hook(save("high_feature"))
    fre_module.channel_cross_h.register_forward_hook(save("low_feature"))
    fre_module.frequency_refine.register_forward_hook(save("agg"))
    fre_module.register_forward_hook(save("output"))
    return store


def reconstruct_mask(threshold: torch.Tensor, h: int, w: int, n: int = 128) -> torch.Tensor:
    """Rebuilds the box mask from the captured threshold, using the exact
    formula independently audited from source in TEST01: h_ = int((h//n)
    * threshold), box centered. n=128 matches fft()'s own default arg."""
    b = threshold.shape[0]
    mask = torch.zeros(b, 1, h, w, device=threshold.device)
    for i in range(b):
        h_ = int((h // n) * float(threshold[i, 0]))
        w_ = int((w // n) * float(threshold[i, 1]))
        mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
    return mask


def save_aflb_panel(store, h, w, deg, aflb_name, path):
    mask = reconstruct_mask(store["threshold"], h, w)
    mask_np = mask[0, 0].detach().cpu().numpy()
    high_mag = fft_mag_log(store["high_feature"])
    low_mag = fft_mag_log(store["low_feature"])
    agg_mag = fft_mag_log(store["agg"])
    out_mag = fft_mag_log(store["output"])
    active_frac = float(mask_np.mean())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    panels = [(f"Mask (active frac={active_frac:.5f})", mask_np, "gray"),
              ("High-freq mined (post cross-attn)", high_mag, "inferno"),
              ("Low-freq mined (post cross-attn)", low_mag, "inferno"),
              ("frequency_refine aggregate", agg_mag, "inferno"),
              ("AFLB output spectrum", out_mag, "inferno")]
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"RELEASED adair3d.ckpt -- {aflb_name} -- {deg}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return active_frac


def main():
    device = "cuda"
    model = AdaIR(decoder=True).to(device)
    state = torch.load(RELEASED_CKPT, map_location=device, weights_only=False)
    sd = state.get("params", state.get("state_dict", state))
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"released ckpt load: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model.eval()

    stores = {name: attach_diagnostic_hooks(getattr(model, name)) for name in ("fre1", "fre2", "fre3")}

    images = {
        "Rain": load_image_tensor(IMAGES["Rain"], device),
        "Haze": load_image_tensor(IMAGES["Haze"], device),
        "Noise": make_noisy(device),
    }

    manifest = {}
    with torch.no_grad():
        for deg, x in images.items():
            print(f"=== {deg} ===", flush=True)
            save_rgb(x, OUT_DIR / f"{deg}_00_input.png")

            inp_enc1 = model.patch_embed(x)
            save_feature_heatmap(inp_enc1, OUT_DIR / f"{deg}_01_patch_embed.png", f"{deg}: patch_embed (shallow features Y0)")

            out_enc1 = model.encoder_level1(inp_enc1)
            save_feature_heatmap(out_enc1, OUT_DIR / f"{deg}_02_encoder_level1.png", f"{deg}: encoder_level1")

            inp_enc2 = model.down1_2(out_enc1)
            out_enc2 = model.encoder_level2(inp_enc2)
            save_feature_heatmap(out_enc2, OUT_DIR / f"{deg}_03_encoder_level2.png", f"{deg}: encoder_level2")

            inp_enc3 = model.down2_3(out_enc2)
            out_enc3 = model.encoder_level3(inp_enc3)
            save_feature_heatmap(out_enc3, OUT_DIR / f"{deg}_04_encoder_level3.png", f"{deg}: encoder_level3")

            inp_enc4 = model.down3_4(out_enc3)
            latent = model.latent(inp_enc4)
            save_feature_heatmap(latent, OUT_DIR / f"{deg}_05_latent_pre_AFLB1.png", f"{deg}: latent (pre-AFLB1)")

            h1, w1 = latent.shape[-2:]
            latent = model.fre1(x, latent)
            af1 = save_aflb_panel(stores["fre1"], h1, w1, deg, "AFLB1 (latent, deepest)",
                                   OUT_DIR / f"{deg}_06_AFLB1_diagnostics.png")
            save_feature_heatmap(latent, OUT_DIR / f"{deg}_07_latent_post_AFLB1.png", f"{deg}: latent (post-AFLB1)")

            inp_dec3 = model.up4_3(latent)
            inp_dec3 = torch.cat([inp_dec3, out_enc3], 1)
            inp_dec3 = model.reduce_chan_level3(inp_dec3)
            out_dec3 = model.decoder_level3(inp_dec3)
            save_feature_heatmap(out_dec3, OUT_DIR / f"{deg}_08_decoder_level3_pre_AFLB2.png", f"{deg}: decoder_level3 (pre-AFLB2)")

            h2, w2 = out_dec3.shape[-2:]
            out_dec3 = model.fre2(x, out_dec3)
            af2 = save_aflb_panel(stores["fre2"], h2, w2, deg, "AFLB2 (decoder_level3)",
                                   OUT_DIR / f"{deg}_09_AFLB2_diagnostics.png")

            inp_dec2 = model.up3_2(out_dec3)
            inp_dec2 = torch.cat([inp_dec2, out_enc2], 1)
            inp_dec2 = model.reduce_chan_level2(inp_dec2)
            out_dec2 = model.decoder_level2(inp_dec2)
            save_feature_heatmap(out_dec2, OUT_DIR / f"{deg}_10_decoder_level2_pre_AFLB3.png", f"{deg}: decoder_level2 (pre-AFLB3)")

            h3, w3 = out_dec2.shape[-2:]
            out_dec2 = model.fre3(x, out_dec2)
            af3 = save_aflb_panel(stores["fre3"], h3, w3, deg, "AFLB3 (decoder_level2)",
                                   OUT_DIR / f"{deg}_11_AFLB3_diagnostics.png")

            inp_dec1 = model.up2_1(out_dec2)
            inp_dec1 = torch.cat([inp_dec1, out_enc1], 1)
            out_dec1 = model.decoder_level1(inp_dec1)
            save_feature_heatmap(out_dec1, OUT_DIR / f"{deg}_12_decoder_level1.png", f"{deg}: decoder_level1")

            out_dec1 = model.refinement(out_dec1)
            restored = model.output(out_dec1) + x
            save_rgb(restored, OUT_DIR / f"{deg}_13_output.png")

            manifest[deg] = {
                "AFLB1_active_fraction": af1, "AFLB2_active_fraction": af2, "AFLB3_active_fraction": af3,
                "AFLB1_feature_hw": [h1, w1], "AFLB2_feature_hw": [h2, w2], "AFLB3_feature_hw": [h3, w3],
                "input_shape": list(x.shape), "output_shape": list(restored.shape),
            }
            print(f"  feature HxW: AFLB1={h1}x{w1} AFLB2={h2}x{w2} AFLB3={h3}x{w3}", flush=True)
            print(f"  AFLB active fractions: {af1:.6f} / {af2:.6f} / {af3:.6f}", flush=True)

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote pipeline stage images + diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
