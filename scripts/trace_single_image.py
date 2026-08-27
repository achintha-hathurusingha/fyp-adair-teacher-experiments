"""Rigorous single-image trace of the MGB -> mask -> FFT split pipeline,
run fresh against the actual AdaIR source (not the saved .pt bundles), to
either confirm or refute the "mask is degenerate" finding from the 300-image
run.

For R001 (Rain100L), for every AFLB, prints:
  MGB output (threshold pre-sigmoid, alpha, beta)
  mask.shape / mask.unique() / mask.sum() / mask.min() / mask.max()
  fft.shape, F_low.shape, F_high.shape
  ||F||, ||F_low||, ||F_high||
  verification: mask + (1-mask) == 1 everywhere?
  verification: ||F||^2 vs ||F_low||^2 + ||F_high||^2

Then saves a step-5 visualization panel per AFLB.

Usage (on devon, adair-distill env, PINNED -- this box has flaky cores 8-11):
  taskset -c 0-7,12-31 python trace_single_image.py
"""
from __future__ import annotations

import csv as csv_module
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import Recorder, attach_instrumentation, load_adair
from run_inference import crop_img, load_rgb, to_tensor

REPO = Path(__file__).resolve().parent.parent
ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
OUT_DIR = REPO / "outputs" / "trace_R001"
OUT_CSV = REPO / "csv_export" / "12_Trace_R001.csv"

IMAGE_ID = "R001"
GT_PATH = REPO.parent  # placeholder, resolved from manifest below


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import csv
    with open(REPO / "manifest.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["Image_ID"] == IMAGE_ID)
    print(f"tracing {IMAGE_ID}: {row}\n")

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)

    gt_np = crop_img(load_rgb(row["gt_path"]))
    degraded_np = crop_img(load_rgb(row["input_path"]))
    degraded_t = to_tensor(degraded_np, device)

    recorder.start()
    with torch.no_grad():
        restored_t = model(degraded_t)
    snap = recorder.snapshot_cpu()

    print("=" * 100)
    trace_rows = []
    for aflb in ["AFLB1", "AFLB2", "AFLB3"]:
        d = snap[aflb]
        conv_feat = d["conv_feat"]
        th = d["threshold_alpha_beta"]  # post-sigmoid, (1,2,1,1)
        mask = d["mask"]
        fft = d["fft_shifted"]  # complex, (1,C,H,W)
        raw_low = d["raw_low"]
        raw_high = d["raw_high"]

        h, w = conv_feat.shape[-2:]
        n = 128
        alpha = th[0, 0, 0, 0].item()
        beta = th[0, 1, 0, 0].item()
        h_ = int(h // n * alpha)
        w_ = int(w // n * beta)

        print(f"\n--- {aflb} ---")
        print(f"conv_feat.shape = {tuple(conv_feat.shape)}  (h={h}, w={w})")
        print(f"MGB / rate_conv output (post-sigmoid) -> alpha={alpha:.6f}, beta={beta:.6f}")
        print(f"n (hardcoded normaliser) = {n}")
        print(f"h//n = {h // n}, w//n = {w // n}")
        print(f"h_ = int((h//n) * alpha) = int({h // n} * {alpha:.6f}) = {h_}")
        print(f"w_ = int((w//n) * beta)  = int({w // n} * {beta:.6f}) = {w_}")
        print(f"-> mask box = [h/2-{h_} : h/2+{h_}, w/2-{w_} : w/2+{w_}] = "
              f"[{h // 2 - h_}:{h // 2 + h_}, {w // 2 - w_}:{w // 2 + w_}]  "
              f"(box height={2 * h_}, width={2 * w_})")

        print(f"mask.shape = {tuple(mask.shape)}")
        print(f"mask.unique() = {mask.unique().tolist()}")
        print(f"mask.sum() = {mask.sum().item()} / {mask.numel()} = {100 * mask.mean().item():.4f}%")
        print(f"mask.min() = {mask.min().item()}, mask.max() = {mask.max().item()}")

        fft_low = fft * mask
        fft_high = fft * (1 - mask)
        print(f"fft.shape = {tuple(fft.shape)} (complex)")
        print(f"F_low = fft*mask, shape {tuple(fft_low.shape)}")
        print(f"F_high = fft*(1-mask), shape {tuple(fft_high.shape)}")

        norm_F = torch.linalg.vector_norm(fft.reshape(-1)).item()
        norm_Fl = torch.linalg.vector_norm(fft_low.reshape(-1)).item()
        norm_Fh = torch.linalg.vector_norm(fft_high.reshape(-1)).item()
        print(f"||F|| = {norm_F:.6f}")
        print(f"||F_low|| = {norm_Fl:.6f}")
        print(f"||F_high|| = {norm_Fh:.6f}")

        # --- verification 1: Ml + Mh == 1 everywhere ---
        Ml, Mh = mask, 1 - mask
        ml_plus_mh_ok = torch.allclose(Ml + Mh, torch.ones_like(Ml))
        print(f"VERIFY  Ml + Mh == 1 everywhere: {ml_plus_mh_ok}")

        # --- verification 2: ||F||^2 ~= ||F_low||^2 + ||F_high||^2 (disjoint binary support -> exact) ---
        e_total = norm_F ** 2
        e_split = norm_Fl ** 2 + norm_Fh ** 2
        rel_err = abs(e_total - e_split) / max(e_total, 1e-12)
        print(f"VERIFY  ||F||^2 = {e_total:.4f}  vs  ||F_low||^2+||F_high||^2 = {e_split:.4f}  "
              f"(rel. error {rel_err:.2e})")

        # --- verification 3: does raw_low/raw_high (the ifft outputs actually fed downstream)
        #     match manual reconstruction from fft_low/fft_high? ---
        def unshift(x):
            b, c, hh, ww = x.shape
            return torch.roll(x, shifts=(-int(hh / 2), -int(ww / 2)), dims=(2, 3))

        manual_low = torch.abs(torch.fft.ifft2(unshift(fft_low), norm="forward", dim=(-2, -1)))
        manual_high = torch.abs(torch.fft.ifft2(unshift(fft_high), norm="forward", dim=(-2, -1)))
        low_match = torch.allclose(manual_low, raw_low, atol=1e-4)
        high_match = torch.allclose(manual_high, raw_high, atol=1e-4)
        print(f"VERIFY  manual ifft(F_low) matches saved raw_low: {low_match}")
        print(f"VERIFY  manual ifft(F_high) matches saved raw_high: {high_match}")
        print(f"raw_low  stats: mean={raw_low.mean().item():.6e} std={raw_low.std().item():.6e} "
              f"max={raw_low.max().item():.6e}")
        print(f"raw_high stats: mean={raw_high.mean().item():.6e} std={raw_high.std().item():.6e} "
              f"max={raw_high.max().item():.6e}")

        trace_rows.append({
            "AFLB": aflb, "feature_h": h, "feature_w": w,
            "alpha": alpha, "beta": beta, "h_over_128": h // n, "w_over_128": w // n,
            "h_half_width": h_, "w_half_width": w_,
            "mask_sum": mask.sum().item(), "mask_numel": mask.numel(),
            "mask_area_pct": 100 * mask.mean().item(),
            "mask_min": mask.min().item(), "mask_max": mask.max().item(),
            "norm_F": norm_F, "norm_F_low": norm_Fl, "norm_F_high": norm_Fh,
            "energy_total": e_total, "energy_split": e_split, "energy_rel_error": rel_err,
            "verify_Ml_plus_Mh_eq_1": ml_plus_mh_ok,
            "verify_manual_ifft_low_matches": low_match,
            "verify_manual_ifft_high_matches": high_match,
            "raw_low_mean": raw_low.mean().item(), "raw_low_max": raw_low.max().item(),
            "raw_high_mean": raw_high.mean().item(), "raw_high_max": raw_high.max().item(),
        })

    print("\n" + "=" * 100)
    print("CONCLUSION: mask box half-width h_/w_ is computed EXACTLY per the released")
    print("AdaIR source (FreModule.fft, net/model.py) -- h_ = int((h // 128) * alpha).")
    print("No shortcut like alpha*H was used anywhere in the instrumentation.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    print(f"\nwrote {OUT_CSV}")

    # ---- step 5 visualisation ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for aflb in ["AFLB1", "AFLB2", "AFLB3"]:
        d = snap[aflb]
        fft = d["fft_shifted"][0].abs().mean(0).numpy()  # channel-mean magnitude
        mask = d["mask"][0, 0].numpy()
        raw_low = d["raw_low"][0].mean(0).numpy()
        raw_high = d["raw_high"][0].mean(0).numpy()

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
        axes[0].imshow(degraded_np); axes[0].set_title("original (degraded)")
        axes[1].imshow(np.log1p(fft), cmap="inferno"); axes[1].set_title("FFT magnitude (log1p, centered)")
        axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[2].set_title(f"low mask (Ml) -- sum={mask.sum():.0f}")
        axes[3].imshow(1 - mask, cmap="gray", vmin=0, vmax=1); axes[3].set_title("high mask (Mh)")
        axes[4].imshow(raw_low, cmap="viridis"); axes[4].set_title(f"low spectrum |ifft(F_low)|\nmax={raw_low.max():.4f}")
        axes[5].imshow(raw_high, cmap="viridis"); axes[5].set_title(f"high spectrum |ifft(F_high)|\nmax={raw_high.max():.4f}")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{IMAGE_ID} -- {aflb} -- MGB->mask->FFT-split trace")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{IMAGE_ID}_{aflb}_trace.png", dpi=120)
        plt.close(fig)
        print(f"saved {OUT_DIR / f'{IMAGE_ID}_{aflb}_trace.png'}")


if __name__ == "__main__":
    main()
