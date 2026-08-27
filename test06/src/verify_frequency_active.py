"""TEST06 Phase 6: verify the frequency path is genuinely non-degenerate at
AFLB3 for the 06-E dataset, BEFORE running any causal swap. Checks, for at
least 10 scenes x 3 degradations: raw_low != 0, raw_high != 0, mask is
non-degenerate (mask_active_fraction > 0). Saves FFT magnitude / low mask /
high mask / raw_low / raw_high visualizations for a handful of examples.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python verify_frequency_active.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

TEST06 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST06.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
from model_variants import load_variant  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST06 / "results" / "frequency_intervention" / "scene_manifest.csv"
OUT_DIR = TEST06 / "results" / "frequency_intervention" / "frequency_activation_examples"
DEGS = ["Rain", "Haze", "Noise"]
N_VERIFY = 10
N_VISUALIZE = 3


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main():
    device = "cuda"
    model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, "released")
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        scenes = list(csv.DictReader(f))[:N_VERIFY]

    rows = []
    for i, scene in enumerate(scenes):
        for deg in DEGS:
            img = np.array(Image.open(scene[f"{deg.lower()}_path"]).convert("RGB"))
            img_t = to_tensor(img, device)
            recorder.start()
            with torch.no_grad():
                _ = model(img_t)
            snap = recorder.snapshot_cpu()
            d = snap["AFLB3"]
            raw_low, raw_high, mask = d["raw_low"], d["raw_high"], d["mask"]
            low_nonzero = bool((raw_low.abs() > 1e-8).any().item())
            high_nonzero = bool((raw_high.abs() > 1e-8).any().item())
            mask_active = float((mask > 0.5).float().mean().item())
            rows.append({"scene_id": scene["scene_id"], "degradation": deg,
                         "raw_low_nonzero": low_nonzero, "raw_high_nonzero": high_nonzero,
                         "mask_active_fraction": mask_active,
                         "raw_low_max": float(raw_low.abs().max().item()),
                         "raw_high_max": float(raw_high.abs().max().item())})

            if i < N_VISUALIZE:
                fft_mag = torch.log1p(d["fft_shifted"].abs()[0].mean(0)).numpy()
                mask_img = mask[0].mean(0).numpy()
                low_img = raw_low[0].mean(0).numpy()
                high_img = raw_high[0].mean(0).numpy()
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, arr, title in zip(
                    axes, [fft_mag, mask_img, low_img, high_img],
                    ["log FFT magnitude", "mask", "raw_low", "raw_high"],
                ):
                    im = ax.imshow(arr, cmap="viridis")
                    ax.set_title(title)
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                fig.suptitle(f"{scene['scene_id']} / {deg} / AFLB3 (mask_active_fraction={mask_active:.6f})")
                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{scene['scene_id']}_{deg}_AFLB3.png", bbox_inches="tight")
                plt.close(fig)

            print(f"{scene['scene_id']} / {deg}: raw_low_nonzero={low_nonzero} raw_high_nonzero={high_nonzero} "
                  f"mask_active={mask_active:.6f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(TEST06 / "results" / "frequency_intervention" / "frequency_activation_verification.csv", index=False)

    all_active = bool((df.mask_active_fraction > 0).all() and df.raw_low_nonzero.all() and df.raw_high_nonzero.all())
    print(f"\nALL {len(df)} (scene, degradation) pairs show genuine non-degenerate frequency path: {all_active}")
    if not all_active:
        print("WARNING: at least one scene/degradation shows a degenerate frequency path at this resolution. "
              "Per Phase 6, DO NOT proceed with the causal swap until this is resolved.")
    else:
        print("Phase 6 PASSED. Proceeding to Phase 7-11 (06-E causal experiment) is justified.")


if __name__ == "__main__":
    main()
