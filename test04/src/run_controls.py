"""TEST04 Phase 8 (skip-connection progressive intervention) + Phase 9
Controls B-E (cross-scene, random, zero/mean). Runs on a subset of scenes
(recomputes normal passes for that subset independently -- cheap, ~20
scenes x 3 degradations = 60 extra forward passes -- rather than sharing
in-memory cache across separate script invocations).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python run_controls.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST04 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST04.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEST04 / "src"))
from instrument import load_adair  # noqa: E402
from intervention import manual_forward, sanity_check  # noqa: E402
from metrics_utils import psnr_ssim_mse, output_diff, residual_stats  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST04 / "results" / "manifest" / "scene_manifest.csv"
CONTROLS_DIR = TEST04 / "results" / "controls"
INTERVENTIONS_DIR = TEST04 / "results" / "interventions"

DEGS = ["Rain", "Haze", "Noise"]
N_SKIP_SCENES = 20
N_CROSS_SCENE_SCENES = 20
N_RANDOM_ZERO_SCENES = 20


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def run_normal(model, scene_row, device):
    clean_t = to_tensor(load_rgb(scene_row["clean_image_path"]), device)
    cache = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), device)
        out = manual_forward(model, img_t)
        out["input"] = img_t
        cache[deg] = out
    return clean_t, cache


def metrics_row(swapped_out, clean_t, normal_recipient_out):
    vs_clean = psnr_ssim_mse(swapped_out, clean_t)
    vs_normal = output_diff(swapped_out, normal_recipient_out)
    res = residual_stats(clean_t, swapped_out)
    san = sanity_check(swapped_out)
    return {"psnr_vs_clean": vs_clean["psnr"], "ssim_vs_clean": vs_clean["ssim"], "mse_vs_clean": vs_clean["mse"],
            "l2_vs_normal_recipient": vs_normal["l2"], "mae_vs_normal_recipient": vs_normal["mae"],
            **res, **{f"sanity_{k}": v for k, v in san.items()}}


def main():
    device = "cuda"
    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    print("checkpoint loaded", flush=True)

    # ---- Phase 8: skip-connection progressive intervention ----
    skip_rows = []
    subset = scene_rows[:N_SKIP_SCENES]
    for i, scene_row in enumerate(subset):
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)
        for recipient in DEGS:
            for donor in DEGS:
                if donor == recipient:
                    continue
                recipient_img = cache[recipient]["input"]
                donor = donor
                d = cache[donor]

                conditions = {
                    "A_latent_only": {"latent_pre": d["latent_pre"]},
                    "B_latent_plus_deepest_skip": {"latent_pre": d["latent_pre"], "enc3": d["enc3"]},
                    "C_latent_plus_all_skips": {"latent_pre": d["latent_pre"], "enc1": d["enc1"],
                                                  "enc2": d["enc2"], "enc3": d["enc3"]},
                }
                for cond_name, overrides in conditions.items():
                    swapped_out = manual_forward(model, recipient_img, overrides=overrides)["output"]
                    row = {"scene_id": scene_id, "recipient": recipient, "donor": donor, "condition": cond_name,
                           "intervention_strength": len(overrides),
                           **metrics_row(swapped_out, clean_t, cache[recipient]["output"])}
                    skip_rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"  skip-progressive [{i + 1}/{len(subset)}] {scene_id}", flush=True)

    import pandas as pd
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skip_rows).to_csv(INTERVENTIONS_DIR / "skip_connection_progressive.csv", index=False)
    print(f"wrote {INTERVENTIONS_DIR / 'skip_connection_progressive.csv'} ({len(skip_rows)} rows)")

    # ---- Control C: cross-scene, same degradation ----
    cross_scene_rows = []
    subset = scene_rows[:N_CROSS_SCENE_SCENES]
    normals = {}
    for scene_row in subset:
        normals[scene_row["scene_id"]] = run_normal(model, scene_row, device)

    scene_ids = list(normals.keys())
    skipped_shape_mismatch = 0
    for i in range(1, len(scene_ids)):
        recipient_scene, donor_scene = scene_ids[i], scene_ids[i - 1]
        r_clean, r_cache = normals[recipient_scene]
        d_clean, d_cache = normals[donor_scene]
        for deg in DEGS:
            recipient_img = r_cache[deg]["input"]
            donor_latent = d_cache[deg]["latent_pre"]
            if donor_latent.shape != r_cache[deg]["latent_pre"].shape:
                # different scenes can have different orientation (portrait vs
                # landscape crops) -- shape mismatch makes the swap undefined;
                # skip rather than silently reshape/crop (would confound the result)
                skipped_shape_mismatch += 1
                continue
            swapped_out = manual_forward(model, recipient_img, overrides={"latent_pre": donor_latent})["output"]
            row = {"recipient_scene": recipient_scene, "donor_scene": donor_scene, "degradation": deg,
                   "point": "latent_pre",
                   **metrics_row(swapped_out, r_clean, r_cache[deg]["output"])}
            cross_scene_rows.append(row)
    if skipped_shape_mismatch:
        print(f"  skipped {skipped_shape_mismatch} cross-scene pairs due to orientation/shape mismatch "
              f"(portrait vs landscape crops -- not a bug, an expected data property)", flush=True)
    pd.DataFrame(cross_scene_rows).to_csv(CONTROLS_DIR / "cross_scene_control.csv", index=False)
    print(f"wrote {CONTROLS_DIR / 'cross_scene_control.csv'} ({len(cross_scene_rows)} rows)")

    # ---- Control D: random representation (distribution-matched) ----
    random_rows = []
    subset = scene_rows[:N_RANDOM_ZERO_SCENES]
    rng = np.random.RandomState(0)
    for scene_row in subset:
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)
        recipient_img = cache["Rain"]["input"]
        own_latent = cache["Rain"]["latent_pre"]
        rand_t = torch.from_numpy(
            rng.randn(*own_latent.shape).astype(np.float32)).to(device)
        rand_t = rand_t * own_latent.std().item() + own_latent.mean().item()
        swapped_out = manual_forward(model, recipient_img, overrides={"latent_pre": rand_t})["output"]
        row = {"scene_id": scene_id, "recipient": "Rain", "point": "latent_pre", "control_type": "random",
               **metrics_row(swapped_out, clean_t, cache["Rain"]["output"])}
        random_rows.append(row)
    pd.DataFrame(random_rows).to_csv(CONTROLS_DIR / "random_control.csv", index=False)
    print(f"wrote {CONTROLS_DIR / 'random_control.csv'} ({len(random_rows)} rows)")

    # ---- Control E: zero + dataset-mean representation ----
    zero_rows = []
    # first pass: accumulate dataset-mean Rain latent_pre, restricted to the
    # MAJORITY orientation (scenes are a mix of portrait/landscape crops,
    # e.g. (1,384,40,60) vs (1,384,60,40) -- averaging across incompatible
    # shapes is undefined, so we average only same-shape tensors and later
    # only apply this control to recipient scenes sharing that shape,
    # skipping the rest (documented, not silently reshaped/cropped)
    from collections import Counter
    latents_by_shape: dict[tuple, list] = {}
    for scene_row in scene_rows:
        img_t = to_tensor(load_rgb(scene_row["rain_image_path"]), device)
        with torch.no_grad():
            lat = manual_forward(model, img_t)["latent_pre"]
        latents_by_shape.setdefault(tuple(lat.shape), []).append(lat)
    majority_shape = max(latents_by_shape, key=lambda k: len(latents_by_shape[k]))
    mean_latent = torch.stack(latents_by_shape[majority_shape]).mean(dim=0)
    print(f"computed dataset-mean Rain latent_pre over {len(latents_by_shape[majority_shape])} scenes "
          f"sharing the majority shape {majority_shape} "
          f"(out of {sum(len(v) for v in latents_by_shape.values())} total scenes)", flush=True)

    subset = scene_rows[:N_RANDOM_ZERO_SCENES]
    skipped_mean_shape = 0
    for scene_row in subset:
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)
        recipient_img = cache["Rain"]["input"]
        own_latent = cache["Rain"]["latent_pre"]

        zero_t = torch.zeros_like(own_latent)
        swapped_zero = manual_forward(model, recipient_img, overrides={"latent_pre": zero_t})["output"]
        zero_rows.append({"scene_id": scene_id, "recipient": "Rain", "point": "latent_pre", "control_type": "zero",
                           **metrics_row(swapped_zero, clean_t, cache["Rain"]["output"])})

        if tuple(own_latent.shape) != majority_shape:
            skipped_mean_shape += 1
            continue
        swapped_mean = manual_forward(model, recipient_img, overrides={"latent_pre": mean_latent})["output"]
        zero_rows.append({"scene_id": scene_id, "recipient": "Rain", "point": "latent_pre",
                           "control_type": "dataset_mean",
                           **metrics_row(swapped_mean, clean_t, cache["Rain"]["output"])})
    pd.DataFrame(zero_rows).to_csv(CONTROLS_DIR / "zero_mean_control.csv", index=False)
    print(f"wrote {CONTROLS_DIR / 'zero_mean_control.csv'} ({len(zero_rows)} rows); "
          f"skipped dataset_mean for {skipped_mean_shape} scenes with non-majority orientation")


if __name__ == "__main__":
    main()
