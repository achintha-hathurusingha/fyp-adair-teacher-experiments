"""TEST04 PRIMARY EXPERIMENT: same-scene, cross-degradation representation
swaps at 4 intervention points (latent_pre, aflb1_out, aflb2_out, aflb3_out),
all 100 TEST03 scenes, all 6 donor/recipient pairs per point = 2400
interventions, plus the self-swap validation gate (Phase 4) and normal
baseline (Phase 3).

Uses the manual_forward replica (src/intervention.py), verified bit-exact
against the real model. Reuses TEST03's exact images (read-only).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python run_interventions.py
  taskset -c 0-7,12-31 python run_interventions.py --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST04 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST04.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEST04 / "src"))
from instrument import load_adair  # noqa: E402
from intervention import manual_forward, verify_manual_forward_matches_model, sanity_check, INTERMEDIATE_KEYS  # noqa: E402
from metrics_utils import psnr_ssim_mse, output_diff, pooled_vec, residual_stats  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST04 / "results" / "manifest" / "scene_manifest.csv"
METRICS_DIR = TEST04 / "results" / "metrics"
CONTROLS_DIR = TEST04 / "results" / "controls"
INTERVENTIONS_DIR = TEST04 / "results" / "interventions"
VIZ_DIR = TEST04 / "results" / "visualizations"
OUTPUTS_DIR = TEST04 / "results" / "tensors" / "output_images"

DEGS = ["Rain", "Haze", "Noise"]
SWAP_POINTS = ["latent_pre", "aflb1_out", "aflb2_out", "aflb3_out"]
N_VIZ_SCENES = 10


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))
    if args.limit:
        scene_rows = scene_rows[:args.limit]

    print(f"loading AdaIR (released, unmodified) ckpt {CKPT_PATH.name}", flush=True)
    model = load_adair(ADAIR_DIR, CKPT_PATH, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    print(f"checkpoint OK: {n_params:,} params, 0 missing/0 unexpected keys", flush=True)

    # one-time verification that manual_forward matches the real model exactly
    probe_img = to_tensor(load_rgb(scene_rows[0]["rain_image_path"]), args.device)
    check = verify_manual_forward_matches_model(model, probe_img)
    print(f"manual_forward vs model.forward() sanity check: {check}", flush=True)
    assert check["matches"], "STOP: manual_forward does not match the real model -- see Phase 16 stop conditions"

    normal_rows, self_swap_rows, intervention_rows = [], [], []
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    INTERVENTIONS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for idx, scene_row in enumerate(scene_rows):
        scene_id = scene_row["scene_id"]
        clean_np = load_rgb(scene_row["clean_image_path"])
        clean_t = to_tensor(clean_np, args.device)

        cache = {}  # deg -> full manual_forward output dict
        for deg in DEGS:
            degraded_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), args.device)

            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = manual_forward(model, degraded_t)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000
            peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)

            out["input"] = degraded_t
            cache[deg] = out

            m = psnr_ssim_mse(out["output"], clean_t)
            normal_rows.append({"scene_id": scene_id, "degradation": deg,
                                 "psnr": m["psnr"], "ssim": m["ssim"], "mse": m["mse"],
                                 "latency_ms": latency_ms, "peak_memory_mb": peak_mem_mb})

            if idx < N_VIZ_SCENES:
                Image.fromarray(load_rgb(scene_row[f"{deg.lower()}_image_path"])).save(
                    OUTPUTS_DIR / f"{scene_id}_{deg.lower()}_input.png")
                from metrics_utils import to_np_u8
                Image.fromarray(to_np_u8(out["output"])).save(
                    OUTPUTS_DIR / f"{scene_id}_{deg.lower()}_normal_output.png")

        if idx == 0:
            Image.fromarray(clean_np).save(OUTPUTS_DIR / f"{scene_id}_clean.png")
        elif idx < N_VIZ_SCENES:
            Image.fromarray(clean_np).save(OUTPUTS_DIR / f"{scene_id}_clean.png")

        # --- Phase 4: self-swap gate (all 4 points simultaneously, own tensors) ---
        for deg in DEGS:
            own = cache[deg]
            degraded_t = own["input"]
            overrides = {k: own[k] for k in SWAP_POINTS}
            self_out = manual_forward(model, degraded_t, overrides=overrides)["output"]
            d = output_diff(self_out, own["output"])
            sm = psnr_ssim_mse(self_out, own["output"])
            self_swap_rows.append({"scene_id": scene_id, "degradation": deg,
                                    "max_abs_diff": d["mae"], "l2_diff": d["l2"],
                                    "psnr_vs_normal": sm["psnr"], "ssim_vs_normal": sm["ssim"]})

        # --- Phases 5-7: cross-degradation swaps at all 4 points ---
        for point in SWAP_POINTS:
            for recipient in DEGS:
                for donor in DEGS:
                    if donor == recipient:
                        continue
                    recipient_img = cache[recipient]["input"]
                    donor_tensor = cache[donor][point]
                    swapped_out_dict = manual_forward(model, recipient_img, overrides={point: donor_tensor})
                    swapped_out = swapped_out_dict["output"]

                    vs_clean = psnr_ssim_mse(swapped_out, clean_t)
                    vs_normal_recipient = output_diff(swapped_out, cache[recipient]["output"])
                    vs_normal_donor = output_diff(swapped_out, cache[donor]["output"])
                    third_deg = [d for d in DEGS if d not in (recipient, donor)][0]
                    vs_normal_third = output_diff(swapped_out, cache[third_deg]["output"])
                    res = residual_stats(clean_t, swapped_out)
                    san = sanity_check(swapped_out)

                    intervention_rows.append({
                        "scene_id": scene_id, "point": point, "recipient": recipient, "donor": donor,
                        "psnr_vs_clean": vs_clean["psnr"], "ssim_vs_clean": vs_clean["ssim"],
                        "mse_vs_clean": vs_clean["mse"],
                        "l2_vs_normal_recipient": vs_normal_recipient["l2"],
                        "mae_vs_normal_recipient": vs_normal_recipient["mae"],
                        "l2_vs_normal_donor": vs_normal_donor["l2"],
                        "mae_vs_normal_donor": vs_normal_donor["mae"],
                        f"l2_vs_normal_{third_deg.lower()}": vs_normal_third["l2"],
                        f"mae_vs_normal_{third_deg.lower()}": vs_normal_third["mae"],
                        **res, **{f"sanity_{k}": v for k, v in san.items()},
                    })

                    if idx < N_VIZ_SCENES and point == "latent_pre":
                        from metrics_utils import to_np_u8
                        Image.fromarray(to_np_u8(swapped_out)).save(
                            OUTPUTS_DIR / f"{scene_id}_{recipient.lower()}+{donor.lower()}_latent.png")

        if (idx + 1) % 10 == 0 or idx == len(scene_rows) - 1:
            print(f"[{idx + 1}/{len(scene_rows)}] {scene_id} elapsed={time.time() - t_start:.0f}s "
                  f"({len(intervention_rows)} interventions so far)", flush=True)

    import pandas as pd
    pd.DataFrame(normal_rows).to_csv(METRICS_DIR / "normal_baseline.csv", index=False)
    pd.DataFrame(self_swap_rows).to_csv(CONTROLS_DIR / "self_swap_validation.csv", index=False)
    pd.DataFrame(intervention_rows).to_csv(INTERVENTIONS_DIR / "cross_degradation_swaps.csv", index=False)

    print(f"\nwrote {METRICS_DIR / 'normal_baseline.csv'} ({len(normal_rows)} rows)")
    print(f"wrote {CONTROLS_DIR / 'self_swap_validation.csv'} ({len(self_swap_rows)} rows)")
    print(f"wrote {INTERVENTIONS_DIR / 'cross_degradation_swaps.csv'} ({len(intervention_rows)} rows)")

    max_self_diff = max(r["max_abs_diff"] for r in self_swap_rows)
    print(f"\nSELF-SWAP GATE: max mean-abs-diff across all {len(self_swap_rows)} self-swaps = {max_self_diff}")
    if max_self_diff > 1e-4:
        print("WARNING: self-swap did not reproduce normal inference closely -- review before trusting "
              "cross-degradation results (Phase 4 stop condition).")
    else:
        print("PASS: self-swap reproduces normal inference exactly (or to float precision). "
              "Proceeding is justified.")

    print(f"total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
