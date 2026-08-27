"""Phases 1, 2, 6, 12: run the 3-condition ablation (released / modified_mask /
no_frequency) over the same 300-image manifest used in the original analysis,
recording per-image PSNR/SSIM/latency/peak-memory for all 300 x 3 = 900 runs,
plus full internal-tensor traces + .pt storage for 9 representative images
(3 per degradation) x 3 variants = 27 deep traces.

Checkpoint compatibility (Phase 4): all three variants load the exact same
adair3d.ckpt via the exact same strict loader -- verified once at startup per
variant (0 missing / 0 unexpected keys) and asserted before any inference
runs. No retraining is performed; see model_variants.py docstring.

Usage (on devon, adair-distill env, PINNED -- flaky cores 8-11):
  taskset -c 0-7,12-31 python run_ablation.py
  taskset -c 0-7,12-31 python run_ablation.py --limit 9   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST01 = Path(__file__).resolve().parent.parent
REPO = TEST01.parent  # teacher-experiments/
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(TEST01 / "scripts"))
from model_variants import load_variant, VARIANTS  # noqa: E402
from run_inference import crop_img, load_rgb, add_gaussian_noise, to_tensor  # noqa: E402
from stats_utils import tensor_stats, psnr_ssim  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = REPO / "manifest.csv"

RESULTS_DIR = TEST01 / "results"
TENSORS_DIR = RESULTS_DIR / "tensors"
CSV_DIR = TEST01 / "csv_export"
AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
FMIM_FEATURES = ["conv_feat", "fft_magnitude", "raw_low", "raw_high", "mined_low", "mined_high"]
N_REPRESENTATIVE_PER_DEG = 3

GIT_SHA = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def pick_representative(rows: list[dict]) -> set[str]:
    ids = set()
    for deg in ("Rain", "Haze", "Noise"):
        deg_rows = [r for r in rows if r["Degradation"] == deg][:N_REPRESENTATIVE_PER_DEG]
        ids.update(r["Image_ID"] for r in deg_rows)
    return ids


def run_variant(variant: str, rows: list[dict], representative_ids: set[str], device: str):
    print(f"\n{'=' * 80}\nVARIANT: {variant}\n{'=' * 80}", flush=True)
    model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, variant)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    print(f"checkpoint compatibility OK: {n_params:,} params, 0 missing/0 unexpected keys "
          f"(see model_variants.py load_variant -> load_adair)", flush=True)

    per_image_rows = []
    mgb_rows, freq_rows, fmim_rows, fmom_rows = [], [], [], []
    t_start = time.time()

    for idx, row in enumerate(rows):
        image_id, deg, dataset = row["Image_ID"], row["Degradation"], row["Dataset"]
        gt_np = crop_img(load_rgb(row["gt_path"]))
        if deg == "Noise":
            sigma = float(row["noise_sigma"])
            # Deterministic PER-IMAGE seed (not just a global seed(0) at process
            # start) -- required because this script evaluates the SAME image
            # under 3 different model variants in sequence. A single global
            # seed would let the RNG stream drift between variants, so each
            # variant would synthesize a DIFFERENT noisy input for the same
            # Image_ID, silently breaking the paired same-image comparison
            # Phase 7 requires. Seeding by a hash of the image_id makes the
            # synthesized noise identical across variants and across runs.
            rng = np.random.RandomState(abs(hash(image_id)) % (2 ** 31))
            degraded_np = add_gaussian_noise(gt_np, sigma, rng=rng)
        else:
            degraded_np = crop_img(load_rgb(row["input_path"]))
            sigma = None

        degraded_t = to_tensor(degraded_np, device)
        gt_t = to_tensor(gt_np, device)

        is_representative = image_id in representative_ids
        recorder.start()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            restored_t = model(degraded_t)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        psnr, ssim = psnr_ssim(restored_t, gt_t)
        per_image_rows.append({
            "model": variant, "Image_ID": image_id, "Degradation": deg, "Dataset": dataset,
            "Filename": Path(row["input_path"]).name, "Noise_Sigma": sigma or "",
            "psnr": psnr, "ssim": ssim, "inference_time_ms": latency_ms, "peak_memory_mb": peak_mem_mb,
        })

        if is_representative:
            snap = recorder.snapshot_cpu()
            for aflb in AFLB_NAMES:
                d = snap[aflb]
                th = d["threshold_alpha_beta"]
                alpha = th[0, 0, 0, 0].item()
                beta = th[0, 1, 0, 0].item()
                mask_pct = d["mask"].float().mean().item() * 100
                mgb_rows.append({"model": variant, "Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                                  "alpha": alpha, "beta": beta, "mask_area_pct": mask_pct})

                if variant != "no_frequency":
                    fft = d["fft_shifted"]
                    mask = d["mask"]
                    energy_total = (fft.abs() ** 2).sum().item()
                    energy_low = ((fft * mask).abs() ** 2).sum().item()
                    energy_high = ((fft * (1 - mask)).abs() ** 2).sum().item()
                    freq_rows.append({
                        "model": variant, "Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                        "fft_energy": energy_total, "low_energy": energy_low, "high_energy": energy_high,
                        "low_pct": 100 * energy_low / energy_total, "high_pct": 100 * energy_high / energy_total,
                    })
                    fft_mag = fft.abs()
                else:
                    fft_mag = torch.zeros(1)

                for feat_name in FMIM_FEATURES:
                    t = fft_mag if feat_name == "fft_magnitude" else d[feat_name]
                    s = tensor_stats(t)
                    fmim_rows.append({"model": variant, "Image_ID": image_id, "Degradation": deg,
                                       "AFLB": aflb, "Feature": feat_name, **s})

                hl_s = tensor_stats(d["hl_spatial_weight"])
                lh_s = tensor_stats(d["lh_channel_weight"])
                agg_s = tensor_stats(d["fmom_agg"])
                out_s = tensor_stats(d["aflb_out"])
                fmom_rows.append({
                    "model": variant, "Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                    "hl_mean": hl_s["mean"], "hl_std": hl_s["std"],
                    "lh_mean": lh_s["mean"], "lh_std": lh_s["std"],
                    "agg_energy": agg_s["energy"],
                    "aflb_out_mean": out_s["mean"], "aflb_out_std": out_s["std"], "aflb_out_energy": out_s["energy"],
                })

                out_dir = TENSORS_DIR / variant / deg.lower() / image_id
                out_dir.mkdir(parents=True, exist_ok=True)
                bundle = {k: (v.half() if v.is_floating_point() else v) for k, v in d.items()
                          if not v.is_complex()}
                bundle["fft_shifted_abs"] = d["fft_shifted"].abs().half()
                torch.save(bundle, out_dir / f"{aflb.lower()}.pt")

        if (idx + 1) % 50 == 0 or idx == len(rows) - 1:
            print(f"  [{idx + 1}/{len(rows)}] {image_id} ({deg}) psnr={psnr:.2f} ssim={ssim:.4f} "
                  f"lat={latency_ms:.1f}ms peak_mem={peak_mem_mb:.0f}MB "
                  f"elapsed={time.time() - t_start:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()
    return per_image_rows, mgb_rows, freq_rows, fmim_rows, fmom_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        by_deg: dict[str, list] = {}
        for r in rows:
            by_deg.setdefault(r["Degradation"], []).append(r)
        per_deg = max(1, args.limit // len(by_deg))
        rows = [r for rs in by_deg.values() for r in rs[:per_deg]]

    representative_ids = pick_representative(rows)
    print(f"{len(rows)} images, {len(representative_ids)} representative "
          f"(deep-traced): {sorted(representative_ids)}", flush=True)

    all_per_image, all_mgb, all_freq, all_fmim, all_fmom = [], [], [], [], []
    for variant in args.variants:
        pi, mgb, freq, fmim, fmom = run_variant(variant, rows, representative_ids, args.device)
        all_per_image += pi
        all_mgb += mgb
        all_freq += freq
        all_fmim += fmim
        all_fmom += fmom

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline").mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_per_image)
    df_all.to_csv(CSV_DIR / "20_Per_Image_All_Variants.csv", index=False)

    if "released" in args.variants:
        df_baseline = df_all[df_all.model == "released"][
            ["Image_ID", "Degradation", "Filename", "psnr", "ssim", "inference_time_ms"]
        ].rename(columns={"Image_ID": "image_id", "Degradation": "degradation",
                           "psnr": "psnr", "ssim": "ssim"})
        df_baseline.insert(0, "image_id_int", range(1, len(df_baseline) + 1))
        df_baseline.to_csv(RESULTS_DIR / "baseline" / "results.csv", index=False)
        print(f"\nwrote {RESULTS_DIR / 'baseline' / 'results.csv'} ({len(df_baseline)} rows)")

    comparison_rows = []
    for model in df_all.model.unique():
        for deg in df_all.Degradation.unique():
            sub = df_all[(df_all.model == model) & (df_all.Degradation == deg)]
            comparison_rows.append({
                "model": model, "degradation": deg, "num_images": len(sub),
                "mean_psnr": sub.psnr.mean(), "std_psnr": sub.psnr.std(), "median_psnr": sub.psnr.median(),
                "mean_ssim": sub.ssim.mean(), "std_ssim": sub.ssim.std(),
                "mean_latency_ms": sub.inference_time_ms.mean(), "peak_memory_mb": sub.peak_memory_mb.max(),
                "parameters": 28_784_824,
            })
        sub = df_all[df_all.model == model]
        comparison_rows.append({
            "model": model, "degradation": "ALL", "num_images": len(sub),
            "mean_psnr": sub.psnr.mean(), "std_psnr": sub.psnr.std(), "median_psnr": sub.psnr.median(),
            "mean_ssim": sub.ssim.mean(), "std_ssim": sub.ssim.std(),
            "mean_latency_ms": sub.inference_time_ms.mean(), "peak_memory_mb": sub.peak_memory_mb.max(),
            "parameters": 28_784_824,
        })
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'comparison.csv'}")
    print(df_comparison.to_string(index=False))

    if all_mgb:
        pd.DataFrame(all_mgb).to_csv(CSV_DIR / "21_Ablation_MGB_Values.csv", index=False)
        pd.DataFrame(all_freq).to_csv(CSV_DIR / "22_Ablation_Frequency_Statistics.csv", index=False)
        pd.DataFrame(all_fmim).to_csv(CSV_DIR / "23_Ablation_FMiM_Statistics.csv", index=False)
        pd.DataFrame(all_fmom).to_csv(CSV_DIR / "24_Ablation_FMoM_Statistics.csv", index=False)
        print(f"wrote representative-image internal traces -> {CSV_DIR}")

    print(f"\nGit SHA (AdaIR source): {GIT_SHA}")
    print("Done.")


if __name__ == "__main__":
    main()
