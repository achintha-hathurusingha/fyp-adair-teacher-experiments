"""TEST17 Phase 9: INT8 PTQ for the 4 full (all TRAINED) graphs via
Qualcomm AI Hub's real quantize -> compile -> profile -> inference path,
exact same protocol as TEST16 (real calibration crops, not placeholder
random data). Only runs if all 4 FP32 graphs compiled cleanly (Phase 9's
stated precondition).

Since all 4 models are trained this pass (unlike TEST16), all 4 get a real
on-device INT8 PSNR/SSIM measurement, not just a subset.

Usage (devon, adair-distill env):
  python int8_full.py --device "Snapdragon 8 Elite QRD"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TEST17 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST17.parent
TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
ONNX_DIR = TEST17 / "results" / "onnx_models"
STATS_DIR = TEST17 / "results" / "statistics"
N_CALIB = 32
N_INFER_CHECK = 12
DEGS = ["Rain", "Haze", "Noise"]
MODEL_NAMES = ["A", "N", "F2", "NF2"]


def load_resized(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def gather_val_samples(n, seed=0):
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]
    samples = []
    for row in val_rows:
        for deg in DEGS:
            samples.append({"clean": row["clean_path"], "degraded": row[f"{deg.lower()}_path"], "deg": deg})
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    return [samples[i] for i in idx]


def psnr_ssim(pred_hwc_u8, target_hwc_u8):
    psnr = float(peak_signal_noise_ratio(target_hwc_u8, pred_hwc_u8, data_range=255))
    ssim = float(structural_similarity(target_hwc_u8, pred_hwc_u8, data_range=255, channel_axis=2))
    return psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    args = ap.parse_args()

    fp32_df = pd.read_csv(STATS_DIR / "full_graph_benchmark.csv")
    if not fp32_df["compile_success"].fillna(False).all():
        print("SKIPPING INT8: not all FP32 graphs compiled cleanly (Phase 9 precondition unmet).")
        pd.DataFrame([]).to_csv(STATS_DIR / "int8_benchmark.csv", index=False)
        return

    import qai_hub as hub

    input_name = onnx.load(str(ONNX_DIR / "A.onnx")).graph.input[0].name
    device = hub.Device(args.device)

    calib_samples = gather_val_samples(N_CALIB, seed=0)
    calib_images = [load_resized(s["degraded"]).transpose(2, 0, 1)[None] for s in calib_samples]
    calibration_data = {input_name: calib_images}

    infer_samples = gather_val_samples(N_INFER_CHECK, seed=1)
    infer_inputs = [load_resized(s["degraded"]).transpose(2, 0, 1)[None].astype(np.float32) for s in infer_samples]

    rows = []
    for name in MODEL_NAMES:
        onnx_path = ONNX_DIR / f"{name}.onnx"
        row = {"name": name}
        try:
            print(f"{name}: submitting INT8 quantize job...", flush=True)
            qjob = hub.submit_quantize_job(
                model=str(onnx_path), calibration_data=calibration_data,
                weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8,
                name=f"test17-{name}-quantize",
            )
            qjob.wait()
            int8_model = qjob.get_target_model()
            row["quantize_success"] = int8_model is not None
            if int8_model is None:
                rows.append(row)
                continue

            print(f"{name}: submitting INT8 compile job...", flush=True)
            cjob = hub.submit_compile_job(model=int8_model, device=device,
                                           options="--target_runtime qnn_context_binary",
                                           name=f"test17-{name}-int8-compile")
            cjob.wait()
            row["compile_success"] = cjob.get_status().success
            compiled = cjob.get_target_model()

            print(f"{name}: submitting INT8 profile job...", flush=True)
            pjob = hub.submit_profile_job(model=compiled, device=device, name=f"test17-{name}-int8-profile")
            pjob.wait()
            row["profile_success"] = pjob.get_status().success
            if row["profile_success"]:
                profile = pjob.download_profile()
                summary = profile.get("execution_summary", {})
                row["latency_ms"] = summary.get("estimated_inference_time")
                row["peak_memory_bytes"] = summary.get("estimated_inference_peak_memory")
                detail = profile.get("execution_detail", [])
                cu = {d.get("compute_unit") for d in detail if "compute_unit" in d}
                row["compute_units_used"] = ",".join(sorted(cu)) if cu else "unknown"
                row["npu_only"] = cu.issubset({"NPU"}) if cu else None
                row["any_cpu_fallback"] = "CPU" in cu

            print(f"{name}: submitting real INT8 inference job for PSNR/SSIM...", flush=True)
            ijob = hub.submit_inference_job(model=compiled, device=device,
                                             inputs={input_name: infer_inputs},
                                             name=f"test17-{name}-int8-infer")
            ijob.wait()
            outputs = ijob.download_output_data()
            out_arrs = list(outputs.values())[0] if isinstance(outputs, dict) else outputs
            psnrs, ssims = [], []
            for out_arr, s in zip(out_arrs, infer_samples):
                pred = np.asarray(out_arr).reshape(3, 256, 256).transpose(1, 2, 0)
                pred_u8 = (np.clip(pred, 0, 1) * 255).round().astype(np.uint8)
                tgt_u8 = (load_resized(s["clean"]) * 255).round().astype(np.uint8)
                p, s_ = psnr_ssim(pred_u8, tgt_u8)
                psnrs.append(p)
                ssims.append(s_)
            row["int8_psnr"] = float(np.mean(psnrs))
            row["int8_ssim"] = float(np.mean(ssims))
            row["int8_n_samples"] = len(psnrs)
        except Exception as e:
            row["error"] = str(e)
            print(f"{name}: INT8 pipeline error -- {e}", flush=True)
        rows.append(row)
        print(f"{name}: {row}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "int8_benchmark.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'int8_benchmark.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
