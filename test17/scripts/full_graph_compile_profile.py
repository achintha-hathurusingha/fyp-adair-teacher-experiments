"""TEST17 Phase 6-7: compile + profile the 4 complete, TRAINED student
graphs on Qualcomm AI Hub, same device/runtime path as TEST15/16
(Snapdragon 8 Elite QRD, --target_runtime qnn_context_binary). Submits then
blocks on job.wait() for each.

Extracts per model: compile success, profile success, per-layer compute
unit assignment (full-graph NPU-only vs CPU/GPU fallback verification --
not accepted on "compile success" alone, per Phase 6's explicit rule), and
the full `all_inference_times` array (~100 warm reps) for
mean/median/p50/p90/p95/std, plus first-load/warm-load time and peak
memory.

Usage (devon, adair-distill env, requires configured qai-hub token):
  python full_graph_compile_profile.py --device "Snapdragon 8 Elite QRD"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEST17 = Path(__file__).resolve().parent.parent
ONNX_DIR = TEST17 / "results" / "onnx_models"
JOBS_DIR = TEST17 / "results" / "profile_jobs"
STATS_DIR = TEST17 / "results" / "statistics"

MODEL_NAMES = ["A", "N", "F2", "NF2"]


def latency_stats(all_times):
    arr = np.array(all_times, dtype=float)
    return {
        "latency_mean_ms": float(arr.mean()), "latency_median_ms": float(np.median(arr)),
        "latency_p50_ms": float(np.percentile(arr, 50)), "latency_p90_ms": float(np.percentile(arr, 90)),
        "latency_p95_ms": float(np.percentile(arr, 95)), "latency_std_ms": float(arr.std()),
        "latency_n_samples": int(arr.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    args = ap.parse_args()

    import qai_hub as hub

    with open(TEST17 / "results" / "export_manifest.json") as f:
        manifest = {e["name"]: e for e in json.load(f)}

    device = hub.Device(args.device)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    job_records = []
    for name in MODEL_NAMES:
        entry = manifest.get(name, {})
        if entry.get("export_status") != "success":
            job_records.append({"name": name, "status": "SKIPPED (ONNX export failed)"})
            continue
        print(f"Uploading {name}...", flush=True)
        model = hub.upload_model(entry["onnx_path"])
        print(f"Submitting compile+profile for {name} on {args.device}...", flush=True)
        compile_job = hub.submit_compile_job(model=model, device=device,
                                              options="--target_runtime qnn_context_binary")
        profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
        job_records.append({"name": name, "status": "SUBMITTED", "compile_job_id": compile_job.job_id,
                             "profile_job_id": profile_job.job_id, "device": args.device})
        print(f"  compile_job={compile_job.job_id} profile_job={profile_job.job_id}", flush=True)

    with open(JOBS_DIR / "submitted_jobs.json", "w") as f:
        json.dump(job_records, f, indent=2)

    rows = []
    for rec in job_records:
        if rec["status"] != "SUBMITTED":
            rows.append({"name": rec["name"], "compile_success": None, "profile_success": None,
                         "note": rec["status"]})
            continue
        name = rec["name"]
        compile_job = hub.get_job(rec["compile_job_id"])
        profile_job = hub.get_job(rec["profile_job_id"])
        print(f"{name}: waiting for compile job {rec['compile_job_id']}...", flush=True)
        compile_job.wait()
        print(f"{name}: waiting for profile job {rec['profile_job_id']}...", flush=True)
        profile_job.wait()

        compile_success = compile_job.get_status().success
        profile_success = profile_job.get_status().success
        row = {"name": name, "compile_success": compile_success, "profile_success": profile_success,
               "device": rec["device"]}

        if profile_success:
            profile_data = profile_job.download_profile()
            with open(STATS_DIR / f"raw_profile_{name}.json", "w") as f:
                json.dump(profile_data, f, indent=2, default=str)
            summary = profile_data.get("execution_summary", {})
            all_times = summary.get("all_inference_times", [])
            if all_times:
                row.update(latency_stats(all_times))
            row["latency_ms"] = summary.get("estimated_inference_time")
            row["peak_memory_bytes"] = summary.get("estimated_inference_peak_memory")
            row["first_load_time_us"] = summary.get("first_load_time")
            row["warm_load_time_us"] = summary.get("warm_load_time")

            layer_details = profile_data.get("execution_detail", [])
            compute_units = {d.get("compute_unit") for d in layer_details if "compute_unit" in d}
            row["compute_units_used"] = ",".join(sorted(compute_units)) if compute_units else "unknown"
            row["npu_only"] = compute_units.issubset({"NPU"}) if compute_units else None
            row["any_cpu_fallback"] = "CPU" in compute_units
            row["any_gpu_fallback"] = "GPU" in compute_units
            row["n_layers"] = len(layer_details)
            row["n_cpu_layers"] = sum(1 for d in layer_details if d.get("compute_unit") == "CPU")
            row["n_gpu_layers"] = sum(1 for d in layer_details if d.get("compute_unit") == "GPU")
        rows.append(row)
        print(f"{name}: compile={compile_success} profile={profile_success} "
              f"latency_mean={row.get('latency_mean_ms')} units={row.get('compute_units_used')}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "full_graph_benchmark.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'full_graph_benchmark.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
