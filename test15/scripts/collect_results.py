"""TEST15: poll and aggregate finished Qualcomm AI Hub compile+profile
jobs into the final operator-support table. Also requires an
authenticated qai-hub client (same as submit_qai_hub_jobs.py).

For each job, extracts:
  - compile success/failure
  - NPU-only execution vs CPU/GPU fallback (from the profile job's
    per-layer / per-op execution summary)
  - latency (ms, from the profile report)
  - peak memory
  - INT8 support (if a quantized job was also submitted)
  - primary failure reason, if any

Usage (on devon, adair-distill env, PINNED, run AFTER jobs finish -- check
https://aihub.qualcomm.com/jobs for status, or this script will wait/poll):
  taskset -c 0-7,12-31 python collect_results.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

TEST15 = Path(__file__).resolve().parent.parent
JOBS_DIR = TEST15 / "results" / "profile_jobs"
OUT_DIR = TEST15 / "results" / "statistics"


def main():
    import qai_hub as hub  # deferred import -- fails loudly if unauthenticated

    with open(JOBS_DIR / "submitted_jobs.json") as f:
        job_records = json.load(f)

    rows = []
    for rec in job_records:
        if rec["status"] != "SUBMITTED":
            rows.append({"name": rec["name"], "npu_supported": None, "compile_status": rec["status"],
                         "note": "ONNX export failed, never submitted"})
            continue

        name = rec["name"]
        compile_job = hub.get_job(rec["compile_job_id"])
        profile_job = hub.get_job(rec["profile_job_id"])

        # job.wait() blocks until the job reaches a terminal state (SUCCESS
        # or FAILED) -- correct handling of the CREATED/queued state, unlike
        # a manual poll loop that only checks .running (queued jobs are
        # .pending, not .running, and were being treated as already-done).
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
            # profile_data structure per Qualcomm AI Hub's profile report --
            # extract per-op execution unit (NPU/GPU/CPU), total latency, peak memory.
            summary = profile_data.get("execution_summary", {})
            row["latency_ms"] = summary.get("estimated_inference_time") or summary.get("inference_time")
            row["peak_memory_bytes"] = summary.get("estimated_inference_peak_memory")

            layer_details = profile_data.get("execution_detail", [])
            compute_units = {d.get("compute_unit") for d in layer_details if "compute_unit" in d}
            row["compute_units_used"] = ",".join(sorted(compute_units)) if compute_units else "unknown"
            row["npu_only"] = compute_units.issubset({"NPU"}) if compute_units else None
            row["any_cpu_fallback"] = "CPU" in compute_units
            row["any_gpu_fallback"] = "GPU" in compute_units
        else:
            row["latency_ms"] = None
            row["compute_units_used"] = None
            row["npu_only"] = False
            row["failure_reason"] = str(profile_job.get_status())

        rows.append(row)
        print(f"{name}: compile={compile_success} profile={profile_success} "
              f"latency={row.get('latency_ms')} units={row.get('compute_units_used')}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "npu_operator_benchmark.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'npu_operator_benchmark.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
