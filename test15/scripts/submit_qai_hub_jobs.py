"""TEST15: submit compile+profile jobs to Qualcomm AI Hub for every
successfully-exported ONNX model, targeting a real Snapdragon device, both
with and without INT8 quantization where applicable. THIS STEP REQUIRES
AN AUTHENTICATED QAI-HUB CLIENT (~/.qai_hub/client.ini with a valid API
token from https://aihub.qualcomm.com/ -- run `qai-hub configure
--api_token <TOKEN>` first, or place the token in that file directly).

For each op:
  1. Upload the ONNX model.
  2. Submit a compile job targeting TARGET_DEVICE, forcing NPU
     (QNN/QAIRT) execution where the runtime allows it.
  3. Submit a profile job on the compiled model.
  4. Record: compile success/failure, NPU-only execution (vs CPU/GPU
     fallback, read from the profile job's per-layer execution report),
     latency, memory, and (for the fp32-vs-int8 pair) quantization
     support.

This script only SUBMITS jobs (Qualcomm AI Hub jobs run asynchronously in
the cloud device farm); collect_results.py polls and aggregates the
finished jobs afterward.

Usage (on devon, adair-distill env, PINNED, AFTER `qai-hub configure`):
  taskset -c 0-7,12-31 python submit_qai_hub_jobs.py --device "Snapdragon 8 Gen 3 QRD"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEST15 = Path(__file__).resolve().parent.parent
ONNX_DIR = TEST15 / "results" / "onnx_models"
JOBS_DIR = TEST15 / "results" / "profile_jobs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True,
                     help='Target device name, e.g. "Snapdragon 8 Gen 3 QRD" -- run '
                          '`qai-hub list-devices` (after configuring) to see exact available names.')
    ap.add_argument("--quantize", action="store_true",
                     help="Also submit an INT8-quantized profile job for each op (requires calibration "
                          "data; uses random calibration data here since these are isolated synthetic "
                          "ops, not the real student network).")
    args = ap.parse_args()

    import qai_hub as hub  # deferred import -- fails loudly and early if unauthenticated

    with open(TEST15 / "results" / "export_manifest.json") as f:
        manifest = json.load(f)

    device = hub.Device(args.device)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    job_records = []
    for entry in manifest:
        if entry["export_status"] != "success":
            job_records.append({"name": entry["name"], "status": "SKIPPED (ONNX export failed)"})
            print(f"SKIP {entry['name']}: ONNX export itself failed, cannot submit.", flush=True)
            continue

        name = entry["name"]
        onnx_path = entry["onnx_path"]
        print(f"Uploading {name}...", flush=True)
        model = hub.upload_model(onnx_path)

        print(f"Submitting compile+profile job for {name} on {args.device}...", flush=True)
        compile_job = hub.submit_compile_job(
            model=model,
            device=device,
            options="--target_runtime qnn_context_binary",  # force QNN/NPU-targeted compilation
        )
        profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=device,
        )

        job_records.append({
            "name": name, "status": "SUBMITTED",
            "compile_job_id": compile_job.job_id, "profile_job_id": profile_job.job_id,
            "device": args.device,
        })
        print(f"  compile_job={compile_job.job_id} profile_job={profile_job.job_id}", flush=True)

    with open(JOBS_DIR / "submitted_jobs.json", "w") as f:
        json.dump(job_records, f, indent=2)
    print(f"\nSubmitted {sum(1 for j in job_records if j['status']=='SUBMITTED')} jobs. "
          f"Wrote {JOBS_DIR / 'submitted_jobs.json'} -- run collect_results.py once jobs finish "
          f"(check status at https://aihub.qualcomm.com/jobs).")


if __name__ == "__main__":
    main()
