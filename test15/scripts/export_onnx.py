"""TEST15: export every operator/combination in op_zoo.py to ONNX. This
step needs no Qualcomm AI Hub credentials -- it only requires torch+onnx,
both already present in the adair-distill environment. Run this first;
submit_qai_hub_jobs.py (which DOES need an API token) consumes its output.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python export_onnx.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from op_zoo import OP_ZOO

TEST15 = Path(__file__).resolve().parent.parent
OUT_DIR = TEST15 / "results" / "onnx_models"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    failed = []

    for name, (module, inputs, input_names) in OP_ZOO.items():
        module.eval()
        onnx_path = OUT_DIR / f"{name}.onnx"
        try:
            torch.onnx.export(
                module, inputs, str(onnx_path),
                input_names=list(input_names),
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
            )
            shapes = [tuple(t.shape) for t in inputs]
            manifest.append({"name": name, "onnx_path": str(onnx_path), "input_names": list(input_names),
                              "input_shapes": shapes, "export_status": "success"})
            print(f"[OK] {name}: exported to {onnx_path.name}, input shapes {shapes}", flush=True)
        except Exception as e:
            failed.append({"name": name, "error": str(e)})
            manifest.append({"name": name, "onnx_path": None, "input_names": list(input_names),
                              "input_shapes": None, "export_status": "FAILED", "error": str(e)})
            print(f"[FAIL] {name}: {e}", flush=True)

    with open(TEST15 / "results" / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(manifest) - len(failed)}/{len(manifest)} operators exported successfully.")
    if failed:
        print("\nFAILED exports (these already tell us something -- an op that can't even reach ONNX")
        print("has no chance of NPU execution regardless of hardware support):")
        for f_ in failed:
            print(f"  {f_['name']}: {f_['error'][:200]}")


if __name__ == "__main__":
    main()
