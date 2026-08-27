"""TEST17 Phase 5: export all 4 complete, TRAINED student models to ONNX
using the exact production export path (fyp-adair-distill/src/export/to_onnx.py,
read-only reuse), at the production fixed resolution (256x256, opset 17).
Uses each model's BEST-validation-PSNR checkpoint (the canonical checkpoint
for this experiment, per Phase 1's checkpoint-selection requirement).

For each export: node count, model size, operator audit (Gather, FFT,
Conv-with-non-constant-weight -- the TEST15 dynamic-conv risk pattern).
For F2 and NF2 specifically, verifies every Conv node's weight resolves to
a graph initializer (U/V/base kernels must remain compile-time constants;
only the small coefficient tensor may be runtime-dependent).

Usage (devon, adair-distill env):
  python export_onnx_full.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx
import torch
from torch import nn

TEST17 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST17.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"
CKPT_DIR = TEST17 / "results" / "checkpoints"
ONNX_DIR = TEST17 / "results" / "onnx_models"
OUT_JSON = TEST17 / "results" / "export_manifest.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS  # noqa: E402

sys.path.insert(0, str(FYP_ADAIR_DISTILL))
from src.export.to_onnx import export_onnx  # noqa: E402 (read-only reuse, production path)

INPUT_SHAPE = (1, 3, 256, 256)
OPSET = 17
MODEL_NAMES = ["A", "N", "F2", "NF2"]
SEED = 0  # canonical seed for hardware profiling (representative, not averaged)


class _OutputOnly(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out, _ = self.inner(x)
        return out


def load_best(name: str, seed: int = SEED, device: str = "cpu"):
    model = MODELS[name]()
    state = torch.load(CKPT_DIR / f"model_{name}_seed{seed}_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def audit_graph(onnx_path: Path) -> dict:
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    op_counts: dict[str, int] = {}
    conv_nodes = []
    gather_count = 0
    fft_count = 0
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type == "Gather":
            gather_count += 1
        if "fft" in node.op_type.lower():
            fft_count += 1
        if node.op_type == "Conv":
            weight_input = node.input[1] if len(node.input) > 1 else None
            weight_is_constant = weight_input in initializer_names
            conv_nodes.append({"name": node.name, "weight_is_constant": weight_is_constant})

    n_conv = len(conv_nodes)
    n_conv_dynamic_weight = sum(1 for c in conv_nodes if not c["weight_is_constant"])
    return {
        "node_count": len(graph.node), "op_histogram": op_counts,
        "gather_count": gather_count, "fft_count": fft_count,
        "n_conv": n_conv, "n_conv_dynamic_weight": n_conv_dynamic_weight,
        "all_conv_weights_constant": n_conv_dynamic_weight == 0,
    }


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for name in MODEL_NAMES:
        model = load_best(name)
        wrapped = _OutputOnly(model)
        onnx_path = ONNX_DIR / f"{name}.onnx"
        entry = {"name": name, "seed": SEED, "input_shape": list(INPUT_SHAPE), "opset": OPSET}
        try:
            export_onnx(wrapped, onnx_path, INPUT_SHAPE, OPSET)
            entry["export_status"] = "success"
            entry["onnx_path"] = str(onnx_path)
            entry["model_size_bytes"] = onnx_path.stat().st_size
            audit = audit_graph(onnx_path)
            entry.update(audit)
            print(f"{name}: export OK, nodes={audit['node_count']} "
                  f"size={entry['model_size_bytes']/1e6:.2f}MB "
                  f"conv={audit['n_conv']} dynamic_weight_conv={audit['n_conv_dynamic_weight']} "
                  f"gather={audit['gather_count']} fft={audit['fft_count']}", flush=True)
            if name in ("F2", "NF2"):
                assert audit["all_conv_weights_constant"], (
                    f"{name} FAILED static-weight audit: {audit['n_conv_dynamic_weight']} "
                    f"Conv nodes have non-constant weights (U/V/base kernels must stay static).")
                print(f"  {name} static-weight audit: PASSED "
                      "(U/V/base kernels are compile-time constants)", flush=True)
        except Exception as e:
            entry["export_status"] = "failed"
            entry["error"] = str(e)
            print(f"{name}: export FAILED -- {e}", flush=True)
        manifest.append(entry)

    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
