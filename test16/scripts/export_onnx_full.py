"""TEST16 Phase 3: export all 4 complete student models to ONNX using the
exact production export path (fyp-adair-distill/src/export/to_onnx.py's
`export_onnx`, read-only reuse), at the production fixed resolution
(configs/export/qnn_int8.yaml: 256x256, opset 17, no dynamic axes).

For each export, records: success/failure, node count, model size (bytes),
and an operator audit -- specifically checking for Gather, Conv nodes whose
weight input is NOT a graph initializer (i.e. a runtime tensor -- the
TEST15 dynamic-conv risk pattern), and any FFT ops.

For Model S specifically, additionally verifies EVERY Conv node's weight
input resolves to a graph initializer (compile-time constant) -- the
explicit Phase-3 requirement that the static-mixture branches are actually
static.

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

TEST16 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST16.parent
FYP_ADAIR_DISTILL = TEACHER_EXP.parent / "fyp-adair-distill"
ONNX_DIR = TEST16 / "results" / "onnx_models"
OUT_JSON = TEST16 / "results" / "export_manifest.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import load_trained, build_untrained  # noqa: E402

sys.path.insert(0, str(FYP_ADAIR_DISTILL))
from src.export.to_onnx import export_onnx  # noqa: E402 (read-only reuse, production path)

INPUT_SHAPE = (1, 3, 256, 256)  # configs/export/qnn_int8.yaml
OPSET = 17


class _OutputOnly(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out, _ = self.inner(x)
        return out


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
        if "fft" in node.op_type.lower() or "FFT" in node.op_type:
            fft_count += 1
        if node.op_type == "Conv":
            weight_input = node.input[1] if len(node.input) > 1 else None
            weight_is_constant = weight_input in initializer_names
            conv_nodes.append({"name": node.name, "weight_is_constant": weight_is_constant,
                                "weight_input": weight_input})

    n_conv = len(conv_nodes)
    n_conv_dynamic_weight = sum(1 for c in conv_nodes if not c["weight_is_constant"])
    return {
        "node_count": len(graph.node),
        "op_histogram": op_counts,
        "gather_count": gather_count,
        "fft_count": fft_count,
        "n_conv": n_conv,
        "n_conv_dynamic_weight": n_conv_dynamic_weight,
        "all_conv_weights_constant": n_conv_dynamic_weight == 0,
        "unsupported_ops_flagged": fft_count > 0,
        "conv_nodes": conv_nodes,
    }


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    builders = {
        "A": lambda: load_trained("A", seed=0, device="cpu"),
        "F2": lambda: load_trained("F2", seed=0, device="cpu"),
        "N": lambda: build_untrained("N", seed=0, device="cpu"),
        "S": lambda: build_untrained("S", seed=0, device="cpu"),
    }

    for name, builder in builders.items():
        model = builder()
        wrapped = _OutputOnly(model)
        onnx_path = ONNX_DIR / f"{name}.onnx"
        entry = {"name": name, "input_shape": list(INPUT_SHAPE), "opset": OPSET}
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
            if name == "S":
                assert audit["all_conv_weights_constant"], (
                    f"Model S FAILED static-weight audit: {audit['n_conv_dynamic_weight']} "
                    f"Conv nodes have non-constant weights.")
                print("  Model S static-weight audit: PASSED "
                      "(every Conv weight is a compile-time constant)", flush=True)
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
