"""TEST17 Phase 8: per-layer hotspot analysis from the raw full-graph
profile JSON. Top 10 layers by cycles per model, plus a coarse-bucket
aggregation (LayerNorm/AffineClamp, Conv, DepthwiseConv, Conv1x1, Add,
Multiply, GAP/GMP, Gemm/Linear [the F2/NF2 operator's a_head/proj],
Gather, Reshape, Resize, Activation) so Phase 8's critical question --
does adding F2 destroy the efficient fused graph N achieved -- is answered
directly from measured cycle shares, not assumed.

Usage (devon, adair-distill env, after full_graph_compile_profile.py):
  python layer_hotspots.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TEST17 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST17 / "results" / "statistics"
MODEL_NAMES = ["A", "N", "F2", "NF2"]

BUCKET_PATTERNS = [
    ("LayerNorm/AffineClamp", re.compile(r"layernorm|affine|clamp|clip|instancenorm", re.I)),
    ("DepthwiseConv", re.compile(r"depthwise|dwconv", re.I)),
    ("Conv1x1", re.compile(r"conv.*1x1|pointwise|sca", re.I)),
    ("Conv", re.compile(r"conv", re.I)),
    ("Gemm/Linear (F2 operator)", re.compile(r"gemm|linear|matmul|proj|a_head|coeff", re.I)),
    ("Add", re.compile(r"^/?add", re.I)),
    ("Multiply", re.compile(r"mul", re.I)),
    ("GAP/GMP", re.compile(r"globalavgpool|globalmaxpool|reducemean|reducemax", re.I)),
    ("Gather", re.compile(r"gather", re.I)),
    ("Reshape/Transpose", re.compile(r"reshape|transpose|permute|einsum", re.I)),
    ("Resize/Upsample", re.compile(r"resize|upsample|interp|depthtospace", re.I)),
    ("Sigmoid/Activation", re.compile(r"sigmoid|relu|gelu|softmax", re.I)),
]


def bucket_for(node_type: str, node_name: str) -> str:
    combined = f"{node_type} {node_name}"
    for label, pattern in BUCKET_PATTERNS:
        if pattern.search(combined):
            return label
    return "Other"


def main():
    top10_rows, bucket_rows = [], []

    for name in MODEL_NAMES:
        raw_path = STATS_DIR / f"raw_profile_{name}.json"
        if not raw_path.exists():
            print(f"{name}: no raw profile found, skipping.")
            continue
        with open(raw_path) as f:
            profile = json.load(f)
        detail = profile.get("execution_detail", [])
        if not detail:
            print(f"{name}: empty execution_detail, skipping.")
            continue

        total_cycles = sum(int(d.get("execution_cycles", 0)) for d in detail)
        df = pd.DataFrame(detail)
        df["execution_cycles"] = df.get("execution_cycles", 0).astype(int)
        df["pct_of_total"] = df["execution_cycles"] / max(total_cycles, 1) * 100
        df_sorted = df.sort_values("execution_cycles", ascending=False)

        for _, r in df_sorted.head(10).iterrows():
            top10_rows.append({
                "model": name, "layer_name": r.get("name"), "layer_type": r.get("type"),
                "compute_unit": r.get("compute_unit"), "execution_cycles": int(r["execution_cycles"]),
                "pct_of_total_cycles": float(r["pct_of_total"]),
            })

        df["bucket"] = [bucket_for(str(t), str(n)) for t, n in zip(df.get("type", ""), df.get("name", ""))]
        bucket_summary = df.groupby("bucket")["execution_cycles"].sum().sort_values(ascending=False)
        for bucket, cycles in bucket_summary.items():
            bucket_rows.append({
                "model": name, "bucket": bucket, "total_cycles": int(cycles),
                "pct_of_total_cycles": float(cycles / max(total_cycles, 1) * 100),
                "n_layers": int((df["bucket"] == bucket).sum()),
            })
        ln_pct = bucket_summary.get("LayerNorm/AffineClamp", 0) / max(total_cycles, 1) * 100
        f2_pct = bucket_summary.get("Gemm/Linear (F2 operator)", 0) / max(total_cycles, 1) * 100
        print(f"{name}: total_cycles={total_cycles} n_layers={len(df)} "
              f"LayerNorm/AffineClamp_pct={ln_pct:.2f}% F2operator_pct={f2_pct:.2f}%", flush=True)

    pd.DataFrame(top10_rows).to_csv(STATS_DIR / "layer_hotspots_top10.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(STATS_DIR / "layer_hotspots_by_bucket.csv", index=False)
    print(f"\nwrote layer_hotspots_top10.csv and layer_hotspots_by_bucket.csv")


if __name__ == "__main__":
    main()
