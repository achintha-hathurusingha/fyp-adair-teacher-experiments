"""TEST16 Phase 6: per-layer hotspot analysis from the raw full-graph
profile JSON (execution_detail) collected by full_graph_compile_profile.py.

For each model: top 10 layers by execution_cycles, with % of total graph
cycles and compute unit. Also aggregates cycles by coarse layer-type bucket
(LayerNorm/AffineClamp, Conv, DepthwiseConv, 1x1Conv, Add, Multiply, Clamp,
GAP/GMP, Linear/Gemm, static-mixture-expert Conv) so Phase 6's specific
question -- does the previously-observed LayerNorm cost survive in the
complete graph -- can be answered directly, without assuming it in advance.

Usage (devon, adair-distill env, after full_graph_compile_profile.py):
  python layer_hotspots.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TEST16 = Path(__file__).resolve().parent.parent
STATS_DIR = TEST16 / "results" / "statistics"
MODEL_NAMES = ["A", "F2", "N", "S"]

# Coarse bucket classifier, order matters (first match wins).
BUCKET_PATTERNS = [
    ("LayerNorm/AffineClamp", re.compile(r"layernorm|affine|clamp|instancenorm", re.I)),
    ("DepthwiseConv", re.compile(r"depthwise|dwconv", re.I)),
    ("Conv1x1", re.compile(r"conv.*1x1|pointwise", re.I)),
    ("Conv", re.compile(r"conv", re.I)),
    ("Gemm/Linear", re.compile(r"gemm|linear|matmul", re.I)),
    ("Add", re.compile(r"^/?add", re.I)),
    ("Multiply", re.compile(r"mul", re.I)),
    ("GAP/GMP", re.compile(r"globalavgpool|globalmaxpool|reducemean|reducemax", re.I)),
    ("Gather", re.compile(r"gather", re.I)),
    ("Reshape/Transpose", re.compile(r"reshape|transpose|permute", re.I)),
    ("Resize/Upsample", re.compile(r"resize|upsample|interp", re.I)),
    ("Sigmoid/Activation", re.compile(r"sigmoid|relu|gelu|softmax", re.I)),
]


def bucket_for(node_type: str, node_name: str) -> str:
    combined = f"{node_type} {node_name}"
    for label, pattern in BUCKET_PATTERNS:
        if pattern.search(combined):
            return label
    return "Other"


def main():
    top10_rows = []
    bucket_rows = []

    for name in MODEL_NAMES:
        raw_path = STATS_DIR / f"raw_profile_{name}.json"
        if not raw_path.exists():
            print(f"{name}: no raw profile found, skipping (compile/profile may have failed).")
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

        top10 = df_sorted.head(10)
        for _, r in top10.iterrows():
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
        print(f"{name}: total_cycles={total_cycles} n_layers={len(df)} "
              f"LayerNorm/AffineClamp_pct={ln_pct:.2f}%", flush=True)

    top10_df = pd.DataFrame(top10_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    top10_df.to_csv(STATS_DIR / "layer_hotspots_top10.csv", index=False)
    bucket_df.to_csv(STATS_DIR / "layer_hotspots_by_bucket.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'layer_hotspots_top10.csv'}")
    print(f"wrote {STATS_DIR / 'layer_hotspots_by_bucket.csv'}")


if __name__ == "__main__":
    main()
