"""TEST06-R: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST06_R = Path(__file__).resolve().parent.parent
REPO = TEST06_R.parent
ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"

git_sha = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
with open(CKPT_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

lines = [
    "TEST06-R -- Corrected Re-Run / Statistical + Internal Propagation Audit -- Environment Record",
    "PURPOSE: tighten TEST06's causal inference (balanced N, paired bootstrap/permutation/"
    "Wilcoxon tests, pre-specified practical-equivalence threshold) and add internal "
    "FMiM/FMoM propagation tracing. Reuses the EXACT original TEST06 06-E dataset "
    "(read-only). Does NOT modify test06/, test01-05.5, fyp-adair-distill, the AdaIR "
    "source, or the checkpoint.",
    f"AdaIR source git SHA: {git_sha}",
    f"Checkpoint path: {CKPT_PATH}",
    f"Checkpoint SHA256: {h.hexdigest()}",
    "Expected checkpoint SHA256 (from TEST01-06 record): "
    "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb -- MUST match.",
    f"Python version: {sys.version}",
    f"PyTorch version: {torch.__version__}",
    f"CUDA version (torch): {torch.version.cuda}",
    f"GPU: {gpu_info}",
    "Host: devon (192.248.10.68), conda env: adair-distill, taskset -c 0-7,12-31 pinning.",
    "Dataset: test06/results/frequency_intervention/scene_manifest.csv (read-only, NOT "
    "regenerated), 25 scenes minus scene_021 (excluded: discovered to be 1024x104, not "
    "1024x1024, a data-quality bug in the original TEST06 dataset-build script -- see "
    "report/rerun_audit.md). N=24 scenes used throughout TEST06-R.",
    "Model: released, UNMODIFIED AdaIR (decoder=True), same checkpoint as TEST01-06. "
    "No retraining anywhere in TEST06-R.",
    "Intervention point: AFLB3's (raw_high, raw_low) tensors only, identical scope to "
    "the original TEST06, re-verified in Phase 0's audit.",
]

out_path = TEST06_R / "results" / "environment.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
