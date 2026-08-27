"""Phase 17: record environment/reproducibility info."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

TEST02 = Path(__file__).resolve().parent.parent
REPO = TEST02.parent
ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"

git_sha = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
import hashlib
h = hashlib.sha256()
with open(CKPT_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

lines = [
    "TEST02 -- Degradation Representation Analysis of AdaIR -- Environment Record",
    f"AdaIR source git SHA: {git_sha}",
    f"Checkpoint path: {CKPT_PATH}",
    f"Checkpoint SHA256: {h.hexdigest()}",
    f"Python version: {sys.version}",
    f"PyTorch version: {torch.__version__}",
    f"CUDA version (torch): {torch.version.cuda}",
    f"GPU: {gpu_info}",
    "Host: devon (192.248.10.68) -- KNOWN ISSUE: logical CPUs 8-11 are unreliable "
    "and have been observed to silently corrupt in-process data. Every script "
    "invocation in this project is pinned: taskset -c 0-7,12-31 python <script>",
    "conda env: adair-distill",
    "Random seed: 0 (np.random.seed, torch.manual_seed) + per-image deterministic "
    "noise seeding (np.random.RandomState(hash(image_id))) for the Noise degradation",
    "Dataset: same 300-image manifest as test01 (100 Rain100L / 100 SOTS-outdoor / 100 BSD68)",
    "Model: released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt, 28,784,824 params, "
    "0 missing / 0 unexpected keys",
]

out_path = TEST02 / "results" / "environment.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
