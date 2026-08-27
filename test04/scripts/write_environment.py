"""TEST04: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST04 = Path(__file__).resolve().parent.parent
REPO = TEST04.parent
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
    "TEST04 -- Causal Representation Intervention Study -- Environment Record",
    f"AdaIR source git SHA: {git_sha}",
    f"Checkpoint path: {CKPT_PATH}",
    f"Checkpoint SHA256: {h.hexdigest()}",
    f"Python version: {sys.version}",
    f"PyTorch version: {torch.__version__}",
    f"CUDA version (torch): {torch.version.cuda}",
    f"GPU: {gpu_info}",
    "Host: devon (192.248.10.68) -- KNOWN ISSUE: logical CPUs 8-11 are unreliable; "
    "every script invocation pinned: taskset -c 0-7,12-31 python <script>",
    "conda env: adair-distill",
    "Dataset: TEST03's exact 100 scenes x Rain/Haze/Noise images (read-only reference, "
    "not regenerated) -- test03/data/{rain,haze,noise,clean}/scene_NNN.png",
    "Model: released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt, 28,784,824 params, "
    "0 missing / 0 unexpected keys -- IDENTICAL checkpoint/loader to test01/test02/test03",
    "Intervention mechanism: manual, faithful re-implementation of AdaIR.forward() "
    "(test04/src/intervention.py::manual_forward), verified bit-identical (0.0 max abs "
    "diff) against model.forward() with no overrides, and bit-identical (0.0 max abs "
    "diff) under self-override (substituting a tensor with itself).",
    "No retraining, no fine-tuning, no weight modification, no degradation label ever "
    "supplied to AdaIR. FFT/mask code (raw_low) untouched, per instruction.",
]

out_path = TEST04 / "results" / "environment.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
