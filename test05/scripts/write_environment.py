"""TEST05: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST05 = Path(__file__).resolve().parent.parent
REPO = TEST05.parent
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
    "TEST05 -- Degradation-Specific Representation Discovery -- Environment Record",
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
    "Dataset: TEST03's exact 100 scenes x Rain/Haze/Noise (read-only reference, not "
    "regenerated) -- test03/data/{rain,haze,noise,clean}/scene_NNN.png",
    "Model: released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt, 28,784,824 params, "
    "0 missing / 0 unexpected keys -- IDENTICAL checkpoint/loader to test01-04",
    "Intervention mechanism: TEST04's verified-bit-exact manual_forward (read-only "
    "import from test04/src/intervention.py, not modified), extended in TEST05 to "
    "substitute partial (channel-subset) tensors, not just whole tensors",
    "No retraining, no fine-tuning, no weight modification, no degradation label ever "
    "supplied to AdaIR. FFT/mask code (raw_low) untouched, per instruction. "
    "raw_low included only as a negative control, per instruction.",
]

out_path = TEST05 / "results" / "environment.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
