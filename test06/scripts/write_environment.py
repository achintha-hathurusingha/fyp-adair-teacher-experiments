"""TEST06: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST06 = Path(__file__).resolve().parent.parent
REPO = TEST06.parent
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
    "TEST06 -- Resolution-Dependent Frequency Influence Study -- Environment Record",
    "PURPOSE: determine whether AdaIR's frequency-adaptive mask ever becomes "
    "non-degenerate within a practical resolution range, and if so, whether the "
    "frequency-path representation is causally relevant to restoration. Does NOT "
    "train NAFNet, does NOT modify fyp-adair-distill, does NOT modify test01-05_5. "
    "All outputs confined to test06/.",
    f"AdaIR source git SHA: {git_sha}",
    f"Checkpoint path: {CKPT_PATH}",
    f"Checkpoint SHA256: {h.hexdigest()}",
    "Expected checkpoint SHA256 (from TEST01-05.5 record): "
    "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb -- "
    "MUST match; any mismatch invalidates cross-experiment comparison.",
    f"Python version: {sys.version}",
    f"PyTorch version: {torch.__version__}",
    f"CUDA version (torch): {torch.version.cuda}",
    f"GPU: {gpu_info}",
    "Host: devon (192.248.10.68), conda env: adair-distill, taskset -c 0-7,12-31 pinning "
    "on every invocation (known CPU 8-11 corruption issue).",
    "Datasets: (1) DIV2K validation set (100 images, 2040x1356, official ETH Zurich CVL "
    "mirror, https://data.vision.ee.ethz.ch/cvl/DIV2K/) -- used for the controlled "
    "resolution x aspect-ratio sweep (images 0-7) and the same-scene 06-E dataset "
    "(images 8-32, disjoint from the sweep). (2) CBSD68 and Rain100L test sets "
    "(read from /home/minura/FYP/Workspace/Himeth/data/, native 481x321 resolution, "
    "a teammate's existing local copy of these standard published benchmarks) -- used "
    "for the native-resolution reference points in the resolution sweep (Phase 1B).",
    "Model: released, UNMODIFIED AdaIR (decoder=True), same checkpoint as TEST01-05.5. "
    "No retraining, no fine-tuning, no weight modification anywhere in TEST06.",
    "Frequency-path intervention: swaps ONLY AFLB3's (raw_high, raw_low) tensors "
    "(the output of the unmodified fft() method), leaving all downstream FMiM/FMoM/"
    "channel_cross_agg architecture and the recipient's spatial branch untouched -- "
    "verified in Phase 0's forward-path audit and confirmed by a self-swap control "
    "producing exactly 0.0 output difference.",
]

out_path = TEST06 / "results" / "environment.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
