"""TEST05.5: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST05_5 = Path(__file__).resolve().parent.parent
REPO = TEST05_5.parent
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
    "TEST05.5 -- Scientific Audit and F2S Hypothesis Validation -- Environment Record",
    "PURPOSE: adversarial re-audit of TEST01-05's conclusions; does NOT train NAFNet, "
    "does NOT modify checkpoint/AdaIR source/any prior test's files (read-only reference "
    "only), all outputs confined to test05_5/.",
    f"AdaIR source git SHA: {git_sha}",
    f"Checkpoint path: {CKPT_PATH}",
    f"Checkpoint SHA256: {h.hexdigest()}",
    "Expected checkpoint SHA256 (from TEST01-05 record): "
    "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb -- "
    "MUST match; any mismatch invalidates cross-experiment comparison.",
    f"Python version: {sys.version}",
    f"PyTorch version: {torch.__version__}",
    f"CUDA version (torch): {torch.version.cuda}",
    f"GPU: {gpu_info}",
    "Host: devon (192.248.10.68) -- KNOWN ISSUE: logical CPUs 8-11 are unreliable; "
    "every script invocation pinned: taskset -c 0-7,12-31 python <script>. Also known: "
    "devon has rebooted unexpectedly during TEST04 and TEST05 (confirmed via journalctl "
    "kernel-version bumps, not crashes); connectivity was intermittent during TEST05.5 setup.",
    "conda env: adair-distill",
    "Datasets used: (1) test03/data/{rain,haze,noise,clean} -- TEST03's exact 100 scenes, "
    "read-only, used for Phase 1-2/7-14; (2) test05_5/data/{rain,haze,noise} -- NEW "
    "parameter-randomized 2-severity-band dataset built in this experiment for Phase 3-4/10, "
    "does not replace or modify test03's dataset.",
    "Model: released, UNMODIFIED AdaIR (decoder=True) for all baseline analysis (Phase "
    "1-6, 10-14); T0-T3 controlled variants (frequency_variants.py, new code in "
    "test05_5/src, NOT modifying test01/model_variants.py or the checkpoint) for the "
    "Phase 7-9 frequency-path ablation.",
    "No retraining, no fine-tuning, no weight modification anywhere in TEST05.5. All PCA/"
    "scaler fits are leakage-safe (fit on training folds only, per GroupKFold(scene_id)).",
]

out_path = TEST05_5 / "results" / "environment.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
