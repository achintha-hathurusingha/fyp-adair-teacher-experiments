"""TEST03 Phase 21: record environment/reproducibility info."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch

TEST03 = Path(__file__).resolve().parent.parent
REPO = TEST03.parent
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
    "TEST03 -- Controlled Same-Scene Degradation Representation Study -- Environment Record",
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
    "Clean image source: Rain100L test-split ground-truth images (norain-*.png), "
    "100 images -- read-only reference, only ever used elsewhere in this project "
    "as derain targets, never before as AdaIR *inputs*.",
    "Degradation synthesis: test03/src/degradation_synthesis.py "
    "(rain: synthetic streak layer via cv2.line + gaussian blur, deterministic per "
    "scene_id seed; haze: atmospheric scattering model I=I*t+A*(1-t), "
    "t=exp(-beta*synthetic_vertical_gradient_depth_proxy), A=0.85, beta=1.2, "
    "deterministic; noise: additive Gaussian, sigma=25, per-scene seeded).",
    "Random seeds: global seed 0 (np.random.seed, torch.manual_seed) + per-scene "
    "deterministic degradation-synthesis seed (np.random.RandomState(abs(hash(scene_id)) % 2**31))",
    "Preprocessing: crop_img(base=16) (same convention as test01/test02), no resize, no padding",
    "Dataset: 100 clean scenes x 3 synthesized degradations = 300 images "
    "(test03/results/manifest/scene_manifest.csv)",
    "Model: released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt, 28,784,824 params, "
    "0 missing / 0 unexpected keys -- IDENTICAL checkpoint/loader to test01/test02",
    "Cross-validation: GroupKFold (group=scene_id), 5 folds -- no fold ever trains and "
    "tests on different degraded versions of the same scene (leakage assertion enforced "
    "in code, see linear_probe_grouped.py)",
]

out_path = TEST03 / "results" / "environment.txt"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
for line in lines:
    print(" ", line)
