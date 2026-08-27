"""TEST06-R Phase 13: absolute-difference heatmaps for 3 representative
scenes, one donor pairing each. Visualization only -- not used as evidence
on its own, per the task's explicit instruction; the quantitative claim
comes from propagation_compact_stats.csv (Phase 10-11).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python propagation_heatmaps.py
"""
from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TEST06_R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST06_R.parent
TEST06 = TEACHER_EXP / "test06"
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
from instrument import Recorder, load_adair  # noqa: E402
from model_variants import _fft_released  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST06 / "results" / "frequency_intervention" / "scene_manifest.csv"
OUT_DIR = TEST06_R / "results" / "visualizations" / "propagation_heatmaps"
DEGS = ["Rain", "Haze", "Noise"]
N_SCENES = 3


def _forward_recording_all_stages(self, x, y, recorder, aflb_name, override_high, override_low):
    _, _, H, W = y.size()
    x_r = F.interpolate(x, (H, W), mode="bilinear")
    real_high, real_low = _fft_released(self, x_r, recorder, aflb_name)
    use_high = override_high if override_high is not None else real_high
    use_low = override_low if override_low is not None else real_low
    recorder.put(aflb_name, "raw_high", use_high)
    high_feature = self.channel_cross_l(use_high, y)
    low_feature = self.channel_cross_h(use_low, y)
    recorder.put(aflb_name, "mined_high", high_feature)
    recorder.put(aflb_name, "mined_low", low_feature)
    agg = self.frequency_refine(low_feature, high_feature)
    recorder.put(aflb_name, "agg", agg)
    out = self.channel_cross_agg(y, agg)
    aflb_out = out * self.para1 + y * self.para2
    return aflb_out


def to_tensor(img_u8, device):
    return torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def build_model():
    device = "cuda"
    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    net = model.net if hasattr(model, "net") else model
    recorder = Recorder()
    fre3 = net.fre3

    def set_override(high, low):
        fre3.forward = types.MethodType(
            lambda self, x, y, r=recorder, n="AFLB3", h=high, l=low: _forward_recording_all_stages(
                self, x, y, r, n, h, l), fre3)
    set_override(None, None)
    return model, recorder, set_override, device


def run(model, recorder, set_override, img_path, override_high=None, override_low=None, device="cuda"):
    set_override(override_high, override_low)
    img_t = to_tensor(load_rgb(img_path), device)
    recorder.start()
    with torch.no_grad():
        _ = model(img_t)
    snap = recorder.snapshot_cpu()
    set_override(None, None)
    return snap["AFLB3"]


def heatmap_panel(scene_id, recipient, donor, normal, swapped, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    stages = ["raw_high", "mined_high", "mined_low", "agg"]
    for col, stage in enumerate(stages):
        n_img = normal[stage][0].mean(0).numpy()
        s_img = swapped[stage][0].mean(0).numpy()
        diff = np.abs(s_img - n_img)
        axes[0, col].imshow(n_img, cmap="viridis")
        axes[0, col].set_title(f"normal {stage}")
        axes[0, col].axis("off")
        axes[1, col].imshow(diff, cmap="hot")
        axes[1, col].set_title(f"|swapped-normal| {stage}")
        axes[1, col].axis("off")
    fig.suptitle(f"{scene_id}: recipient={recipient}, donor={donor}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    with open(MANIFEST_PATH) as f:
        scenes = list(csv.DictReader(f))[:N_SCENES]

    model, recorder, set_override, device = build_model()
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        for recipient in DEGS:
            for donor in DEGS:
                if donor == recipient:
                    continue
                normal_recipient = run(model, recorder, set_override, scene[f"{recipient.lower()}_path"], device=device)
                donor_snap = run(model, recorder, set_override, scene[f"{donor.lower()}_path"], device=device)
                swapped = run(model, recorder, set_override, scene[f"{recipient.lower()}_path"],
                               override_high=donor_snap["raw_high"].to(device),
                               override_low=donor_snap["raw_low"].to(device), device=device)
                out_path = OUT_DIR / f"{scene['scene_id']}_{recipient}_from_{donor}.png"
                heatmap_panel(scene["scene_id"], recipient, donor, normal_recipient, swapped, out_path)


if __name__ == "__main__":
    main()
