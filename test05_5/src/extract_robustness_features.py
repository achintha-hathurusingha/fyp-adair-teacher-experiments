"""TEST05.5 Phase 3-4: run released AdaIR over the parameter-randomized
robustness dataset (200 images: 100 scenes x 2 severity bands, x3
degradations = 600 total images), extract input/latent_pre/AFLB1-3
pooled features, then:
  Phase 3: probe degradation FAMILY (Rain/Haze/Noise) using ALL bands
           pooled together, grouped CV by scene_id
  Phase 4: train on band A only, test on band B only (and vice versa) --
           severity generalization

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_robustness_features.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST05_5 / "results" / "robustness" / "robustness_manifest.csv"
OUT_DIR = TEST05_5 / "results" / "robustness"
DEGS = ["Rain", "Haze", "Noise"]
BANDS = ["A", "B"]
CANDIDATE_KEYS = ["input", "latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def pooled_vec(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    gap = x.mean(dim=(2, 3))[0]
    gmp = x.amax(dim=(2, 3))[0]
    return torch.cat([gap, gmp]).cpu().numpy()


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)
    print(f"checkpoint OK: {n_params:,} params", flush=True)

    pooled: dict[str, list] = {k: [] for k in CANDIDATE_KEYS}
    t_start = time.time()
    idx = 0
    for scene_row in scene_rows:
        scene_id = scene_row["scene_id"]
        for deg in DEGS:
            for band in BANDS:
                path = scene_row[f"{deg.lower()}_band{band}_path"]
                img_t = to_tensor(load_rgb(path), device)

                recorder.start()
                with torch.no_grad():
                    _ = model(img_t)
                snap = recorder.snapshot_cpu()

                tensors = {"input": img_t, "latent_pre": snap["_stages"]["latent"],
                           "AFLB1_aflb_out": snap["AFLB1"]["aflb_out"],
                           "AFLB2_aflb_out": snap["AFLB2"]["aflb_out"],
                           "AFLB3_aflb_out": snap["AFLB3"]["aflb_out"]}
                for key, tensor in tensors.items():
                    pooled[key].append((scene_id, deg, band, pooled_vec(tensor)))
                idx += 1

        if idx % 60 == 0:
            print(f"[{idx}/{len(scene_rows) * 6}] {scene_id} elapsed={time.time() - t_start:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, entries in pooled.items():
        scene_ids = np.array([e[0] for e in entries])
        degs = np.array([e[1] for e in entries])
        bands = np.array([e[2] for e in entries])
        X = np.stack([e[3] for e in entries])
        np.savez(OUT_DIR / f"{key}.npz", X=X, degradation=degs, band=bands, scene_id=scene_ids)
        print(f"saved {key}.npz: {X.shape}", flush=True)

    print(f"total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
