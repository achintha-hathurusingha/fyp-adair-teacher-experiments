"""TEST05 Phase 1-2: extract candidate features for all 300 TEST03 images
(100 scenes x Rain/Haze/Noise), released unmodified AdaIR, reusing the
exact same non-intrusive instrumentation (Recorder + attach_instrumentation
+ attach_stage_hooks) as test01-03 -- read-only shared infra, not modified.

Candidate features (per scene x degradation):
  Global: latent_pre (== "latent" stage hook), aflb1/2/3_out
  Per-AFLB (x3): y_in, raw_high, raw_low, mined_high, mined_low,
                 hl_spatial_weight, lh_channel_weight, fmom_agg, cross_agg_out
  Scalars: alpha, beta (per AFLB)

For all 300 images, saves BOTH:
  - pooled GAP+GMP vector (2C-dim, for probe comparability with test02/03)
  - per-channel GAP vector (C-dim, the basis for Phase 6 channel-level analysis)
  - per-channel std vector (C-dim)
into results/feature_analysis/<feature>.npz (X_gap, X_gmp, X_std, degradation, scene_id).

For 15 representative scenes (x3 degradations = 45 images), saves full
float16 raw tensors to results/tensors/ for spatial/frequency analysis
(Phase 11-12) and on-demand causal intervention (Phase 7-10/14, which
re-derives fresh tensors via the model rather than relying on this cache,
but this cache supports fast qualitative/visual inspection).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_features.py
  taskset -c 0-7,12-31 python extract_features.py --limit 9   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST05 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair  # noqa: E402
from stats_utils import psnr_ssim  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST05 / "results" / "manifest.csv"
FEATURES_DIR = TEST05 / "results" / "feature_analysis"
STATS_DIR = TEST05 / "results" / "statistics"
TENSORS_DIR = TEST05 / "results" / "tensors"

AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
AFLB_KEYS = ["y_in", "raw_high", "raw_low", "mined_high", "mined_low",
             "hl_spatial_weight", "lh_channel_weight", "fmom_agg", "cross_agg_out", "aflb_out"]
DEGS = ["Rain", "Haze", "Noise"]
N_REPRESENTATIVE_SCENES = 15


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def channel_stats(t: torch.Tensor):
    """t: (1,C,H,W) or (1,C,1,1). Returns (gap[C], gmp[C], std[C])."""
    x = t.detach().float()
    if x.dim() == 4 and x.shape[-2:] != (1, 1):
        gap = x.mean(dim=(2, 3))[0]
        gmp = x.amax(dim=(2, 3))[0]
        std = x.std(dim=(2, 3), unbiased=False)[0]
    else:
        gap = x.reshape(x.shape[0], -1)[0]
        gmp = gap
        std = torch.zeros_like(gap)
    return gap.cpu().numpy(), gmp.cpu().numpy(), std.cpu().numpy()


def attach_patch_embed_hook(net, recorder: Recorder):
    def _hook(module, inputs, output):
        recorder.put("_stages", "shallow_Y0", output)
    return net.patch_embed.register_forward_hook(_hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))
    if args.limit:
        scene_rows = scene_rows[:max(1, args.limit // 3)]

    representative_scenes = {r["scene_id"] for r in scene_rows[:N_REPRESENTATIVE_SCENES]}
    print(f"{len(scene_rows)} scenes x 3 = {len(scene_rows) * 3} images, "
          f"{len(representative_scenes)} representative scenes (raw tensors saved)", flush=True)

    print(f"loading AdaIR (released, unmodified) ckpt {CKPT_PATH.name}", flush=True)
    model = load_adair(ADAIR_DIR, CKPT_PATH, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)
    attach_patch_embed_hook(net, recorder)
    print(f"checkpoint OK: {n_params:,} params, 0 missing/0 unexpected keys", flush=True)

    pooled: dict[str, list] = {}
    alpha_beta_rows = []
    psnr_ssim_rows = []
    tensor_index_rows = []
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    idx = 0
    for scene_row in scene_rows:
        scene_id = scene_row["scene_id"]
        clean_np = load_rgb(scene_row["clean_image_path"])
        gt_t = to_tensor(clean_np, args.device)
        is_representative = scene_id in representative_scenes

        for deg in DEGS:
            degraded_np = load_rgb(scene_row[f"{deg.lower()}_image_path"])
            degraded_t = to_tensor(degraded_np, args.device)

            recorder.start()
            with torch.no_grad():
                restored_t = model(degraded_t)
            snap = recorder.snapshot_cpu()

            psnr, ssim = psnr_ssim(restored_t, gt_t)
            psnr_ssim_rows.append({"scene_id": scene_id, "degradation": deg, "psnr": psnr, "ssim": ssim})

            feature_tensors: dict[str, torch.Tensor] = {"latent_pre": snap["_stages"]["latent"]}
            for aflb in AFLB_NAMES:
                for key in AFLB_KEYS:
                    feature_tensors[f"{aflb}_{key}"] = snap[aflb][key]
                th = snap[aflb]["threshold_alpha_beta"]
                alpha_beta_rows.append({"scene_id": scene_id, "degradation": deg, "AFLB": aflb,
                                         "alpha": th[0, 0, 0, 0].item(), "beta": th[0, 1, 0, 0].item()})

            for fname, tensor in feature_tensors.items():
                gap, gmp, std = channel_stats(tensor)
                pooled.setdefault(fname, []).append((scene_id, deg, gap, gmp, std))

                if is_representative:
                    out_dir = TENSORS_DIR / fname / deg.lower()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pt_path = out_dir / f"{scene_id}.pt"
                    t16 = tensor.half() if tensor.is_floating_point() else tensor
                    torch.save(t16, pt_path)
                    tensor_index_rows.append({
                        "scene_id": scene_id, "degradation": deg, "feature_name": fname,
                        "shape": "x".join(str(s) for s in tensor.shape), "dtype": "float16",
                        "file_path": str(pt_path.relative_to(TEST05)),
                    })

            idx += 1

        if len(psnr_ssim_rows) % 30 == 0 or scene_id == scene_rows[-1]["scene_id"]:
            print(f"[{idx}/{len(scene_rows) * 3}] {scene_id} elapsed={time.time() - t_start:.0f}s", flush=True)

    import pandas as pd
    pd.DataFrame(psnr_ssim_rows).to_csv(STATS_DIR / "psnr_ssim.csv", index=False)
    pd.DataFrame(alpha_beta_rows).to_csv(STATS_DIR / "alpha_beta.csv", index=False)
    if tensor_index_rows:
        pd.DataFrame(tensor_index_rows).to_csv(TENSORS_DIR / "tensor_index.csv", index=False)

    for fname, entries in pooled.items():
        scene_ids = np.array([e[0] for e in entries])
        degs = np.array([e[1] for e in entries])
        X_gap = np.stack([e[2] for e in entries])
        X_gmp = np.stack([e[3] for e in entries])
        X_std = np.stack([e[4] for e in entries])
        np.savez(FEATURES_DIR / f"{fname}.npz", X_gap=X_gap, X_gmp=X_gmp, X_std=X_std,
                 degradation=degs, scene_id=scene_ids)

    print(f"\nwrote {len(pooled)} candidate-feature .npz files -> {FEATURES_DIR}")
    print(f"wrote raw tensors for {len(representative_scenes)} representative scenes -> {TENSORS_DIR}")
    print(f"total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
