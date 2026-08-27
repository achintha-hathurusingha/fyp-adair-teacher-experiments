"""TEST03 Phase 5-7/17: run the released, UNMODIFIED AdaIR over all 300
same-scene degraded images (100 scenes x Rain/Haze/Noise), extract the same
41 intermediate representations as TEST02, pool via GAP+GMP, and record
PSNR/SSIM against the ORIGINAL CLEAN scene image (not the synthetic
degraded image) per Phase 17.

Saves:
  results/statistics/feature_statistics.csv
  results/features/<feature_name>.npz   (X, degradation, scene_id)
  results/tensors/<feature>/<degradation>/<scene_id>.pt   (10 scenes x 3 = 30 images)
  results/tensors/tensor_index.csv
  results/statistics/restoration_metrics.csv   (Phase 17: PSNR/SSIM/latency/peak-mem vs CLEAN)

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_features.py
  taskset -c 0-7,12-31 python extract_features.py --limit 9   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST03 = Path(__file__).resolve().parent.parent
REPO = TEST03.parent
sys.path.insert(0, str(REPO / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair, TRANSFORMER_STAGES  # noqa: E402
from stats_utils import psnr_ssim  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST03 / "results" / "manifest" / "scene_manifest.csv"

STATS_DIR = TEST03 / "results" / "statistics"
FEATURES_DIR = TEST03 / "results" / "features"
TENSORS_DIR = TEST03 / "results" / "tensors"
N_REPRESENTATIVE_SCENES = 10

AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
AFLB_FEATURE_KEYS = ["y_in", "raw_high", "raw_low", "mined_high", "mined_low",
                      "hl_spatial_weight", "lh_channel_weight", "fmom_agg", "cross_agg_out", "aflb_out"]
GLOBAL_STAGE_KEYS = TRANSFORMER_STAGES
DEGS = ["Rain", "Haze", "Noise"]

GIT_SHA = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def gap_gmp(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    if x.dim() == 4 and x.shape[-2:] != (1, 1):
        gap = x.mean(dim=(2, 3))[0]
        gmp = x.amax(dim=(2, 3))[0]
    else:
        gap = x.reshape(x.shape[0], -1)[0]
        gmp = gap
    return torch.cat([gap, gmp]).numpy()


def tensor_stats_row(t: torch.Tensor) -> dict:
    x = t.detach().float()
    l2 = torch.linalg.vector_norm(x.reshape(-1)).item()
    return {
        "shape": "x".join(str(s) for s in x.shape), "channel_count": x.shape[1] if x.dim() >= 2 else 1,
        "height": x.shape[2] if x.dim() == 4 else "", "width": x.shape[3] if x.dim() == 4 else "",
        "mean": x.mean().item(), "std": x.std(unbiased=False).item(),
        "min": x.min().item(), "max": x.max().item(),
        "L1": x.abs().sum().item(), "L2": l2, "energy": l2 ** 2,
    }


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
    print(f"{len(scene_rows)} scenes x 3 degradations = {len(scene_rows) * 3} images, "
          f"{len(representative_scenes)} representative scenes (raw tensors saved for all 3 variants each)",
          flush=True)

    print(f"loading AdaIR (released, unmodified) from {ADAIR_DIR} (git {GIT_SHA[:8]}), "
          f"ckpt {CKPT_PATH.name}", flush=True)
    model = load_adair(ADAIR_DIR, CKPT_PATH, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)
    attach_patch_embed_hook(net, recorder)
    print(f"checkpoint OK: {n_params:,} params, 0 missing/0 unexpected keys", flush=True)

    pooled: dict[str, list] = {}
    stat_rows = []
    restoration_rows = []
    tensor_index_rows = []
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)

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
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                restored_t = model(degraded_t)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000
            peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
            snap = recorder.snapshot_cpu()

            psnr, ssim = psnr_ssim(restored_t, gt_t)
            restoration_rows.append({"scene_id": scene_id, "degradation": deg,
                                      "psnr": psnr, "ssim": ssim,
                                      "inference_time_ms": latency_ms, "peak_memory_mb": peak_mem_mb})

            feature_tensors: dict[str, torch.Tensor] = {"input": degraded_t.cpu(), "output": restored_t.cpu()}
            for key in GLOBAL_STAGE_KEYS:
                feature_tensors[key] = snap["_stages"][key]
            feature_tensors["shallow_Y0"] = snap["_stages"]["shallow_Y0"]
            for aflb in AFLB_NAMES:
                for key in AFLB_FEATURE_KEYS:
                    feature_tensors[f"{aflb}_{key}"] = snap[aflb][key]

            for fname, tensor in feature_tensors.items():
                vec = gap_gmp(tensor)
                pooled.setdefault(fname, []).append((scene_id, deg, vec))
                s = tensor_stats_row(tensor)
                stat_rows.append({"feature_name": fname, "module_name": fname,
                                   "scene_id": scene_id, "degradation": deg, **s})

                if is_representative:
                    out_dir = TENSORS_DIR / fname / deg.lower()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pt_path = out_dir / f"{scene_id}.pt"
                    t16 = tensor.half() if tensor.is_floating_point() else tensor
                    torch.save(t16, pt_path)
                    tensor_index_rows.append({
                        "scene_id": scene_id, "degradation": deg, "feature_name": fname,
                        "source_module": fname, "shape": s["shape"], "dtype": "float16",
                        "file_path": str(pt_path.relative_to(TEST03)),
                    })

            idx += 1
            if idx % 30 == 0 or idx == len(scene_rows) * 3:
                print(f"[{idx}/{len(scene_rows) * 3}] {scene_id} ({deg}) psnr={psnr:.2f} ssim={ssim:.4f} "
                      f"elapsed={time.time() - t_start:.0f}s", flush=True)

    import pandas as pd
    pd.DataFrame(stat_rows).to_csv(STATS_DIR / "feature_statistics.csv", index=False)
    pd.DataFrame(restoration_rows).to_csv(STATS_DIR / "restoration_metrics.csv", index=False)
    if tensor_index_rows:
        pd.DataFrame(tensor_index_rows).to_csv(TENSORS_DIR / "tensor_index.csv", index=False)

    for fname, entries in pooled.items():
        scene_ids = np.array([e[0] for e in entries])
        degs = np.array([e[1] for e in entries])
        X = np.stack([e[2] for e in entries])
        np.savez(FEATURES_DIR / f"{fname}.npz", X=X, degradation=degs, scene_id=scene_ids)

    print(f"\nwrote {STATS_DIR / 'feature_statistics.csv'} ({len(stat_rows)} rows)")
    print(f"wrote {STATS_DIR / 'restoration_metrics.csv'} ({len(restoration_rows)} rows)")
    print(f"wrote {len(pooled)} pooled-feature .npz files -> {FEATURES_DIR}")
    print(f"wrote raw tensors for {len(representative_scenes)} representative scenes x 3 degradations "
          f"-> {TENSORS_DIR}")
    print(f"total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
