"""Phase 3/4/5: run the released, UNMODIFIED AdaIR model over all 300 images,
extract every intermediate representation listed in the TEST02 spec (A-R),
compute a compact GAP+GMP pooled vector + scalar statistics for every one,
and save:

  - results/statistics/feature_statistics.csv   (long format, ALL 300 images x ALL features)
  - results/features/<feature_name>.npz         (pooled vectors, X (300,dim) + y + image_ids -- for classifiers)
  - results/tensors/<feature>/<degradation>/<image_id>.pt   (raw tensors, 15 representative images only)
  - results/tensors/tensor_index.csv
  - results/statistics/psnr_ssim.csv            (per-image PSNR/SSIM, for Phase 14)

Model is used EXACTLY as released: no retraining, no architecture change, no
preprocessing change. The only addition is non-intrusive forward hooks /
monkey-patched capture points (identical mechanism to test01's
instrument.py, reused here unmodified).

Usage (on devon, adair-distill env, PINNED -- flaky cores 8-11):
  taskset -c 0-7,12-31 python extract_features.py
  taskset -c 0-7,12-31 python extract_features.py --limit 9   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TEST02 = Path(__file__).resolve().parent.parent
REPO = TEST02.parent
sys.path.insert(0, str(REPO / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair, TRANSFORMER_STAGES  # noqa: E402
from run_inference import crop_img, load_rgb, add_gaussian_noise, to_tensor  # noqa: E402
from stats_utils import psnr_ssim  # noqa: E402

ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST02 / "results" / "dataset_manifest.csv"

STATS_DIR = TEST02 / "results" / "statistics"
FEATURES_DIR = TEST02 / "results" / "features"
TENSORS_DIR = TEST02 / "results" / "tensors"
N_REPRESENTATIVE_PER_DEG = 5

AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
AFLB_FEATURE_KEYS = ["y_in", "raw_high", "raw_low", "mined_high", "mined_low",
                      "hl_spatial_weight", "lh_channel_weight", "fmom_agg", "cross_agg_out", "aflb_out"]
GLOBAL_STAGE_KEYS = TRANSFORMER_STAGES  # encoder1-3, latent, decoder3-1, refinement

GIT_SHA = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def pick_representative(rows: list[dict]) -> set[str]:
    ids = set()
    for deg in ("Rain", "Haze", "Noise"):
        deg_rows = [r for r in rows if r["degradation"] == deg][:N_REPRESENTATIVE_PER_DEG]
        ids.update(r["image_id"] for r in deg_rows)
    return ids


def gap_gmp(t: torch.Tensor) -> np.ndarray:
    """t: (1,C,H,W) or (1,C,1,1). Returns concat[GAP,GMP] as (2C,) float32 numpy."""
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
        rows = list(csv.DictReader(f))
    if args.limit:
        by_deg: dict[str, list] = {}
        for r in rows:
            by_deg.setdefault(r["degradation"], []).append(r)
        per_deg = max(1, args.limit // len(by_deg))
        rows = [r for rs in by_deg.values() for r in rs[:per_deg]]

    representative_ids = pick_representative(rows)
    print(f"{len(rows)} images, {len(representative_ids)} representative (raw tensors saved): "
          f"{sorted(representative_ids)}", flush=True)

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

    # accumulate pooled vectors per feature name -> list of (image_id, degradation, vector)
    pooled: dict[str, list] = {}
    stat_rows = []
    psnr_ssim_rows = []
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    tensor_index_rows = []

    t_start = time.time()
    for idx, row in enumerate(rows):
        image_id, deg = row["image_id"], row["degradation"]
        gt_np = crop_img(load_rgb(row["ground_truth_path"]))
        if deg == "Noise":
            sigma = float(row["noise_sigma"])
            rng = np.random.RandomState(abs(hash(image_id)) % (2 ** 31))
            degraded_np = add_gaussian_noise(gt_np, sigma, rng=rng)
        else:
            degraded_np = crop_img(load_rgb(row["input_path"]))

        degraded_t = to_tensor(degraded_np, args.device)
        gt_t = to_tensor(gt_np, args.device)

        recorder.start()
        with torch.no_grad():
            restored_t = model(degraded_t)
        snap = recorder.snapshot_cpu()

        psnr, ssim = psnr_ssim(restored_t, gt_t)
        psnr_ssim_rows.append({"image_id": image_id, "degradation": deg, "psnr": psnr, "ssim": ssim})

        is_representative = image_id in representative_ids

        feature_tensors: dict[str, torch.Tensor] = {"input": degraded_t.cpu(), "output": restored_t.cpu()}
        for key in GLOBAL_STAGE_KEYS:
            feature_tensors[key] = snap["_stages"][key]
        feature_tensors["shallow_Y0"] = snap["_stages"]["shallow_Y0"]
        for aflb in AFLB_NAMES:
            for key in AFLB_FEATURE_KEYS:
                feature_tensors[f"{aflb}_{key}"] = snap[aflb][key]

        for fname, tensor in feature_tensors.items():
            vec = gap_gmp(tensor)
            pooled.setdefault(fname, []).append((image_id, deg, vec))
            s = tensor_stats_row(tensor)
            module_name = fname  # source module documented in report; CSV keeps the exact key used to extract
            stat_rows.append({"feature_name": fname, "module_name": module_name,
                               "image_id": image_id, "degradation": deg, **s})

            if is_representative:
                out_dir = TENSORS_DIR / fname / deg.lower()
                out_dir.mkdir(parents=True, exist_ok=True)
                pt_path = out_dir / f"{image_id}.pt"
                t16 = tensor.half() if tensor.is_floating_point() else tensor
                torch.save(t16, pt_path)
                tensor_index_rows.append({
                    "image_id": image_id, "degradation": deg, "feature_name": fname,
                    "source_module": module_name, "shape": s["shape"], "dtype": "float16",
                    "file_path": str(pt_path.relative_to(TEST02)),
                })

        if (idx + 1) % 25 == 0 or idx == len(rows) - 1:
            print(f"[{idx + 1}/{len(rows)}] {image_id} ({deg}) psnr={psnr:.2f} ssim={ssim:.4f} "
                  f"elapsed={time.time() - t_start:.0f}s", flush=True)

    # write feature_statistics.csv
    import pandas as pd
    pd.DataFrame(stat_rows).to_csv(STATS_DIR / "feature_statistics.csv", index=False)
    pd.DataFrame(psnr_ssim_rows).to_csv(STATS_DIR / "psnr_ssim.csv", index=False)
    if tensor_index_rows:
        pd.DataFrame(tensor_index_rows).to_csv(TENSORS_DIR / "tensor_index.csv", index=False)

    # write pooled vectors per feature
    for fname, entries in pooled.items():
        image_ids = np.array([e[0] for e in entries])
        degs = np.array([e[1] for e in entries])
        X = np.stack([e[2] for e in entries])
        np.savez(FEATURES_DIR / f"{fname}.npz", X=X, degradation=degs, image_id=image_ids)

    print(f"\nwrote {STATS_DIR / 'feature_statistics.csv'} ({len(stat_rows)} rows)")
    print(f"wrote {len(pooled)} pooled-feature .npz files -> {FEATURES_DIR}")
    print(f"wrote raw tensors for {len(representative_ids)} representative images -> {TENSORS_DIR}")
    print(f"total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
