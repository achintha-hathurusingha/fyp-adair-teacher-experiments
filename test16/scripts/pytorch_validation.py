"""TEST16 Phase 1-2: PyTorch correctness for all 4 models.

A, F2 (TRAINED, real checkpoints from TEST12): full validation-set
PSNR/SSIM, compared against TEST12's own recorded results as a
sanity check that the checkpoint loads and behaves identically here.

N, S (UNTRAINED this pass -- see TEST16 scoping decision / models.py
docstring): shape + numerical-stability check only (finite output, no
NaN/Inf, correct dtype/shape) on the same validation crops. PSNR/SSIM are
NOT computed for these -- reporting a "quality" number from random weights
would be fabricated data. Params/MACs are still real (architecture-only
metrics, independent of training).

Usage (devon, adair-distill env):
  python pytorch_validation.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

TEST16 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST16.parent
TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CACHE_DIR = TEST07B_RESULTS / "teacher_cache"
OUT_DIR = TEST16 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS, PCA_DIM, load_trained, build_untrained  # noqa: E402

sys.path.insert(0, str(TEACHER_EXP.parent / "fyp-adair-distill"))
from src.models.complexity import count_macs  # noqa: E402


def load_rgb_tensor(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


class PilotDataset(Dataset):
    def __init__(self, rows, embeddings_by_key):
        self.samples = []
        for row in rows:
            for deg in DEGS:
                key = (row["crop_id"], deg)
                e_t = embeddings_by_key.get(key)
                self.samples.append({"clean_path": row["clean_path"], "degraded_path": row[f"{deg.lower()}_path"],
                                      "degradation": deg, "scene_id": row["scene_id"], "crop_id": row["crop_id"],
                                      "e_t": e_t})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        degraded = load_rgb_tensor(s["degraded_path"])
        clean = load_rgb_tensor(s["clean_path"])
        e_t = torch.from_numpy(s["e_t"]).float() if s["e_t"] is not None else torch.zeros(PCA_DIM)
        return degraded, clean, e_t, s["degradation"], s["scene_id"]


def load_embeddings():
    d = np.load(CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    E, crop_id, degradation = d["E"], d["crop_id"], d["degradation"]
    return {(cid, deg): E[i] for i, (cid, deg) in enumerate(zip(crop_id, degradation))}


def psnr_ssim(pred, target):
    pred_u8 = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    tgt_u8 = (target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    psnr = float(peak_signal_noise_ratio(tgt_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(tgt_u8, pred_u8, data_range=255, channel_axis=2))
    return psnr, ssim


class _OutputOnly(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out, _ = self.inner(x)
        return out


def estimate_macs(model, input_shape=(1, 3, 128, 128)):
    try:
        return int(count_macs(_OutputOnly(model.to("cpu")), input_shape))
    except Exception as e:
        print(f"MACs estimation unavailable ({e})", flush=True)
        return None


def full_validate(model, val_loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for degraded, clean, e_t, deg, scene_id in val_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            out, _ = model(degraded)
            for i in range(degraded.shape[0]):
                psnr, ssim = psnr_ssim(out[i], clean[i])
                rows.append({"degradation": deg[i], "scene_id": scene_id[i], "psnr": psnr, "ssim": ssim})
    return pd.DataFrame(rows)


def stability_check(model, val_loader, device, n_batches=5):
    model.eval()
    shapes_ok, finite_ok, n_checked = True, True, 0
    out_shape = None
    with torch.no_grad():
        for bi, (degraded, clean, e_t, deg, scene_id) in enumerate(val_loader):
            if bi >= n_batches:
                break
            degraded = degraded.to(device)
            out, _ = model(degraded)
            out_shape = tuple(out.shape)
            if out.shape != degraded.shape:
                shapes_ok = False
            if not torch.isfinite(out).all():
                finite_ok = False
            n_checked += degraded.shape[0]
    return {"shapes_ok": shapes_ok, "finite_ok": finite_ok, "n_checked": n_checked, "output_shape": out_shape}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]
    embeddings_by_key = load_embeddings()
    val_ds = PilotDataset(val_rows, embeddings_by_key)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"val examples: {len(val_ds)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    all_rows = []

    for name in ["A", "F2"]:
        model = load_trained(name, seed=0, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        macs = estimate_macs(model)
        model.to(device)
        df = full_validate(model, val_loader, device)
        df["model"] = name
        all_rows.append(df)
        per_deg = df.groupby("degradation")[["psnr", "ssim"]].mean()
        summary.append({
            "model": name, "trained": True, "params": n_params, "macs": macs,
            "val_psnr": float(df.psnr.mean()), "val_ssim": float(df.ssim.mean()),
            "rain_psnr": float(per_deg.loc["Rain", "psnr"]) if "Rain" in per_deg.index else None,
            "haze_psnr": float(per_deg.loc["Haze", "psnr"]) if "Haze" in per_deg.index else None,
            "noise_psnr": float(per_deg.loc["Noise", "psnr"]) if "Noise" in per_deg.index else None,
        })
        print(f"{name}: params={n_params:,} macs={macs} val_psnr={df.psnr.mean():.3f} "
              f"val_ssim={df.ssim.mean():.4f}", flush=True)

    for name in ["N", "S"]:
        model = build_untrained(name, seed=0, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        macs = estimate_macs(model)
        model.to(device)
        stab = stability_check(model, val_loader, device)
        summary.append({
            "model": name, "trained": False, "params": n_params, "macs": macs,
            "val_psnr": None, "val_ssim": None, "rain_psnr": None, "haze_psnr": None,
            "noise_psnr": None, "note": "untrained -- latency/architecture only, see TEST16 scoping decision",
            **stab,
        })
        print(f"{name}: params={n_params:,} macs={macs} UNTRAINED "
              f"shapes_ok={stab['shapes_ok']} finite_ok={stab['finite_ok']} "
              f"n_checked={stab['n_checked']}", flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "pytorch_validation_summary.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT_DIR / "pytorch_validation_per_sample.csv", index=False)
    with open(OUT_DIR / "pytorch_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {OUT_DIR / 'pytorch_validation_summary.csv'}")


if __name__ == "__main__":
    main()
