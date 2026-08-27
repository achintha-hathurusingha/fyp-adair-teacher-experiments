"""TEST07-Pilot: shared training loop for Models A-D. Identical training
images, augmentation (none -- fixed crops, documented in
build_pilot_dataset.py), optimizer, LR, batch size, epochs, and seed across
all four models; only the student mechanism differs.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python train_pilot.py --model A
  taskset -c 0-7,12-31 python train_pilot.py --model B
  taskset -c 0-7,12-31 python train_pilot.py --model C
  taskset -c 0-7,12-31 python train_pilot.py --model D
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

TEST07 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST07 / "src"))
from models import MODELS, PCA_DIM  # noqa: E402

MANIFEST_PATH = TEST07 / "results" / "pilot_manifest.csv"
CACHE_DIR = TEST07 / "results" / "teacher_cache"
CKPT_DIR = TEST07 / "results" / "checkpoints"
RESULTS_CSV = TEST07 / "results" / "results.csv"
DEGS = ["Rain", "Haze", "Noise"]

SEED = 0
EPOCHS = 15
BATCH_SIZE = 8
LR = 2e-4
LAMBDA_KD = 0.1


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_rgb_tensor(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


class PilotDataset(Dataset):
    def __init__(self, rows, embeddings_by_key):
        self.samples = []
        for row in rows:
            for deg in DEGS:
                key = (row["scene_id"], deg)
                e_t = embeddings_by_key.get(key)
                self.samples.append({"clean_path": row["clean_path"], "degraded_path": row[f"{deg.lower()}_path"],
                                      "degradation": deg, "scene_id": row["scene_id"], "e_t": e_t})

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
    E, scene_id, degradation = d["E"], d["scene_id"], d["degradation"]
    return {(sid, deg): E[i] for i, (sid, deg) in enumerate(zip(scene_id, degradation))}


def psnr_ssim(pred, target):
    pred_u8 = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    tgt_u8 = (target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    psnr = float(peak_signal_noise_ratio(tgt_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(tgt_u8, pred_u8, data_range=255, channel_axis=2))
    return psnr, ssim


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model, input_shape=(1, 3, 128, 128), device="cuda"):
    """Reuse fyp-adair-distill's own validated MAC counter (read-only import,
    torch.utils.flop_counter based, no external dependency) instead of a
    separate thop-based estimate -- consistent with the team's established
    MAC convention (see complexity.py's docstring: MACs = FLOPs/2, cheap
    elementwise ops like LayerNorm/SimpleGate excluded, matching F1's
    finding that MACs understate normalization-heavy graphs' true cost)."""
    try:
        import sys as _sys
        from pathlib import Path as _Path
        fyp = _Path(__file__).resolve().parent.parent.parent.parent / "fyp-adair-distill"
        _sys.path.insert(0, str(fyp))
        from src.models.complexity import count_macs
        model_cpu = model.to("cpu")

        class _OutputOnly(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out, _ = self.inner(x)
                return out

        macs = count_macs(_OutputOnly(model_cpu), input_shape)
        model.to(device)
        return int(macs)
    except Exception as e:
        print(f"MACs estimation unavailable ({e}); reporting N/A", flush=True)
        model.to(device)
        return None


def validate(model, val_loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for degraded, clean, e_t, deg, scene_id in val_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            out, e_s = model(degraded)
            for i in range(degraded.shape[0]):
                psnr, ssim = psnr_ssim(out[i], clean[i])
                rows.append({"degradation": deg[i], "scene_id": scene_id[i], "psnr": psnr, "ssim": ssim})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda"

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    embeddings_by_key = load_embeddings()
    train_ds = PilotDataset(train_rows, embeddings_by_key)
    val_ds = PilotDataset(val_rows, embeddings_by_key)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                               generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MODELS[args.model]().to(device)
    n_params = count_params(model)
    macs = estimate_macs(model, device=device)
    print(f"Model {args.model}: {n_params:,} params, MACs={macs}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    use_kd = args.model in ("B", "C", "D")

    epoch_rows = []
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses, kd_losses = [], []
        for degraded, clean, e_t, deg, scene_id in train_loader:
            degraded, clean, e_t = degraded.to(device), clean.to(device), e_t.to(device)
            optimizer.zero_grad()
            out, e_s = model(degraded)
            l_restore = F.l1_loss(out, clean)
            loss = l_restore
            kd_val = 0.0
            if use_kd:
                l_kd = F.mse_loss(e_s, e_t)
                loss = l_restore + LAMBDA_KD * l_kd
                kd_val = float(l_kd.item())
            loss.backward()
            optimizer.step()
            train_losses.append(float(l_restore.item()))
            kd_losses.append(kd_val)

        val_df = validate(model, val_loader, device)
        overall_psnr, overall_ssim = val_df.psnr.mean(), val_df.ssim.mean()
        per_deg = val_df.groupby("degradation")[["psnr", "ssim"]].mean()
        row = {
            "model": args.model, "epoch": epoch,
            "train_l1_loss": float(np.mean(train_losses)), "train_kd_loss": float(np.mean(kd_losses)),
            "val_psnr": float(overall_psnr), "val_ssim": float(overall_ssim),
            "rain_psnr": float(per_deg.loc["Rain", "psnr"]) if "Rain" in per_deg.index else None,
            "haze_psnr": float(per_deg.loc["Haze", "psnr"]) if "Haze" in per_deg.index else None,
            "noise_psnr": float(per_deg.loc["Noise", "psnr"]) if "Noise" in per_deg.index else None,
            "rain_ssim": float(per_deg.loc["Rain", "ssim"]) if "Rain" in per_deg.index else None,
            "haze_ssim": float(per_deg.loc["Haze", "ssim"]) if "Haze" in per_deg.index else None,
            "noise_ssim": float(per_deg.loc["Noise", "ssim"]) if "Noise" in per_deg.index else None,
            "params": n_params, "macs": macs,
        }
        epoch_rows.append(row)
        print(f"[{args.model}] epoch {epoch}/{EPOCHS}: L1={row['train_l1_loss']:.4f} "
              f"KD={row['train_kd_loss']:.4f} val_psnr={row['val_psnr']:.3f} val_ssim={row['val_ssim']:.4f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"model_{args.model}.pt")

    epoch_df = pd.DataFrame(epoch_rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[existing.model != args.model]  # replace this model's rows if re-run
        epoch_df = pd.concat([existing, epoch_df], ignore_index=True)
    epoch_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV} (model {args.model} added/updated)")


if __name__ == "__main__":
    main()
