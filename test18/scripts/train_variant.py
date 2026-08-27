"""TEST18: train one AdaIR ablation variant (A_baseline..E_full) on the
real 3-in-1 degradation setting (dehaze OTS + derain Rain100L + denoise
DIV2K-clean-pool w/ online Gaussian noise), matching the released
train.py's optimizer/schedule/loss exactly but as a plain PyTorch loop
(this project's established convention, TEST07-B onward) instead of
Lightning+wandb, with per-epoch checkpointing and CSV logging.

Usage (devon, adair-distill env):
  python train_variant.py --variant A_baseline --epochs 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

TEST18 = Path(__file__).resolve().parent.parent
ADAIR_ROOT = next((p for p in [TEST18.parent.parent / "AdaIR", TEST18.parent / "AdaIR"]
                    if (p / "net" / "model.py").exists()), TEST18.parent.parent / "AdaIR")
sys.path.insert(0, str(ADAIR_ROOT))
sys.path.insert(0, str(TEST18 / "scripts"))

from ablatable_model import build_variant, VARIANTS  # noqa: E402
from utils.dataset_utils import AdaIRTrainDataset  # noqa: E402 (read-only reuse)
from utils.schedulers import LinearWarmupCosineAnnealingLR  # noqa: E402 (read-only reuse)

CKPT_DIR = TEST18 / "results" / "checkpoints"
RESULTS_DIR = TEST18 / "results"


class Opt:
    """Mimics AdaIR/options.py's argparse Namespace for AdaIRTrainDataset,
    pointed at TEST18's own data directories (never touching AdaIR/data_dir
    or fyp-adair-distill's data)."""
    def __init__(self, epochs, batch_size, patch_size, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = 2e-4
        self.de_type = ["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze"]
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.data_file_dir = str(TEST18 / "data_dir") + "/"
        self.denoise_dir = str(TEST18 / "data" / "Train" / "Denoise") + "/"
        self.derain_dir = str(TEST18 / "data" / "Train" / "Derain") + "/"
        self.dehaze_dir = str(TEST18 / "data" / "Train" / "Dehaze") + "/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    opt = Opt(args.epochs, args.batch_size, args.patch_size, args.num_workers)
    train_ds = AdaIRTrainDataset(opt)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                               drop_last=True, num_workers=opt.num_workers, pin_memory=True)

    model = build_variant(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.variant}] {n_params:,} params, {len(train_ds):,} training samples, "
          f"{len(train_loader):,} steps/epoch", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
    loss_fn = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_rows = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t_epoch = time.time()
        for step, (names_deid, degrad_patch, clean_patch) in enumerate(train_loader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                restored = model(degrad_patch)
                loss = loss_fn(restored, clean_patch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if step % 200 == 0:
                print(f"[{args.variant}] epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                      f"L1={np.mean(losses[-200:]):.4f} elapsed={time.time()-t_start:.0f}s", flush=True)
        scheduler.step(epoch - 1)

        mean_loss = float(np.mean(losses))
        row = {"variant": args.variant, "epoch": epoch, "train_l1_loss": mean_loss,
               "epoch_time_s": time.time() - t_epoch, "total_elapsed_s": time.time() - t_start,
               "nan_or_inf": bool(np.isnan(mean_loss) or np.isinf(mean_loss))}
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(RESULTS_DIR / f"epoch_metrics_{args.variant}.csv", index=False)

        torch.save(model.state_dict(), CKPT_DIR / f"model_{args.variant}_epoch{epoch}.pt")
        prev = CKPT_DIR / f"model_{args.variant}_epoch{epoch - 1}.pt"
        if prev.exists() and (epoch - 1) % 5 != 0:  # keep every 5th + latest, save disk
            prev.unlink()

        print(f"[{args.variant}] epoch {epoch}/{args.epochs} DONE mean_L1={mean_loss:.4f} "
              f"epoch_time={row['epoch_time_s']:.0f}s total={row['total_elapsed_s']:.0f}s", flush=True)

    torch.save(model.state_dict(), CKPT_DIR / f"model_{args.variant}_final.pt")
    print(f"[{args.variant}] TRAINING COMPLETE. total_time={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
