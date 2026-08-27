"""TEST10-R: training loop for Models A/F/G x seeds {0,1,2} (9 runs).
Teacher trajectory targets are FIXED and precomputed (build_teacher_targets.py)
-- no online teacher forward pass needed during training (a correction and a
speed/memory simplification over TEST10).

MANDATORY collapse monitoring (Phase 6): every epoch, for Model G, computes
cross-input diversity of the student's OWN stage embeddings on the
validation set (mean pairwise cosine, mean variance). If collapse is
detected (mean pairwise cosine > 0.98) at epoch >= 10, the run is ABORTED
immediately rather than continuing to 50 epochs and analyzing a broken
model, per the task's explicit instruction.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 env OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python train.py --model A --seed 0
  ... (9 total: A/F/G x 0/1/2)
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

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
sys.path.insert(0, str(TEST10R / "src"))
from models import MODELS, PCA_DIM, TRAJ_DIM  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
KD_CACHE_DIR = TEST07B_RESULTS / "teacher_cache"          # final 16-dim KD embedding (reused read-only)
TRAJ_CACHE_DIR = TEST10R / "results" / "teacher_cache"     # this experiment's own fixed 3-stage targets

CKPT_DIR = TEST10R / "results" / "checkpoints"
RESULTS_CSV = TEST10R / "results" / "epoch_metrics.csv"
SEED_SUMMARY_CSV = TEST10R / "results" / "seed_summary.csv"
COLLAPSE_MONITOR_CSV = TEST10R / "results" / "collapse_monitor.csv"
DEGS = ["Rain", "Haze", "Noise"]

EPOCHS = 50
BATCH_SIZE = 8
LR = 2e-4
LAMBDA_KD = 0.1
LAMBDA_TRAJ = 0.1
STAGE_WEIGHTS = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
COLLAPSE_COSINE_THRESHOLD = 0.98
COLLAPSE_CHECK_FROM_EPOCH = 10


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_rgb_tensor(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


class PilotDataset(Dataset):
    def __init__(self, rows, kd_embeddings, traj_embeddings):
        self.samples = []
        for row in rows:
            for deg in DEGS:
                key = (row["crop_id"], deg)
                e_kd = kd_embeddings.get(key)
                e_traj = traj_embeddings.get(key)
                self.samples.append({"clean_path": row["clean_path"], "degraded_path": row[f"{deg.lower()}_path"],
                                      "degradation": deg, "scene_id": row["scene_id"], "crop_id": row["crop_id"],
                                      "e_kd": e_kd, "e_traj": e_traj})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        degraded = load_rgb_tensor(s["degraded_path"])
        clean = load_rgb_tensor(s["clean_path"])
        e_kd = torch.from_numpy(s["e_kd"]).float() if s["e_kd"] is not None else torch.zeros(PCA_DIM)
        if s["e_traj"] is not None:
            e_traj = torch.stack([torch.from_numpy(s["e_traj"][i]).float() for i in range(3)])
        else:
            e_traj = torch.zeros(3, TRAJ_DIM)
        return degraded, clean, e_kd, e_traj, s["degradation"], s["scene_id"]


def load_kd_embeddings():
    d = np.load(KD_CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    E, crop_id, degradation = d["E"], d["crop_id"], d["degradation"]
    return {(cid, deg): E[i] for i, (cid, deg) in enumerate(zip(crop_id, degradation))}


def load_traj_embeddings():
    d = np.load(TRAJ_CACHE_DIR / "trajectory_targets.npz", allow_pickle=True)
    crop_id, degradation = d["crop_id"], d["degradation"]
    E0, E1, E2 = d["E_stage0"], d["E_stage1"], d["E_stage2"]
    return {(cid, deg): [E0[i], E1[i], E2[i]] for i, (cid, deg) in enumerate(zip(crop_id, degradation))}


def psnr_ssim(pred, target):
    pred_u8 = (pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    tgt_u8 = (target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    psnr = float(peak_signal_noise_ratio(tgt_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(tgt_u8, pred_u8, data_range=255, channel_axis=2))
    return psnr, ssim


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model, input_shape=(1, 3, 128, 128), device="cuda"):
    try:
        import sys as _sys
        from pathlib import Path as _Path
        fyp = _Path(__file__).resolve().parent.parent.parent.parent / "fyp-adair-distill"
        _sys.path.insert(0, str(fyp))
        from src.models.complexity import count_macs

        class _OutputOnly(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out, _ = self.inner(x)
                return out

        model_cpu = model.to("cpu")
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
        for degraded, clean, e_kd, e_traj, deg, scene_id in val_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            out, e_s = model(degraded)
            for i in range(degraded.shape[0]):
                psnr, ssim = psnr_ssim(out[i], clean[i])
                rows.append({"degradation": deg[i], "scene_id": scene_id[i], "psnr": psnr, "ssim": ssim})
    return pd.DataFrame(rows)


def collapse_check(model, val_loader, device, rng):
    """Cross-input diversity check on Model G's OWN stage embeddings over
    the validation set (mandatory, Phase 6)."""
    model.eval()
    stage_es = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for degraded, clean, e_kd, e_traj, deg, scene_id in val_loader:
            degraded = degraded.to(device)
            _, _, e_s_traj = model.forward_trajectory(degraded)
            for stage_idx in (0, 1, 2):
                stage_es[stage_idx].append(e_s_traj[stage_idx].cpu().numpy())

    results = {}
    collapsed_any = False
    for stage_idx in (0, 1, 2):
        E = np.concatenate(stage_es[stage_idx], axis=0)
        std = float(E.std(axis=0).mean())
        n = len(E)
        idx_i = rng.randint(0, n, size=min(1000, n * n))
        idx_j = rng.randint(0, n, size=min(1000, n * n))
        mask = idx_i != idx_j
        a, b = E[idx_i[mask]], E[idx_j[mask]]
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        cos = (a_n * b_n).sum(axis=1)
        dist = np.linalg.norm(a - b, axis=1)
        collapsed = bool(cos.mean() > COLLAPSE_COSINE_THRESHOLD)
        collapsed_any = collapsed_any or collapsed
        results[stage_idx] = {"mean_variance": std, "mean_pairwise_cosine": float(cos.mean()),
                               "mean_pairwise_distance": float(dist.mean()), "collapsed": collapsed}
    return results, collapsed_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    rng = np.random.RandomState(args.seed)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    kd_embeddings = load_kd_embeddings()
    traj_embeddings = load_traj_embeddings() if args.model == "G" else {}
    train_ds = PilotDataset(train_rows, kd_embeddings, traj_embeddings)
    val_ds = PilotDataset(val_rows, kd_embeddings, traj_embeddings)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                               generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MODELS[args.model]().to(device)
    n_params = count_params(model)
    macs = estimate_macs(model, device=device)

    use_kd = args.model in ("F", "G")
    use_traj = args.model == "G"

    print(f"Model {args.model} seed {args.seed}: {n_params:,} params, MACs={macs}, "
          f"train_examples={len(train_ds)}, val_examples={len(val_ds)}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_rows = []
    collapse_rows = []
    aborted = False
    abort_epoch = None
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses, kd_losses, traj_losses = [], [], []
        stage_losses = {0: [], 1: [], 2: []}
        for degraded, clean, e_kd, e_traj, deg, scene_id in train_loader:
            degraded, clean, e_kd, e_traj = degraded.to(device), clean.to(device), e_kd.to(device), e_traj.to(device)
            optimizer.zero_grad()

            if use_traj:
                out, e_s, e_s_traj = model.forward_trajectory(degraded)
            else:
                out, e_s = model(degraded)

            l_restore = F.l1_loss(out, clean)
            loss = l_restore
            kd_val, traj_val = 0.0, 0.0

            if use_kd:
                l_kd = F.mse_loss(e_s, e_kd)
                loss = loss + LAMBDA_KD * l_kd
                kd_val = float(l_kd.item())

            if use_traj:
                l_traj = 0.0
                for stage_idx in (0, 1, 2):
                    z_s = F.normalize(e_s_traj[stage_idx], dim=1)
                    z_t = F.normalize(e_traj[:, stage_idx, :], dim=1)
                    sl = F.mse_loss(z_s, z_t)
                    stage_losses[stage_idx].append(float(sl.item()))
                    l_traj = l_traj + STAGE_WEIGHTS[stage_idx] * sl
                loss = loss + LAMBDA_TRAJ * l_traj
                traj_val = float(l_traj.item())

            loss.backward()
            optimizer.step()
            train_losses.append(float(l_restore.item()))
            kd_losses.append(kd_val)
            traj_losses.append(traj_val)

        val_df = validate(model, val_loader, device)
        overall_psnr, overall_ssim = val_df.psnr.mean(), val_df.ssim.mean()
        per_deg = val_df.groupby("degradation")[["psnr", "ssim"]].mean()
        row = {
            "model": args.model, "seed": args.seed, "epoch": epoch,
            "train_l1_loss": float(np.mean(train_losses)), "train_kd_loss": float(np.mean(kd_losses)),
            "train_traj_loss": float(np.mean(traj_losses)),
            "train_stage0_loss": float(np.mean(stage_losses[0])) if stage_losses[0] else None,
            "train_stage1_loss": float(np.mean(stage_losses[1])) if stage_losses[1] else None,
            "train_stage2_loss": float(np.mean(stage_losses[2])) if stage_losses[2] else None,
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

        if use_traj:
            cresults, collapsed_any = collapse_check(model, val_loader, device, rng)
            for stage_idx, r in cresults.items():
                collapse_rows.append({"model": args.model, "seed": args.seed, "epoch": epoch, "stage": stage_idx,
                                       **r})
            if collapsed_any and epoch >= COLLAPSE_CHECK_FROM_EPOCH:
                print(f"[{args.model}/s{args.seed}] COLLAPSE DETECTED at epoch {epoch}: {cresults}", flush=True)
                aborted = True
                abort_epoch = epoch
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{args.model}/s{args.seed}] epoch {epoch}/{EPOCHS}: L1={row['train_l1_loss']:.4f} "
                  f"KD={row['train_kd_loss']:.4f} traj={row['train_traj_loss']:.4f} "
                  f"val_psnr={row['val_psnr']:.3f} val_ssim={row['val_ssim']:.4f} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"model_{args.model}_seed{args.seed}.pt")

    this_run_df = pd.DataFrame(epoch_rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    per_run_csv = RESULTS_CSV.parent / f"epoch_metrics_{args.model}_seed{args.seed}.csv"
    this_run_df.to_csv(per_run_csv, index=False)

    if collapse_rows:
        collapse_df = pd.DataFrame(collapse_rows)
        per_run_collapse_csv = COLLAPSE_MONITOR_CSV.parent / f"collapse_monitor_{args.model}_seed{args.seed}.csv"
        collapse_df.to_csv(per_run_collapse_csv, index=False)

    n_completed = len(this_run_df)
    best_psnr_row = this_run_df.loc[this_run_df.val_psnr.idxmax()]
    best_ssim_row = this_run_df.loc[this_run_df.val_ssim.idxmax()]
    final_row = this_run_df.iloc[-1]
    last5 = this_run_df.iloc[max(0, n_completed - 5):]
    summary_row = {
        "model": args.model, "seed": args.seed, "aborted": aborted, "abort_epoch": abort_epoch,
        "n_epochs_completed": n_completed,
        "best_psnr_epoch": int(best_psnr_row.epoch), "best_psnr": float(best_psnr_row.val_psnr),
        "best_ssim_epoch": int(best_ssim_row.epoch), "best_ssim": float(best_ssim_row.val_ssim),
        "final_epoch": int(final_row.epoch), "final_psnr": float(final_row.val_psnr),
        "final_ssim": float(final_row.val_ssim),
        "last5_mean_psnr": float(last5.val_psnr.mean()), "last5_mean_ssim": float(last5.val_ssim.mean()),
        "last5_mean_rain_psnr": float(last5.rain_psnr.mean()), "last5_mean_haze_psnr": float(last5.haze_psnr.mean()),
        "last5_mean_noise_psnr": float(last5.noise_psnr.mean()),
        "last5_mean_rain_ssim": float(last5.rain_ssim.mean()), "last5_mean_haze_ssim": float(last5.haze_ssim.mean()),
        "last5_mean_noise_ssim": float(last5.noise_ssim.mean()),
        "params": n_params, "macs": macs,
    }
    seed_summary_df = pd.DataFrame([summary_row])
    per_run_summary_csv = SEED_SUMMARY_CSV.parent / f"seed_summary_{args.model}_seed{args.seed}.csv"
    seed_summary_df.to_csv(per_run_summary_csv, index=False)
    status = "ABORTED (collapse)" if aborted else "DONE"
    print(f"\n[{args.model}/s{args.seed}] {status}. last5_mean_psnr={summary_row['last5_mean_psnr']:.3f} "
          f"last5_mean_ssim={summary_row['last5_mean_ssim']:.4f} best_psnr={summary_row['best_psnr']:.3f} "
          f"(epoch {summary_row['best_psnr_epoch']})")


if __name__ == "__main__":
    main()
