"""TEST10-R Phases 1-2: build FIXED, leakage-safe teacher trajectory
targets -- the primary correction over TEST10.

Phase 1: reuses TEST10's validated 3 stage locations (read from
test10/report/teacher_stage_audit.md, read-only) -- teacher AFLB1/2/3
aflb_out, matched to student decoders[0]/[1]/[2] by spatial resolution.

Phase 2: for each of the 3 stages, pool (GAP+GMP) the raw teacher tensor,
then fit StandardScaler + PCA(32) on TRAINING crops ONLY (leakage-safe,
same discipline as the validated final-16-dim KD embedding since
TEST07-B). The resulting per-stage transform is FROZEN before student
training and NEVER updated by backprop -- this is the fix for TEST10's
collapse (there, both sides were jointly, freely trained with no anchor).

Precomputes e_T^l for every (crop, degradation) pair and caches it, exactly
like test07_b's final-embedding cache -- so training needs NO online
teacher forward pass at all (a further simplification/speedup over TEST10,
which required loading the 28.8M-param teacher during every training step).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python build_teacher_targets.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

TEST10R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10R.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, load_adair  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
CACHE_DIR = TEST10R / "results" / "teacher_cache"
DEGS = ["Rain", "Haze", "Noise"]
TRAJ_DIM = 32
EXPECTED_SHA256 = "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb"

# student stage_idx -> teacher AFLB name (matched by spatial resolution, per
# test10/report/teacher_stage_audit.md)
STAGE_TO_AFLB = {0: "AFLB1", 1: "AFLB2", 2: "AFLB3"}


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pooled_gap_gmp(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float()
    gap = x.mean(dim=(2, 3))[0]
    gmp = x.amax(dim=(2, 3))[0]
    return torch.cat([gap, gmp]).cpu().numpy()


def checkpoint_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = "cuda"
    sha = checkpoint_sha256(CKPT_PATH)
    assert sha == EXPECTED_SHA256, f"CHECKPOINT SHA MISMATCH: {sha} != {EXPECTED_SHA256}"
    print(f"checkpoint SHA256 verified: {sha}", flush=True)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    print(f"teacher frozen, {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    records = []
    t0 = __import__("time").time()
    for idx, row in enumerate(rows):
        for deg in DEGS:
            img_t = load_rgb(row[f"{deg.lower()}_path"], device)
            recorder.start()
            with torch.no_grad():
                _ = model(img_t)
            rec = {"scene_id": row["scene_id"], "crop_id": row["crop_id"], "split": row["split"],
                   "degradation": deg}
            for stage_idx, aflb_name in STAGE_TO_AFLB.items():
                raw = recorder._store[aflb_name]["aflb_out"]
                rec[f"stage{stage_idx}_vec"] = pooled_gap_gmp(raw)
            records.append(rec)
        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{len(rows)}] elapsed={__import__('time').time()-t0:.0f}s", flush=True)
    print(f"extracted {len(records)} records, elapsed={__import__('time').time()-t0:.0f}s", flush=True)

    is_train = np.array([r["split"] == "train" for r in records])
    crop_id = np.array([r["crop_id"] for r in records])
    scene_id = np.array([r["scene_id"] for r in records])
    deg_arr = np.array([r["degradation"] for r in records])
    split_arr = np.array([r["split"] for r in records])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint_sha256": sha, "n_records_total": len(records),
                "n_train_records": int(is_train.sum()), "n_val_records": int((~is_train).sum()),
                "traj_dim": TRAJ_DIM, "stages": {}}

    E_by_stage = {}
    for stage_idx in STAGE_TO_AFLB:
        X = np.stack([r[f"stage{stage_idx}_vec"] for r in records])
        scaler = StandardScaler().fit(X[is_train])
        X_scaled_train = scaler.transform(X[is_train])
        pca = PCA(n_components=TRAJ_DIM, random_state=0).fit(X_scaled_train)
        explained_var = float(pca.explained_variance_ratio_.sum())
        X_scaled_all = scaler.transform(X)
        E = pca.transform(X_scaled_all)
        E_by_stage[stage_idx] = E

        joblib.dump({"scaler": scaler, "pca": pca}, CACHE_DIR / f"stage{stage_idx}_transform.joblib")
        metadata["stages"][str(stage_idx)] = {
            "aflb_name": STAGE_TO_AFLB[stage_idx], "raw_pooled_dim": int(X.shape[1]),
            "pca_fit_sample_count": int(is_train.sum()),
            "pca_explained_variance_ratio_sum": explained_var,
            "pca_per_component_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
        print(f"stage{stage_idx} ({STAGE_TO_AFLB[stage_idx]}): raw_dim={X.shape[1]}, "
              f"PCA-{TRAJ_DIM} explained_variance={explained_var:.4f} "
              f"(fit on {is_train.sum()} training records)", flush=True)

    np.savez(CACHE_DIR / "trajectory_targets.npz",
              E_stage0=E_by_stage[0], E_stage1=E_by_stage[1], E_stage2=E_by_stage[2],
              crop_id=crop_id, scene_id=scene_id, degradation=deg_arr, split=split_arr)
    with open(CACHE_DIR / "trajectory_targets_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nwrote {CACHE_DIR / 'trajectory_targets.npz'} and metadata")


if __name__ == "__main__":
    main()
