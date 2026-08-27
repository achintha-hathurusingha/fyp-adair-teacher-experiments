"""TEST07-B: extract AdaIR latent_pre (GAP+GMP pooled, 768-dim) for every
crop, fit StandardScaler+PCA-16 on TRAINING crops ONLY (leakage-safe, per
the task's strict rule), cache embeddings + transform + metadata.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python extract_teacher_embeddings.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

TEST07B = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07B.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST07B / "results" / "dataset_manifest.csv"
CACHE_DIR = TEST07B / "results" / "teacher_cache"
DEGS = ["Rain", "Haze", "Noise"]
PCA_DIM = 16
EXPECTED_SHA256 = "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb"


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    return torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def pooled_vec(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    gap = x.mean(dim=(2, 3))[0]
    gmp = x.amax(dim=(2, 3))[0]
    return torch.cat([gap, gmp]).cpu().numpy()


def checkpoint_sha256(path):
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
        p.requires_grad_(False)  # teacher frozen, no gradients ever
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)
    print(f"checkpoint OK: {n_params:,} params, frozen (requires_grad=False)", flush=True)

    records = []
    t0 = __import__("time").time()
    for idx, row in enumerate(rows):
        for deg in DEGS:
            img_t = to_tensor(load_rgb(row[f"{deg.lower()}_path"]), device)
            recorder.start()
            with torch.no_grad():
                _ = model(img_t)
            snap = recorder.snapshot_cpu()
            vec = pooled_vec(snap["_stages"]["latent"])
            records.append({"scene_id": row["scene_id"], "crop_id": row["crop_id"], "split": row["split"],
                             "degradation": deg, "vec": vec})
        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{len(rows)}] elapsed={__import__('time').time()-t0:.0f}s", flush=True)
    print(f"extracted {len(records)} latent_pre vectors ({len(records[0]['vec'])}-dim pooled GAP+GMP), "
          f"elapsed={__import__('time').time()-t0:.0f}s", flush=True)

    X = np.stack([r["vec"] for r in records])
    is_train = np.array([r["split"] == "train" for r in records])
    crop_id = np.array([r["crop_id"] for r in records])
    scene_id = np.array([r["scene_id"] for r in records])
    deg = np.array([r["degradation"] for r in records])

    scaler = StandardScaler().fit(X[is_train])
    X_scaled_train = scaler.transform(X[is_train])
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(X_scaled_train)
    explained_var = float(pca.explained_variance_ratio_.sum())
    per_component_var = pca.explained_variance_ratio_.tolist()
    print(f"PCA-{PCA_DIM} fit on {is_train.sum()} TRAINING rows ONLY. "
          f"Explained variance: {explained_var:.4f}", flush=True)

    X_scaled_all = scaler.transform(X)
    E = pca.transform(X_scaled_all)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / "latent_pre_pooled.npz", X=X, crop_id=crop_id, scene_id=scene_id,
              split=np.array([r["split"] for r in records]), degradation=deg)
    np.savez(CACHE_DIR / "pca16_embeddings.npz", E=E, crop_id=crop_id, scene_id=scene_id,
              split=np.array([r["split"] for r in records]), degradation=deg)
    joblib.dump({"scaler": scaler, "pca": pca}, CACHE_DIR / "pca16_transform.joblib")

    e_train = E[is_train]
    metadata = {
        "teacher_checkpoint_path": str(CKPT_PATH), "teacher_checkpoint_sha256": sha,
        "expected_sha256": EXPECTED_SHA256, "sha256_matches": True,
        "n_params": n_params, "teacher_frozen": True, "gradients_through_teacher": False,
        "raw_pooled_dim": int(X.shape[1]), "pca_dim": PCA_DIM,
        "pca_fit_sample_count": int(is_train.sum()),
        "pca_fit_source": "TEST07-B's own training split ONLY (fresh, leakage-safe fit; NOT reused from "
                          "test05_5/test07_pilot's transforms)",
        "pca_explained_variance_ratio_sum": explained_var,
        "pca_per_component_explained_variance_ratio": per_component_var,
        "teacher_embedding_mean": e_train.mean(axis=0).tolist(),
        "teacher_embedding_std": e_train.std(axis=0).tolist(),
        "n_records_total": len(records), "n_train_records": int(is_train.sum()),
        "n_val_records": int((~is_train).sum()),
    }
    with open(CACHE_DIR / "teacher_cache_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nwrote teacher cache to {CACHE_DIR}")
    print(f"explained_variance={explained_var:.4f}  n_train={is_train.sum()}  n_val={(~is_train).sum()}")


if __name__ == "__main__":
    main()
