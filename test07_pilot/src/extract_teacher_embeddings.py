"""TEST07-Pilot: extract AdaIR latent_pre for every pilot image, fit PCA-16
on TRAINING images ONLY (per the task's explicit PCA-safety rule -- no
fitting on validation/test data), and cache both raw pooled latents and
the PCA-16 projection.

Reuses the read-only AdaIR loader/instrumentation from
teacher-experiments/scripts/instrument.py (test01-06_r's established
pattern), never modifies the checkpoint.

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

TEST07 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST07.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST07 / "results" / "pilot_manifest.csv"
CACHE_DIR = TEST07 / "results" / "teacher_cache"
DEGS = ["Rain", "Haze", "Noise"]
PCA_DIM = 16


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
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)
    print(f"checkpoint OK: {n_params:,} params", flush=True)

    records = []
    for row in rows:
        for deg in DEGS:
            img_t = to_tensor(load_rgb(row[f"{deg.lower()}_path"]), device)
            recorder.start()
            with torch.no_grad():
                _ = model(img_t)
            snap = recorder.snapshot_cpu()
            vec = pooled_vec(snap["_stages"]["latent"])
            records.append({"scene_id": row["scene_id"], "split": row["split"], "degradation": deg, "vec": vec})
    print(f"extracted {len(records)} latent_pre vectors ({len(records[0]['vec'])}-dim pooled GAP+GMP)", flush=True)

    X = np.stack([r["vec"] for r in records])
    is_train = np.array([r["split"] == "train" for r in records])
    y_scene = np.array([r["scene_id"] for r in records])
    y_deg = np.array([r["degradation"] for r in records])

    # ---- PCA safety: fit scaler + PCA on TRAINING rows ONLY ----
    scaler = StandardScaler().fit(X[is_train])
    X_scaled_train = scaler.transform(X[is_train])
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(X_scaled_train)
    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"PCA-{PCA_DIM} fit on {is_train.sum()} TRAINING rows ONLY. "
          f"Explained variance: {explained_var:.4f}", flush=True)

    X_scaled_all = scaler.transform(X)
    E = pca.transform(X_scaled_all)  # (N, 16), leakage-safe: transform only, fit was train-only

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / "latent_pre_pooled.npz", X=X, scene_id=y_scene, split=np.array([r["split"] for r in records]),
              degradation=y_deg)
    np.savez(CACHE_DIR / "pca16_embeddings.npz", E=E, scene_id=y_scene, split=np.array([r["split"] for r in records]),
              degradation=y_deg)
    joblib.dump({"scaler": scaler, "pca": pca}, CACHE_DIR / "pca16_transform.joblib")

    metadata = {
        "teacher_checkpoint_path": str(CKPT_PATH),
        "teacher_checkpoint_sha256": checkpoint_sha256(CKPT_PATH),
        "expected_sha256_from_test01_06r": "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb",
        "sha256_matches": checkpoint_sha256(CKPT_PATH) == "f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb",
        "n_params": n_params,
        "raw_pooled_dim": int(X.shape[1]),
        "pca_dim": PCA_DIM,
        "pca_fit_sample_count": int(is_train.sum()),
        "pca_fit_source": "TEST07-Pilot's own training split ONLY (NOT test05/test05_5's PCA transform, "
                          "NOT fit on validation data) -- a fresh, pilot-scoped, leakage-safe PCA, "
                          "methodology matches TEST05.5's leakage-safe re-audit (fit inside train fold only).",
        "pca_explained_variance_ratio_sum": explained_var,
        "n_scenes_total": len(rows), "n_records_total": len(records),
        "n_train_records": int(is_train.sum()), "n_val_records": int((~is_train).sum()),
    }
    with open(CACHE_DIR / "teacher_cache_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nwrote teacher cache to {CACHE_DIR}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
