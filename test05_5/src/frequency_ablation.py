"""TEST05.5 Phase 7-9: run T0/T1/T2/T3 (frequency_variants.py) over the
TEST03 dataset, extract pooled GAP+GMP representations at latent_pre and
AFLB1-3 aflb_out, and compare across variants:

  (a) degradation classification accuracy (grouped CV by scene)
  (b) scene sensitivity (probe accuracy for scene_id, capped at a
      manageable number of scene classes via pairwise same/diff-scene
      distance instead of full classification)
  (c) degradation/scene representation-distance ratio
  (d) leakage-safe compact PCA-16 accuracy
  (e) representation distance from T0 (mean pairwise L2, normalized)
  (f) restoration output PSNR/SSIM vs clean, and vs T0's own output

Honest-reporting requirement (task Phase 7-9): if T0 and T1 are identical
or near-identical, this MUST be reported as evidence against (not for) the
frequency-causal part of H_F2S, not hidden or explained away. Given the
degenerate mask fact documented in frequency_variants.py, T0==T1 EXACTLY
is the expected, mechanically-guaranteed outcome here, not a surprise --
but it is still reported plainly, and T2/T3 (the two conditions that
actually perturb the frequency-domain computation) carry the real
evidentiary weight for Phase 7-9.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python frequency_ablation.py [--limit N]
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
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from frequency_variants import load_variant, VARIANTS  # noqa: E402
from metrics_utils import psnr_ssim_mse  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEACHER_EXP / "test03" / "results" / "manifest" / "scene_manifest.csv"
OUT_DIR = TEST05_5 / "results" / "frequency"
DEGS = ["Rain", "Haze", "Noise"]
POINTS = ["latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def pooled_vec(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float()
    gap = x.mean(dim=(2, 3))[0]
    gmp = x.amax(dim=(2, 3))[0]
    return torch.cat([gap, gmp]).cpu().numpy()


def grouped_probe_acc(X, y, groups, n_splits=5):
    Xs = StandardScaler().fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(Xs, y_enc, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xs[train_idx], y_enc[train_idx])
        pred = clf.predict(Xs[test_idx])
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def leakage_safe_pca16_acc(X, y, groups, n_splits=5):
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        pca = PCA(n_components=16, random_state=0).fit(scaler.transform(X[train_idx]))
        X_train = pca.transform(scaler.transform(X[train_idx]))
        X_test = pca.transform(scaler.transform(X[test_idx]))
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(X_test)))
    return float(np.mean(accs))


def scene_sensitivity(X, scene_id, degradation, n_pairs=2000, seed=0):
    """Mean normalized L2 distance for same-scene/diff-degradation pairs vs
    diff-scene/same-degradation pairs -- avoids a full N-class scene probe
    (too many singleton classes for grouped CV) while still answering
    "does this representation separate scene identity from degradation
    identity." Ratio = degradation_effect / scene_effect (higher = more
    degradation-specific relative to scene)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    same_scene_diff_deg, diff_scene_same_deg = [], []
    idx_by_scene = {}
    for i, s in enumerate(scene_id):
        idx_by_scene.setdefault(s, []).append(i)
    scenes = list(idx_by_scene.keys())

    tries = 0
    while len(same_scene_diff_deg) < n_pairs and tries < n_pairs * 20:
        tries += 1
        s = scenes[rng.randint(len(scenes))]
        idxs = idx_by_scene[s]
        if len(idxs) < 2:
            continue
        i, j = rng.choice(idxs, size=2, replace=False)
        if degradation[i] != degradation[j]:
            d = np.linalg.norm(X[i] - X[j]) / (np.linalg.norm(X[i]) + 1e-12)
            same_scene_diff_deg.append(d)

    tries = 0
    while len(diff_scene_same_deg) < n_pairs and tries < n_pairs * 20:
        tries += 1
        i, j = rng.randint(0, n, size=2)
        if scene_id[i] != scene_id[j] and degradation[i] == degradation[j]:
            d = np.linalg.norm(X[i] - X[j]) / (np.linalg.norm(X[i]) + 1e-12)
            diff_scene_same_deg.append(d)

    return float(np.mean(same_scene_diff_deg)), float(np.mean(diff_scene_same_deg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = "cuda"

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))
    if args.limit:
        scene_rows = scene_rows[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled: dict[str, dict[str, list]] = {v: {p: [] for p in POINTS} for v in VARIANTS}
    outputs: dict[str, list] = {v: [] for v in VARIANTS}
    clean_imgs = []

    t_start = time.time()
    for vi, variant in enumerate(VARIANTS):
        model, recorder = load_variant(ADAIR_DIR, CKPT_PATH, device, variant, seed=0)
        for idx, scene_row in enumerate(scene_rows):
            scene_id = scene_row["scene_id"]
            clean_t = to_tensor(load_rgb(scene_row["clean_image_path"]), device)
            if vi == 0:
                clean_imgs.append(clean_t.cpu())
            for deg in DEGS:
                img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), device)
                recorder.start()
                with torch.no_grad():
                    out = model(img_t)
                snap = recorder.snapshot_cpu()
                tensors = {"latent_pre": snap["_stages"]["latent"],
                           "AFLB1_aflb_out": snap["AFLB1"]["aflb_out"],
                           "AFLB2_aflb_out": snap["AFLB2"]["aflb_out"],
                           "AFLB3_aflb_out": snap["AFLB3"]["aflb_out"]}
                for p in POINTS:
                    pooled[variant][p].append((scene_id, deg, pooled_vec(tensors[p])))
                m = psnr_ssim_mse(out, clean_t)
                outputs[variant].append({"scene_id": scene_id, "degradation": deg,
                                          "psnr_vs_clean": m["psnr"], "ssim_vs_clean": m["ssim"]})
            if (idx + 1) % 20 == 0:
                print(f"[{variant}] [{idx+1}/{len(scene_rows)}] elapsed={time.time()-t_start:.0f}s", flush=True)
        del model, recorder
        torch.cuda.empty_cache()
        print(f"variant {variant} done, elapsed={time.time()-t_start:.0f}s", flush=True)

    # ---- classification / scene-sensitivity / PCA-16 / distance-from-T0 ----
    summary_rows = []
    repr_arrays = {}
    for variant in VARIANTS:
        for p in POINTS:
            entries = pooled[variant][p]
            scene_id = np.array([e[0] for e in entries])
            deg = np.array([e[1] for e in entries])
            X = np.stack([e[2] for e in entries])
            repr_arrays[(variant, p)] = (X, deg, scene_id)

            acc, f1 = grouped_probe_acc(X, deg, scene_id)
            pca16_acc = leakage_safe_pca16_acc(X, deg, scene_id)
            deg_effect, scene_effect = scene_sensitivity(X, scene_id, deg)
            ratio = deg_effect / (scene_effect + 1e-12)
            summary_rows.append({
                "variant": variant, "point": p, "n_images": len(X),
                "degradation_probe_accuracy": acc, "degradation_probe_macro_f1": f1,
                "pca16_leakage_safe_accuracy": pca16_acc,
                "same_scene_diff_deg_distance": deg_effect,
                "diff_scene_same_deg_distance": scene_effect,
                "degradation_scene_ratio": ratio,
            })
            print(f"{variant} / {p}: deg_probe={acc*100:.1f}% pca16={pca16_acc*100:.1f}% "
                  f"deg/scene_ratio={ratio:.3f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "variant_representation_summary.csv", index=False)

    # ---- representation distance from T0 (same scene/degradation, paired) ----
    dist_rows = []
    for p in POINTS:
        X0, deg0, scene0 = repr_arrays[("T0_released", p)]
        key0 = {(s, d): X0[i] for i, (s, d) in enumerate(zip(scene0, deg0))}
        for variant in VARIANTS[1:]:
            Xv, degv, scenev = repr_arrays[(variant, p)]
            dists = []
            for i, (s, d) in enumerate(zip(scenev, degv)):
                if (s, d) in key0:
                    x0 = key0[(s, d)]
                    dists.append(np.linalg.norm(Xv[i] - x0) / (np.linalg.norm(x0) + 1e-12))
            dist_rows.append({"variant": variant, "point": p,
                               "mean_normalized_distance_from_T0": float(np.mean(dists)),
                               "n_pairs": len(dists)})
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "variant_distance_from_T0.csv", index=False)
    print("\nRepresentation distance from T0:")
    print(dist_df.to_string(index=False))

    # ---- restoration output quality per variant ----
    out_rows = []
    for variant in VARIANTS:
        df_out = pd.DataFrame(outputs[variant])
        out_rows.append({"variant": variant,
                          "mean_psnr_vs_clean": df_out.psnr_vs_clean.mean(),
                          "mean_ssim_vs_clean": df_out.ssim_vs_clean.mean()})
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_DIR / "variant_restoration_quality.csv", index=False)
    print("\nRestoration output quality per variant:")
    print(out_df.to_string(index=False))

    # ---- explicit T0 vs T1 identity check (expected: identical, per degenerate mask) ----
    t0_vs_t1 = dist_df[dist_df.variant == "T1_no_frequency"]
    max_dist = t0_vs_t1.mean_normalized_distance_from_T0.max() if len(t0_vs_t1) else float("nan")
    print(f"\nT0 vs T1 max mean normalized distance across points: {max_dist:.6f} "
          f"({'CONFIRMS degenerate-mask identity' if max_dist < 1e-4 else 'UNEXPECTED: T0 and T1 differ'})")

    print(f"\nwrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
