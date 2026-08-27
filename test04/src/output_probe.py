"""TEST04 Phase 14: external degradation probe trained ONLY on normal
outputs (scene-grouped, mirroring TEST03's methodology), then evaluated on
both normal and swapped outputs to see whether interventions make
degradation information MORE recoverable from the final image.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python output_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST04 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST04.parent
sys.path.insert(0, str(TEST04 / "src"))
from metrics_utils import pooled_vec  # noqa: E402

OUTPUTS_DIR = TEST04 / "results" / "tensors" / "output_images"
INTERVENTIONS_DIR = TEST04 / "results" / "interventions"
STATS_DIR = TEST04 / "results" / "statistics"
MANIFEST_PATH = TEST04 / "results" / "manifest" / "scene_manifest.csv"
DEGS = ["Rain", "Haze", "Noise"]


def load_output_vec(path) -> np.ndarray:
    t = torch.from_numpy(np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return pooled_vec(t)


def main():
    import csv as csv_module
    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv_module.DictReader(f))
    scene_ids_with_viz = sorted({p.stem.split("_")[0] + "_" + p.stem.split("_")[1]
                                  for p in OUTPUTS_DIR.glob("scene_*_normal_output.png")})

    # Train probe on ALL normal outputs across the full 100-scene dataset --
    # need pooled vectors for every normal output. We only saved images for
    # the first 10 viz scenes as PNGs; recompute pooled stats for the rest
    # from run_interventions.py's cross_degradation_swaps.csv is not
    # possible (that CSV holds swapped outputs only) -- so train on the
    # available 10-scene x 3-degradation set of saved normal outputs
    # (n=30), which is what Phase 20's visualization subset provides, and
    # note this scope explicitly rather than silently re-running inference.
    X, y, groups = [], [], []
    for scene_id in scene_ids_with_viz:
        for deg in DEGS:
            p = OUTPUTS_DIR / f"{scene_id}_{deg.lower()}_normal_output.png"
            if p.exists():
                X.append(load_output_vec(p))
                y.append(deg)
                groups.append(scene_id)
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"trained on {len(X)} normal outputs from {len(set(groups))} scenes "
          f"(the Phase-20 visualization subset -- see report for scope note)", flush=True)

    Xs = StandardScaler().fit(X)
    X_scaled = Xs.transform(X)
    y_enc = LabelEncoder().fit(y)
    y_scaled = y_enc.transform(y)

    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_acc = []
    for train_idx, test_idx in gkf.split(X_scaled, y_scaled, groups=groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_scaled[train_idx], y_scaled[train_idx])
        fold_acc.append(accuracy_score(y_scaled[test_idx], clf.predict(X_scaled[test_idx])))
    normal_cv_acc = float(np.mean(fold_acc))
    print(f"normal-output probe accuracy (grouped CV, n_splits={n_splits}): {normal_cv_acc:.3f}", flush=True)

    # final classifier trained on ALL normal outputs, then applied to swapped outputs
    final_clf = LogisticRegression(max_iter=2000)
    final_clf.fit(X_scaled, y_scaled)

    rows = [{"eval_set": "normal_outputs_cv", "accuracy": normal_cv_acc, "n": len(X)}]

    for latent_path in sorted(OUTPUTS_DIR.glob("*+*_latent.png")):
        stem = latent_path.stem  # e.g. scene_001_rain+haze_latent
        parts = stem.split("_")
        scene_id = "_".join(parts[:2])
        recip_donor = parts[2]  # "rain+haze"
        recipient, donor = recip_donor.split("+")
        vec = load_output_vec(latent_path)
        vec_scaled = Xs.transform(vec[None, :])
        pred = y_enc.inverse_transform(final_clf.predict(vec_scaled))[0]
        proba = final_clf.predict_proba(vec_scaled)[0]
        rows.append({
            "eval_set": "swapped_output", "scene_id": scene_id, "recipient": recipient.capitalize(),
            "donor": donor.capitalize(), "predicted_class": pred,
            **{f"proba_{c}": p for c, p in zip(y_enc.classes_, proba)},
        })

    out = pd.DataFrame(rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATS_DIR / "output_degradation_probe.csv", index=False)
    print(f"\nwrote {STATS_DIR / 'output_degradation_probe.csv'} ({len(out)} rows)")

    swapped = out[out.eval_set == "swapped_output"]
    if len(swapped):
        pred_matches_donor = (swapped.predicted_class == swapped.donor).mean()
        pred_matches_recipient = (swapped.predicted_class == swapped.recipient).mean()
        print(f"\nOf {len(swapped)} swapped outputs: probe predicts DONOR class {pred_matches_donor*100:.1f}% "
              f"of the time, RECIPIENT class {pred_matches_recipient*100:.1f}% of the time")


if __name__ == "__main__":
    main()
