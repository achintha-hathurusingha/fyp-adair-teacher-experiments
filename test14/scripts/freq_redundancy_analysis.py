"""TEST14: frequency vs degradation-embedding redundancy, and frequency vs
spatial-content redundancy. Computes linear correlation and grouped-probe
comparisons using e_D alone, q_F alone, phi(F) alone, and combinations, to
determine whether q_F provides information BEYOND e_D and phi(F) -- not
merely "both correlate with degradation."

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python freq_redundancy_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import CCA

TEST14 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST14.parent
sys.path.insert(0, str(TEST14 / "scripts"))
from models import MODELS  # noqa: E402
from frequency_descriptor import compute_qF  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CKPT_DIR = TEST14 / "results" / "checkpoints"
OUT_DIR = TEST14 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEED = 0  # use seed0 checkpoint for the representation-based redundancy analysis


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def grouped_probe(X, y, groups):
    y_enc = LabelEncoder().fit_transform(y)
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train_idx]), y_enc[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    t14 = MODELS["T14"]().to(device)
    t14.load_state_dict(torch.load(CKPT_DIR / f"model_T14_seed{SEED}.pt", map_location=device, weights_only=True))
    t14.eval()

    e_d_list, phi_list, qf_list, degs, scenes = [], [], [], [], []
    with torch.no_grad():
        for row in rows:
            for deg in DEGS:
                img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                out, e_d, a, phi, q_f = t14.forward_diagnostics(img_t)
                e_d_list.append(e_d[0].cpu().numpy())
                phi_list.append(phi[0].cpu().numpy())
                qf_list.append(q_f[0].cpu().numpy())
                degs.append(deg)
                scenes.append(row["scene_id"])

    E = np.stack(e_d_list)     # (N, 16)
    PHI = np.stack(phi_list)   # (N, 512)
    Q = np.stack(qf_list)      # (N, 8)
    degs, scenes = np.array(degs), np.array(scenes)

    # ---- linear correlation: max |corr| between any q_F band and any e_D dim ----
    corr_eq = np.zeros((E.shape[1], Q.shape[1]))
    for i in range(E.shape[1]):
        for j in range(Q.shape[1]):
            corr_eq[i, j] = np.corrcoef(E[:, i], Q[:, j])[0, 1]
    max_corr_eq = float(np.abs(corr_eq).max())
    mean_abs_corr_eq = float(np.abs(corr_eq).mean())

    # phi(F) is 512-dim; use a PCA-reduced summary (first 16 PCs) for a tractable correlation scan
    from sklearn.decomposition import PCA
    phi_pca = PCA(n_components=16, random_state=0).fit_transform(PHI)
    corr_pq = np.zeros((phi_pca.shape[1], Q.shape[1]))
    for i in range(phi_pca.shape[1]):
        for j in range(Q.shape[1]):
            corr_pq[i, j] = np.corrcoef(phi_pca[:, i], Q[:, j])[0, 1]
    max_corr_pq = float(np.abs(corr_pq).max())
    mean_abs_corr_pq = float(np.abs(corr_pq).mean())

    # ---- canonical correlation analysis: e_D vs q_F ----
    n_comp = min(E.shape[1], Q.shape[1])
    cca = CCA(n_components=n_comp)
    E_c, Q_c = cca.fit_transform(E, Q)
    cca_corrs = [float(np.corrcoef(E_c[:, i], Q_c[:, i])[0, 1]) for i in range(n_comp)]

    # ---- probe comparison: e_D alone, q_F alone, phi(F) alone, combinations ----
    probe_rows = []
    for name, X in [("e_D", E), ("q_F", Q), ("phi(F)_pca16", phi_pca),
                     ("[e_D,q_F]", np.concatenate([E, Q], axis=1)),
                     ("[phi(F)_pca16,q_F]", np.concatenate([phi_pca, Q], axis=1)),
                     ("[e_D,phi(F)_pca16,q_F]", np.concatenate([E, phi_pca, Q], axis=1)),
                     ("[e_D,phi(F)_pca16]", np.concatenate([E, phi_pca], axis=1))]:
        acc, f1 = grouped_probe(X, degs, scenes)
        probe_rows.append({"features": name, "dim": X.shape[1], "accuracy": acc, "macro_f1": f1})
        print(f"probe[{name}] (dim={X.shape[1]}): acc={acc*100:.1f}% f1={f1:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(OUT_DIR / "freq_redundancy_probe.csv", index=False)

    redundancy_summary = pd.DataFrame({
        "metric": ["max_abs_corr_eD_qF", "mean_abs_corr_eD_qF", "max_abs_corr_phiPCA_qF",
                   "mean_abs_corr_phiPCA_qF", "cca_top_correlation_eD_qF"],
        "value": [max_corr_eq, mean_abs_corr_eq, max_corr_pq, mean_abs_corr_pq, cca_corrs[0]],
    })
    redundancy_summary.to_csv(OUT_DIR / "freq_redundancy_summary.csv", index=False)
    print(f"\nwrote freq_redundancy_probe.csv and freq_redundancy_summary.csv")
    print(f"\nmax |corr| e_D vs q_F: {max_corr_eq:.3f} (mean {mean_abs_corr_eq:.3f})")
    print(f"max |corr| phi(F)_pca16 vs q_F: {max_corr_pq:.3f} (mean {mean_abs_corr_pq:.3f})")
    print(f"CCA top correlation e_D<->q_F: {cca_corrs[0]:.3f}")

    # ---- frequency signature: between-degradation vs within-scene-cross-degradation distance ----
    deg_means = {d: Q[degs == d].mean(axis=0) for d in DEGS}
    between_deg_dists = []
    for i, d1 in enumerate(DEGS):
        for d2 in DEGS[i + 1:]:
            between_deg_dists.append(float(np.linalg.norm(deg_means[d1] - deg_means[d2])))
    mean_between_deg = float(np.mean(between_deg_dists))

    df_q = pd.DataFrame(Q, columns=[f"band{i+1}" for i in range(8)])
    df_q["degradation"], df_q["scene_id"] = degs, scenes
    within_scene_dists = []
    for scene in np.unique(scenes):
        sub = df_q[df_q.scene_id == scene]
        vecs = {row.degradation: row[[f"band{i+1}" for i in range(8)]].values.astype(float)
                for _, row in sub.iterrows()}
        for i, d1 in enumerate(DEGS):
            for d2 in DEGS[i + 1:]:
                if d1 in vecs and d2 in vecs:
                    within_scene_dists.append(float(np.linalg.norm(vecs[d1] - vecs[d2])))
    mean_within_scene = float(np.mean(within_scene_dists))

    print(f"\nBetween-degradation q_F distance (mean): {mean_between_deg:.4f}")
    print(f"Within-scene, cross-degradation q_F distance (mean): {mean_within_scene:.4f}")
    print(f"(these should be similar in magnitude if degradation dominates q_F over scene identity)")

    sig_summary = pd.DataFrame({
        "metric": ["mean_between_degradation_distance", "mean_within_scene_cross_degradation_distance"],
        "value": [mean_between_deg, mean_within_scene],
    })
    sig_summary.to_csv(OUT_DIR / "freq_signature_summary.csv", index=False)


if __name__ == "__main__":
    main()
