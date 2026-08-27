"""TEST10 Phase 10: stage-wise teacher/student trajectory alignment for
Model G, plus a REPRESENTATIONAL-COLLAPSE CHECK that is not explicitly in
the task spec but is scientifically necessary here: a jointly-trained,
negative-free MSE-after-L2-normalize loss (as specified) is a well-known
setup for collapse -- both projection heads can trivially drive the loss to
~0 by mapping every input to the same constant unit vector, which would
show up as near-perfect cosine similarity WITHOUT the representation
carrying any actual per-sample information. The training log already showed
train_traj_loss collapsing to ~0.0000 by epoch 10, which is a red flag this
script investigates directly by measuring embedding VARIANCE across varied
inputs (degradation/scene), not just alignment to the teacher.

Also reports the final compact embedding (e_S/e_T, 16-dim) alignment for
comparability with TEST07-B/08-C/09's own alignment numbers.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python stage_alignment.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

TEST10 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10.parent
sys.path.insert(0, str(TEST10 / "src"))
from models import MODELS  # noqa: E402
from teacher_trajectory import TeacherTrajectoryHeads, load_frozen_teacher, extract_teacher_stage_pooled  # noqa: E402

TEST07B_RESULTS = TEACHER_EXP / "test07_b" / "results"
MANIFEST_PATH = TEST07B_RESULTS / "dataset_manifest.csv"
CACHE_DIR = TEST07B_RESULTS / "teacher_cache"
CKPT_DIR = TEST10 / "results" / "checkpoints"
OUT_DIR = TEST10 / "results" / "statistics"
DEGS = ["Rain", "Haze", "Noise"]
SEEDS = [0, 1, 2]


def load_rgb(path, device):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    device = "cuda"
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    d = np.load(CACHE_DIR / "pca16_embeddings.npz", allow_pickle=True)
    e_t_lookup = {(crop_id, deg): d["E"][i] for i, (crop_id, deg) in
                  enumerate(zip(d["crop_id"], d["degradation"]))}

    teacher_model, teacher_net, recorder = load_frozen_teacher(device)

    stage_rows, final_align_rows, collapse_rows = [], [], []

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"model_G_seed{seed}.pt"
        traj_ckpt_path = CKPT_DIR / f"trajheads_G_seed{seed}.pt"
        model = MODELS["G"]().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        traj_heads = TeacherTrajectoryHeads().to(device)
        traj_heads.load_state_dict(torch.load(traj_ckpt_path, map_location=device, weights_only=True))
        traj_heads.eval()

        stage_es = {0: [], 1: [], 2: []}
        stage_et = {0: [], 1: [], 2: []}
        final_es, final_et, degs_list, scenes_list = [], [], [], []

        with torch.no_grad():
            for row in val_rows:
                for deg in DEGS:
                    img_t = load_rgb(row[f"{deg.lower()}_path"], device)
                    out, e_s, e_s_traj = model.forward_trajectory(img_t)
                    teacher_pooled = extract_teacher_stage_pooled(teacher_model, recorder, img_t)
                    e_t_traj = traj_heads(teacher_pooled)

                    for stage_idx in (0, 1, 2):
                        stage_es[stage_idx].append(e_s_traj[stage_idx][0].cpu().numpy())
                        stage_et[stage_idx].append(e_t_traj[stage_idx][0].cpu().numpy())

                    final_es.append(e_s[0].cpu().numpy())
                    e_t_final = e_t_lookup.get((row["crop_id"], deg))
                    final_et.append(e_t_final)
                    degs_list.append(deg)
                    scenes_list.append(row["scene_id"])

        # ---- stage-wise cosine + normalized MSE + collapse diagnostics ----
        for stage_idx in (0, 1, 2):
            es = np.stack(stage_es[stage_idx])
            et = np.stack(stage_et[stage_idx])
            es_n = es / (np.linalg.norm(es, axis=1, keepdims=True) + 1e-12)
            et_n = et / (np.linalg.norm(et, axis=1, keepdims=True) + 1e-12)
            cos = (es_n * et_n).sum(axis=1)
            norm_mse = ((es_n - et_n) ** 2).mean()

            # collapse check: per-component std across the 60 val samples.
            # A genuinely input-dependent embedding should vary substantially
            # sample-to-sample; a collapsed one will have near-zero std.
            es_std = es_n.std(axis=0).mean()
            et_std = et_n.std(axis=0).mean()
            # reference: std of a set of random unit vectors in the same dim
            # (the "no collapse" ceiling) -- computed analytically is messy,
            # so instead compare against inter-sample pairwise cosine sim:
            # collapse => ALL sample pairs have cosine ~1 (embedding constant).
            pairwise_cos_es = es_n @ es_n.T
            pairwise_cos_et = et_n @ et_n.T
            iu = np.triu_indices(len(es_n), k=1)
            mean_pairwise_cos_es = float(pairwise_cos_es[iu].mean())
            mean_pairwise_cos_et = float(pairwise_cos_et[iu].mean())

            stage_rows.append({
                "seed": seed, "stage": stage_idx,
                "cosine_similarity_mean": float(cos.mean()), "cosine_similarity_std": float(cos.std()),
                "normalized_mse": float(norm_mse),
                "student_embedding_std": float(es_std), "teacher_embedding_std": float(et_std),
                "student_mean_pairwise_cosine": mean_pairwise_cos_es,
                "teacher_mean_pairwise_cosine": mean_pairwise_cos_et,
            })
            collapse_rows.append({
                "seed": seed, "stage": stage_idx,
                "collapsed": bool(mean_pairwise_cos_es > 0.98 or mean_pairwise_cos_et > 0.98),
                "student_mean_pairwise_cosine": mean_pairwise_cos_es,
                "teacher_mean_pairwise_cosine": mean_pairwise_cos_et,
                "interpretation": (
                    "LIKELY COLLAPSED: embeddings barely vary across different inputs "
                    "(mean pairwise cosine > 0.98 for a set of 60 varied degradation/scene "
                    "samples) -- the near-zero training loss reflects a trivial constant "
                    "mapping, not genuine trajectory matching."
                    if (mean_pairwise_cos_es > 0.98 or mean_pairwise_cos_et > 0.98)
                    else "Embeddings vary meaningfully across inputs; low training loss is "
                         "consistent with genuine (not collapsed) alignment."
                ),
            })
            print(f"seed{seed} stage{stage_idx}: cos={cos.mean():.4f} norm_mse={norm_mse:.4f} "
                  f"student_pairwise_cos={mean_pairwise_cos_es:.4f} teacher_pairwise_cos={mean_pairwise_cos_et:.4f}",
                  flush=True)

        # ---- final compact embedding alignment (comparable to TEST07-B/08-C/09) ----
        final_es_arr = np.stack(final_es)
        final_et_arr = np.stack(final_et)
        cos_final = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                     for a, b in zip(final_es_arr, final_et_arr)]
        mse_final = [float(np.mean((a - b) ** 2)) for a, b in zip(final_es_arr, final_et_arr)]
        final_align_rows.append({"seed": seed, "mean_cosine_similarity": float(np.mean(cos_final)),
                                  "mean_mse": float(np.mean(mse_final)), "n_examples": len(cos_final)})
        print(f"seed{seed} FINAL (16-dim KD) embedding: cosine={np.mean(cos_final):.4f} mse={np.mean(mse_final):.4f}",
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(OUT_DIR / "stage_alignment.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'stage_alignment.csv'}")

    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(OUT_DIR / "collapse_diagnostics.csv", index=False)
    print(f"wrote {OUT_DIR / 'collapse_diagnostics.csv'}")
    print("\n=== COLLAPSE CHECK SUMMARY ===")
    print(collapse_df[["seed", "stage", "collapsed", "student_mean_pairwise_cosine",
                        "teacher_mean_pairwise_cosine"]].to_string(index=False))

    final_align_df = pd.DataFrame(final_align_rows)
    final_align_df.to_csv(OUT_DIR / "final_embedding_alignment.csv", index=False)
    print(f"wrote {OUT_DIR / 'final_embedding_alignment.csv'}")


if __name__ == "__main__":
    main()
