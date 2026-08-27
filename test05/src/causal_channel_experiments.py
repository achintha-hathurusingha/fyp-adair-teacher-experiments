"""TEST05 Phase 7-10: causal channel-level experiments, reusing TEST04's
verified-bit-exact manual_forward replica (test04/src/intervention.py,
read-only import -- not modified) to substitute PARTIAL (channel-subset)
tensors mid-forward-pass, not just whole tensors.

Phase 7 (channel ablation): for the top-ranked channels of `latent_pre`,
compare keep / zero / scene-average / degradation-group-average.

Phase 8 (grouped channel intervention): swap the top 5/10/20/30/50% most
degradation-specific channels (by channel_rank.csv) from a donor into a
recipient, holding all other channels at the recipient's own value;
compare against random channel groups of the same size.

Phase 9 (full vs. selected-channel swap): same-scene cross-degradation,
full tensor vs. top-K% channel subset.

Phase 10 (content control): the SAME channel-subset swap but cross-scene,
same-degradation -- does the selected subset remain sensitive to scene
changes the way the full tensor was in TEST04?

Runs on `latent_pre` only (the primary candidate, smallest tensor,
architecturally central) -- extending to every candidate feature would
multiply runtime ~30x for marginal additional evidence at this stage;
`latent_pre`'s ranking is representative of the aflb1_out family (TEST04
showed they are nearly behaviorally identical) and this is stated as an
explicit scope decision, not an oversight.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python causal_channel_experiments.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TEST05 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from instrument import load_adair  # noqa: E402
from intervention import manual_forward, sanity_check  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST05 / "results" / "manifest.csv"
CHANNEL_RANK_PATH = TEST05 / "results" / "channel_analysis" / "channel_rank.csv"
OUT_DIR = TEST05 / "results" / "intervention"
DEGS = ["Rain", "Haze", "Noise"]
POINT = "latent_pre"

N_SCENES_ABLATION = 20
N_SCENES_GROUP = 30
N_SCENES_CONTENT_CONTROL = 20
GROUP_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
TOP_K_ABLATION = 10


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def run_normal(model, scene_row, device):
    clean_t = to_tensor(load_rgb(scene_row["clean_image_path"]), device)
    cache = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), device)
        out = manual_forward(model, img_t)
        out["input"] = img_t
        cache[deg] = out
    return clean_t, cache


def output_l2_mae(a, b):
    diff = (a.detach().float() - b.detach().float())
    return float(torch.linalg.vector_norm(diff.reshape(-1)).item()), float(diff.abs().mean().item())


def main():
    device = "cuda"
    np.random.seed(0)
    torch.manual_seed(0)

    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    chan_rank = pd.read_csv(CHANNEL_RANK_PATH)
    latent_rank = chan_rank[chan_rank.feature == "latent_pre"].sort_values(
        "degradation_probe_accuracy", ascending=False)
    top_channels_by_pct = {}
    n_total = len(latent_rank)
    for pct in GROUP_FRACTIONS:
        k = max(1, int(round(n_total * pct)))
        top_channels_by_pct[pct] = latent_rank.head(k)["channel"].to_numpy()
    top10_channels = latent_rank.head(TOP_K_ABLATION)["channel"].to_numpy()
    print(f"latent_pre has {n_total} channels; top-{TOP_K_ABLATION}: {top10_channels}", flush=True)
    for pct, ch in top_channels_by_pct.items():
        print(f"  top-{int(pct*100)}%: {len(ch)} channels", flush=True)

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    print("checkpoint loaded", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ================= Phase 7: channel ablation =================
    ablation_rows = []
    subset = scene_rows[:N_SCENES_ABLATION]
    # need scene-average and degradation-group-average latent_pre values first
    all_latents = {}  # deg -> list of (scene_id, tensor)
    for scene_row in scene_rows:
        _, cache = run_normal(model, scene_row, device)
        for deg in DEGS:
            all_latents.setdefault(deg, []).append((scene_row["scene_id"], cache[deg]["latent_pre"]))
    # degradation-group average (only over the majority shape, matching TEST04's fix)
    from collections import Counter
    shape_counts = Counter(tuple(t.shape) for _, t in all_latents["Rain"])
    majority_shape = shape_counts.most_common(1)[0][0]
    deg_avg = {}
    for deg in DEGS:
        same_shape = [t for _, t in all_latents[deg] if tuple(t.shape) == majority_shape]
        deg_avg[deg] = torch.stack(same_shape).mean(dim=0)
    print(f"computed degradation-group averages over majority shape {majority_shape}", flush=True)

    for scene_row in subset:
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)
        for deg in DEGS:
            own_latent = cache[deg]["latent_pre"]
            if tuple(own_latent.shape) != majority_shape:
                continue
            recipient_img = cache[deg]["input"]
            normal_out = cache[deg]["output"]

            for ablation_type, replacement_fn in [
                ("keep", lambda t: t.clone()),
                ("zero", lambda t: torch.zeros_like(t)),
                ("degradation_group_avg", lambda t: deg_avg[deg]),
            ]:
                mod = own_latent.clone()
                repl = replacement_fn(own_latent)
                mod[:, top10_channels] = repl[:, top10_channels] if repl.shape == own_latent.shape else repl[:, top10_channels]
                swapped = manual_forward(model, recipient_img, overrides={POINT: mod})["output"]
                l2, mae = output_l2_mae(swapped, normal_out)
                ablation_rows.append({"scene_id": scene_id, "degradation": deg, "ablation_type": ablation_type,
                                       "n_channels": len(top10_channels), "l2_vs_normal": l2, "mae_vs_normal": mae})
        print(f"  ablation: {scene_id}", flush=True) if scene_id.endswith("005") or scene_id.endswith("010") or scene_id.endswith("015") or scene_id.endswith("020") else None

    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "channel_ablation.csv", index=False)
    print(f"wrote {OUT_DIR / 'channel_ablation.csv'} ({len(ablation_rows)} rows)", flush=True)

    # ================= Phase 8-9: grouped channel intervention (full vs subset) =================
    group_rows = []
    subset = scene_rows[:N_SCENES_GROUP]
    rng = np.random.RandomState(1)
    for scene_row in subset:
        scene_id = scene_row["scene_id"]
        clean_t, cache = run_normal(model, scene_row, device)
        for recipient in DEGS:
            for donor in DEGS:
                if donor == recipient:
                    continue
                recipient_img = cache[recipient]["input"]
                recipient_latent = cache[recipient]["latent_pre"]
                donor_latent = cache[donor]["latent_pre"]
                if recipient_latent.shape != donor_latent.shape:
                    continue
                normal_out = cache[recipient]["output"]

                # full swap (reference, matches TEST04)
                full_swapped = manual_forward(model, recipient_img, overrides={POINT: donor_latent})["output"]
                l2_full, mae_full = output_l2_mae(full_swapped, normal_out)
                group_rows.append({"scene_id": scene_id, "recipient": recipient, "donor": donor,
                                    "group_type": "full_tensor", "pct": 1.0, "n_channels": recipient_latent.shape[1],
                                    "l2_vs_normal": l2_full, "mae_vs_normal": mae_full})

                n_ch = recipient_latent.shape[1]
                for pct, top_ch in top_channels_by_pct.items():
                    mod = recipient_latent.clone()
                    mod[:, top_ch] = donor_latent[:, top_ch]
                    swapped = manual_forward(model, recipient_img, overrides={POINT: mod})["output"]
                    l2, mae = output_l2_mae(swapped, normal_out)
                    group_rows.append({"scene_id": scene_id, "recipient": recipient, "donor": donor,
                                        "group_type": "top_degradation_specific", "pct": pct, "n_channels": len(top_ch),
                                        "l2_vs_normal": l2, "mae_vs_normal": mae})

                    # random control group, same size
                    rand_ch = rng.choice(n_ch, size=len(top_ch), replace=False)
                    mod_r = recipient_latent.clone()
                    mod_r[:, rand_ch] = donor_latent[:, rand_ch]
                    swapped_r = manual_forward(model, recipient_img, overrides={POINT: mod_r})["output"]
                    l2_r, mae_r = output_l2_mae(swapped_r, normal_out)
                    group_rows.append({"scene_id": scene_id, "recipient": recipient, "donor": donor,
                                        "group_type": "random_same_size", "pct": pct, "n_channels": len(rand_ch),
                                        "l2_vs_normal": l2_r, "mae_vs_normal": mae_r})
        print(f"  group intervention: {scene_id}", flush=True)

    pd.DataFrame(group_rows).to_csv(OUT_DIR / "channel_group_intervention.csv", index=False)
    print(f"wrote {OUT_DIR / 'channel_group_intervention.csv'} ({len(group_rows)} rows)", flush=True)

    # ================= Phase 10: content control (cross-scene, top-10% channels) =================
    content_rows = []
    subset = scene_rows[:N_SCENES_CONTENT_CONTROL]
    normals = {sr["scene_id"]: run_normal(model, sr, device) for sr in subset}
    scene_ids = list(normals.keys())
    top_ch_10pct = top_channels_by_pct[0.10]
    skipped = 0
    for i in range(1, len(scene_ids)):
        recipient_scene, donor_scene = scene_ids[i], scene_ids[i - 1]
        r_clean, r_cache = normals[recipient_scene]
        d_clean, d_cache = normals[donor_scene]
        for deg in DEGS:
            recipient_latent = r_cache[deg]["latent_pre"]
            donor_latent = d_cache[deg]["latent_pre"]
            if recipient_latent.shape != donor_latent.shape:
                skipped += 1
                continue
            recipient_img = r_cache[deg]["input"]
            normal_out = r_cache[deg]["output"]

            mod = recipient_latent.clone()
            mod[:, top_ch_10pct] = donor_latent[:, top_ch_10pct]
            swapped = manual_forward(model, recipient_img, overrides={POINT: mod})["output"]
            l2, mae = output_l2_mae(swapped, normal_out)
            content_rows.append({"recipient_scene": recipient_scene, "donor_scene": donor_scene,
                                  "degradation": deg, "group_type": "top_10pct_channels_cross_scene",
                                  "l2_vs_normal": l2, "mae_vs_normal": mae})
    if skipped:
        print(f"  skipped {skipped} cross-scene pairs (shape mismatch)", flush=True)
    pd.DataFrame(content_rows).to_csv(OUT_DIR / "content_control.csv", index=False)
    print(f"wrote {OUT_DIR / 'content_control.csv'} ({len(content_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
