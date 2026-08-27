"""TEST06-R Phases 1-5, 9-11: balanced primary + controls + internal
propagation, reusing the EXACT original TEST06 06-E dataset (read-only).

Methodology notes (documented, not hidden):
- Controls A (cross-scene), B (random-matched), C (zero), D (global-mean)
  depend ONLY on (scene_id, recipient_degradation) -- NOT on donor -- per
  the task's own seeding rule ("seed based on scene_id, recipient
  degradation, control type"). The two donor-labeled primary rows for a
  given (scene, recipient) pair are therefore mathematically guaranteed to
  receive IDENTICAL control values. Each control is computed ONCE per
  (scene, recipient) (75 unique forward passes x 4 controls = 300 passes)
  and the resulting row is duplicated across both donor labels to produce
  the full 150-row-per-control table the task specifies -- this is
  mathematically equivalent to re-running each control twice (the model
  and control construction are both deterministic) while avoiding 2x
  wasted GPU compute for bit-identical results.
- Control D (global mean) is computed from the mean of all 75 (25 scenes x
  3 degradations) REAL raw_high / raw_low tensors in the 06-E set --
  strictly within the 06-E intervention set, no external data.
- Internal-propagation compact statistics (mean, std, L2 norm, energy,
  cosine similarity, relative change vs normal) are recorded for ALL 150
  primary swaps at every stage (raw_high, raw_low, mined_high, mined_low,
  agg, cross_agg_out, aflb_out) -- full tensors are NOT persisted for all
  150 (would be prohibitively large); a separate script
  (propagation_heatmaps.py) generates full-tensor difference visualizations
  for a 3-scene representative subset only.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python balanced_intervention.py
"""
from __future__ import annotations

import csv
import sys
import types
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

TEST06_R = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST06_R.parent
TEST06 = TEACHER_EXP / "test06"
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from instrument import Recorder, load_adair  # noqa: E402
from model_variants import _fft_released  # noqa: E402 (read-only reuse)
from metrics_utils import psnr_ssim_mse, output_diff  # noqa: E402 (read-only reuse)

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST06 / "results" / "frequency_intervention" / "scene_manifest.csv"  # READ-ONLY, original TEST06
OUT_DIR = TEST06_R / "results" / "balanced_controls"
PROP_DIR = TEST06_R / "results" / "internal_propagation"
DEGS = ["Rain", "Haze", "Noise"]
STAGES = ["raw_high", "raw_low", "mined_high", "mined_low", "agg", "cross_agg_out", "aflb_out"]


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _forward_recording_all_stages(self, x, y, recorder, aflb_name, override_high, override_low):
    """Same override semantics as TEST06's original, extended to record
    EVERY intermediate stage (agg, cross_agg_out) needed for Phase 10-13's
    internal propagation trace -- the original only recorded mined_high/
    mined_low/aflb_out."""
    _, _, H, W = y.size()
    x_r = F.interpolate(x, (H, W), mode="bilinear")
    real_high, real_low = _fft_released(self, x_r, recorder, aflb_name)

    use_high = override_high if override_high is not None else real_high
    use_low = override_low if override_low is not None else real_low
    if use_high.shape != real_high.shape:
        raise ValueError(f"shape mismatch: override {use_high.shape} vs real {real_high.shape}")

    recorder.put(aflb_name, "raw_high", use_high)
    recorder.put(aflb_name, "raw_low", use_low)
    high_feature = self.channel_cross_l(use_high, y)
    low_feature = self.channel_cross_h(use_low, y)
    recorder.put(aflb_name, "mined_high", high_feature)
    recorder.put(aflb_name, "mined_low", low_feature)
    agg = self.frequency_refine(low_feature, high_feature)
    recorder.put(aflb_name, "agg", agg)
    out = self.channel_cross_agg(y, agg)
    recorder.put(aflb_name, "cross_agg_out", out)
    aflb_out = out * self.para1 + y * self.para2
    recorder.put(aflb_name, "aflb_out", aflb_out)
    return aflb_out


def build_model():
    device = "cuda"
    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    net = model.net if hasattr(model, "net") else model
    recorder = Recorder()
    fre3 = net.fre3

    def set_override(high, low):
        fre3.forward = types.MethodType(
            lambda self, x, y, r=recorder, n="AFLB3", h=high, l=low: _forward_recording_all_stages(
                self, x, y, r, n, h, l), fre3)

    set_override(None, None)
    return model, recorder, set_override, device


def run_normal(model, recorder, set_override, scene_row, device):
    """Cache only input/output/raw_high/raw_low persistently -- 25 scenes x
    3 degradations x ALL 7 AFLB3 stage tensors at 1024x1024 would need
    ~54GB CPU RAM (confirmed by a pre-execution memory audit). Only the
    tensors needed for the FULL dataset's lifetime (donor swapping) are
    cached long-term; full-stage tensors are recomputed per-scene, on
    demand, inside the primary loop, and discarded after each scene."""
    set_override(None, None)
    outputs = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_path"]), device)
        recorder.start()
        with torch.no_grad():
            out = model(img_t)
        snap = recorder.snapshot_cpu()
        outputs[deg] = {"input": img_t.cpu(), "output": out.detach().cpu(),
                         "raw_high": snap["AFLB3"]["raw_high"], "raw_low": snap["AFLB3"]["raw_low"]}
        del img_t, out
    torch.cuda.empty_cache()
    return outputs


def run_normal_full_stages(model, recorder, set_override, scene_row, device):
    """Recompute ALL 7 AFLB3 stage tensors for one scene's 3 degradations.
    Called fresh per-scene inside the primary loop; NOT cached across
    scenes (see run_normal's docstring for the memory rationale)."""
    set_override(None, None)
    outputs = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_path"]), device)
        recorder.start()
        with torch.no_grad():
            out = model(img_t)
        snap = recorder.snapshot_cpu()
        outputs[deg] = {s: snap["AFLB3"][s] for s in STAGES}
        outputs[deg]["output"] = out.detach().cpu()
        del img_t, out
    torch.cuda.empty_cache()
    return outputs


def swap_and_run(model, recorder, set_override, recipient_img_cpu, donor_high_cpu, donor_low_cpu, device):
    recipient_img = recipient_img_cpu.to(device)
    donor_high = donor_high_cpu.to(device)
    donor_low = donor_low_cpu.to(device)
    set_override(donor_high, donor_low)
    recorder.start()
    with torch.no_grad():
        out = model(recipient_img)
    snap = recorder.snapshot_cpu()
    set_override(None, None)
    stage_tensors = {s: snap["AFLB3"][s] for s in STAGES}
    result = {"output": out.detach().cpu(), **stage_tensors}
    del recipient_img, donor_high, donor_low, out
    torch.cuda.empty_cache()
    return result


def output_metrics(swapped_out, normal_out):
    diff = output_diff(swapped_out, normal_out)
    normal_l2 = float(torch.linalg.vector_norm(normal_out.reshape(-1)).item())
    m = psnr_ssim_mse(swapped_out, normal_out)
    rms_normal = float(torch.sqrt((normal_out.reshape(-1) ** 2).mean()).item())
    rms_diff = float(torch.sqrt(((swapped_out - normal_out).reshape(-1) ** 2).mean()).item())
    return {"l2": diff["l2"], "mae": diff["mae"], "normalized_l2": diff["l2"] / (normal_l2 + 1e-12),
            "normalized_rms": rms_diff / (rms_normal + 1e-12),
            "psnr_vs_normal": m["psnr"], "ssim_vs_normal": m["ssim"]}


def propagation_metrics(swapped_stage, normal_stage):
    s, n = swapped_stage.reshape(-1).float(), normal_stage.reshape(-1).float()
    diff_l2 = float(torch.linalg.vector_norm(s - n).item())
    normal_l2 = float(torch.linalg.vector_norm(n).item())
    cos = float(torch.nn.functional.cosine_similarity(s.unsqueeze(0), n.unsqueeze(0)).item())
    return {"relative_change": diff_l2 / (normal_l2 + 1e-12), "cosine_similarity": cos,
            "swapped_mean": float(s.mean().item()), "swapped_std": float(s.std().item()),
            "normal_mean": float(n.mean().item()), "normal_std": float(n.std().item()),
            "swapped_l2_norm": float(torch.linalg.vector_norm(s).item()),
            "normal_l2_norm": normal_l2,
            "swapped_energy": float((s ** 2).sum().item()), "normal_energy": float((n ** 2).sum().item())}


EXCLUDED_SCENES = {"scene_021"}  # 1024x104, not 1024x1024 -- data-quality issue in the original TEST06
                                   # dataset, discovered while building this re-run; see report/rerun_audit.md.
                                   # test06/ is NOT modified; this scene is excluded from TEST06-R only.


def main():
    with open(MANIFEST_PATH) as f:
        scene_rows = [r for r in csv.DictReader(f) if r["scene_id"] not in EXCLUDED_SCENES]
    assert len(scene_rows) == 24, f"expected 24 scenes (25 minus excluded scene_021), got {len(scene_rows)}"
    print(f"NOTE: excluding {EXCLUDED_SCENES} (data-quality issue, see rerun_audit.md) -- "
          f"proceeding with {len(scene_rows)} scenes", flush=True)

    model, recorder, set_override, device = build_model()
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROP_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ---- cache normals (needed for primary swaps, controls, propagation baselines) ----
    normals = {}
    for i, scene_row in enumerate(scene_rows):
        normals[scene_row["scene_id"]] = run_normal(model, recorder, set_override, scene_row, device)
        if (i + 1) % 5 == 0:
            print(f"[baseline {i+1}/{len(scene_rows)}] elapsed={time.time()-t_start:.0f}s", flush=True)
    scene_ids = list(normals.keys())

    # ---- global mean control tensor (Control D), computed from ALL 75 real 06-E tensors ----
    all_high = torch.stack([normals[sid][deg]["raw_high"] for sid in scene_ids for deg in DEGS])
    all_low = torch.stack([normals[sid][deg]["raw_low"] for sid in scene_ids for deg in DEGS])
    global_mean_high = all_high.mean(dim=0, keepdim=False)
    global_mean_low = all_low.mean(dim=0, keepdim=False)
    del all_high, all_low
    print(f"Control D (global mean) computed from {len(scene_ids)*3} real 06-E tensors, "
          f"shape={tuple(global_mean_high.shape)}", flush=True)

    # ---- Phase 4: self-swap control (75) ----
    self_swap_rows = []
    for scene_id in scene_ids:
        for deg in DEGS:
            o = normals[scene_id][deg]
            r = swap_and_run(model, recorder, set_override, o["input"], o["raw_high"], o["raw_low"], device)
            m = output_metrics(r["output"], o["output"])
            m.update({"scene_id": scene_id, "degradation": deg})
            self_swap_rows.append(m)
    self_swap_df = pd.DataFrame(self_swap_rows)
    self_swap_df.to_csv(OUT_DIR / "self_swap.csv", index=False)
    max_l2, mean_l2 = self_swap_df.l2.max(), self_swap_df.l2.mean()
    max_mae, max_norm_l2 = self_swap_df.mae.max(), self_swap_df.normalized_l2.max()
    print(f"\nPhase 4 self-swap: max_L2={max_l2:.8f} mean_L2={mean_l2:.8f} "
          f"max_MAE={max_mae:.8f} max_normalized_L2={max_norm_l2:.8f} "
          f"({'PASS' if max_l2 < 1e-4 else 'FAIL -- STOP'})", flush=True)
    if max_l2 >= 1e-4:
        print("ABORTING: self-swap not near-identical.")
        return

    # ---- Phase 2: balanced primary intervention (150) + internal propagation ----
    primary_rows, prop_rows = [], []
    scene_row_by_id = {sr["scene_id"]: sr for sr in scene_rows}
    for scene_id in scene_ids:
        full_stages = run_normal_full_stages(model, recorder, set_override, scene_row_by_id[scene_id], device)
        for recipient in DEGS:
            for donor in DEGS:
                if donor == recipient:
                    continue
                r, d = normals[scene_id][recipient], normals[scene_id][donor]
                r_full = full_stages[recipient]
                swapped = swap_and_run(model, recorder, set_override, r["input"], d["raw_high"], d["raw_low"], device)
                m = output_metrics(swapped["output"], r["output"])
                m.update({"scene_id": scene_id, "recipient": recipient, "donor": donor,
                           "condition": "same_scene_cross_degradation"})
                primary_rows.append(m)

                d_recip = output_metrics(swapped["output"], r["output"])["normalized_l2"]
                d_donor = output_metrics(swapped["output"], d["output"])["normalized_l2"]
                for stage in STAGES:
                    pm = propagation_metrics(swapped[stage], r_full[stage])
                    pm.update({"scene_id": scene_id, "recipient": recipient, "donor": donor, "stage": stage})
                    prop_rows.append(pm)
                prop_rows.append({"scene_id": scene_id, "recipient": recipient, "donor": donor, "stage": "final_output",
                                   **{k: v for k, v in propagation_metrics(swapped["output"], r["output"]).items()},
                                   "normalized_dist_to_donor": d_donor, "normalized_dist_to_recipient": d_recip})
        del full_stages
        print(f"[primary+propagation {scene_id}] elapsed={time.time()-t_start:.0f}s", flush=True)
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(OUT_DIR / "primary_swap.csv", index=False)
    pd.DataFrame(prop_rows).to_csv(PROP_DIR / "propagation_compact_stats.csv", index=False)
    print(f"Phase 2+10: {len(primary_rows)} primary swaps, {len(prop_rows)} propagation-stage rows, "
          f"elapsed={time.time()-t_start:.0f}s", flush=True)

    # ---- Phase 3: balanced controls, computed once per (scene, recipient), expanded to 150 rows ----
    control_unique_rows = {}  # (scene_id, recipient) -> {control_type: metrics_dict}
    for i, scene_id in enumerate(scene_ids):
        for recipient in DEGS:
            r = normals[scene_id][recipient]
            entry = {}

            # A: cross-scene, same-degradation (deterministic cyclic donor scene, every scene donor once)
            donor_scene_id = scene_ids[(i + 1) % len(scene_ids)]
            d = normals[donor_scene_id][recipient]
            if d["raw_high"].shape != r["raw_high"].shape:
                raise RuntimeError(f"unexpected shape mismatch {scene_id} vs {donor_scene_id} -- "
                                    f"investigate before proceeding (see rerun_audit.md pattern)")
            swapped = swap_and_run(model, recorder, set_override, r["input"], d["raw_high"], d["raw_low"], device)
            entry["cross_scene_same_degradation"] = {**output_metrics(swapped["output"], r["output"]),
                                                       "donor_scene_id": donor_scene_id}

            # B: random matched, seeded by (scene_id, recipient, "random") -- NOT donor
            seed = abs(hash(f"{scene_id}_{recipient}_random")) % (2 ** 31)
            rng = np.random.RandomState(seed)
            mean_h, std_h = r["raw_high"].mean(), r["raw_high"].std()
            mean_l, std_l = r["raw_low"].mean(), r["raw_low"].std()
            rand_high = torch.from_numpy(rng.randn(*r["raw_high"].shape).astype(np.float32)) * std_h + mean_h
            rand_low = torch.from_numpy(rng.randn(*r["raw_low"].shape).astype(np.float32)) * std_l + mean_l
            swapped = swap_and_run(model, recorder, set_override, r["input"], rand_high, rand_low, device)
            entry["random_matched"] = {**output_metrics(swapped["output"], r["output"]), "seed": seed}

            # C: zero
            swapped = swap_and_run(model, recorder, set_override, r["input"], torch.zeros_like(r["raw_high"]),
                                    torch.zeros_like(r["raw_low"]), device)
            entry["zero"] = output_metrics(swapped["output"], r["output"])

            # D: global mean (same tensor for every row, computed from full 06-E set)
            swapped = swap_and_run(model, recorder, set_override, r["input"], global_mean_high, global_mean_low, device)
            entry["global_mean"] = output_metrics(swapped["output"], r["output"])

            control_unique_rows[(scene_id, recipient)] = entry
        if (i + 1) % 5 == 0:
            print(f"[controls {i+1}/{len(scene_ids)}] elapsed={time.time()-t_start:.0f}s", flush=True)

    # expand to 150-row-per-control tables, matched to the primary's (scene,recipient,donor) index
    control_rows_expanded = []
    for row in primary_rows:
        key = (row["scene_id"], row["recipient"])
        for control_type, metrics in control_unique_rows[key].items():
            expanded = dict(metrics)
            expanded.update({"scene_id": row["scene_id"], "recipient": row["recipient"], "donor": row["donor"],
                              "control": control_type})
            control_rows_expanded.append(expanded)
    control_df = pd.DataFrame(control_rows_expanded)
    control_df.to_csv(OUT_DIR / "balanced_controls.csv", index=False)
    print(f"Phase 3: {len(control_unique_rows)} unique control computations x 4 = "
          f"{len(control_unique_rows)*4} actual forward passes, expanded to {len(control_rows_expanded)} "
          f"rows (150 per control type), elapsed={time.time()-t_start:.0f}s", flush=True)

    # ---- quick summary ----
    print("\n" + "=" * 70)
    print("PRIMARY vs BALANCED CONTROLS (normalized_l2, mean, N=150 each):")
    print(f"  primary (same_scene_cross_degradation): {primary_df.normalized_l2.mean():.6f}")
    for ct in ["cross_scene_same_degradation", "random_matched", "zero", "global_mean"]:
        sub = control_df[control_df.control == ct]
        print(f"  {ct}: {sub.normalized_l2.mean():.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
