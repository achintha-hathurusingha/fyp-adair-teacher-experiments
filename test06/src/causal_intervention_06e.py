"""TEST06 Phases 7-11 (06-E): THE PRIMARY CAUSAL EXPERIMENT.

At the confirmed-active resolution (1024x1024, R_first=768px for AFLB3),
swap ONLY AFLB3's (raw_high, raw_low) frequency-path tensors -- computed
by the released, unmodified fft() -- between conditions, leaving the
recipient's spatial branch (y), the checkpoint weights, and all downstream
FMiM/FMoM/channel_cross_agg architecture completely untouched. This
directly mirrors TEST05.5's Phase 10 methodology, but at a resolution
where the frequency path is genuinely non-degenerate (verified in Phase 6),
not the degenerate benchmark resolution TEST05.5 used.

Phase 7: baseline (normal) inference for every scene x degradation.
Phase 8: self-swap control (donor==recipient) -- must be near-zero, else
         the intervention mechanism itself is broken and we STOP.
Phase 9: same-scene cross-degradation swap (the primary causal test).
Phase 10: swap controls -- cross-scene same-degradation (content control),
          random matched-distribution tensor, zero tensor, mean tensor.
Phase 11: donor-behavior analysis -- does a swapped output move toward the
          DONOR's own normal output, or just get perturbed generically?

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python causal_intervention_06e.py
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
from PIL import Image

TEST06 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST06.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from instrument import Recorder, load_adair, TRANSFORMER_STAGES  # noqa: E402
from model_variants import _fft_released  # noqa: E402 (read-only reuse)
from metrics_utils import psnr_ssim_mse, output_diff  # noqa: E402 (read-only reuse)

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEST06 / "results" / "frequency_intervention" / "scene_manifest.csv"
OUT_DIR = TEST06 / "results" / "frequency_intervention"
DEGS = ["Rain", "Haze", "Noise"]


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _forward_with_aflb3_override(self, x, y, recorder, aflb_name, override_high, override_low):
    """Replica of FreModule.forward, but AFLB3's (high, low) can be
    overridden AFTER the real fft() has been computed and recorded (so we
    always know what the real values were, for the self-swap/energy checks)."""
    import torch.nn.functional as F
    _, _, H, W = y.size()
    x_r = F.interpolate(x, (H, W), mode="bilinear")
    real_high, real_low = _fft_released(self, x_r, recorder, aflb_name)

    use_high = override_high if override_high is not None else real_high
    use_low = override_low if override_low is not None else real_low
    if use_high.shape != real_high.shape:
        raise ValueError(f"shape mismatch: override {use_high.shape} vs real {real_high.shape}")

    high_feature = self.channel_cross_l(use_high, y)
    low_feature = self.channel_cross_h(use_low, y)
    recorder.put(aflb_name, "mined_high", high_feature)
    recorder.put(aflb_name, "mined_low", low_feature)
    agg = self.frequency_refine(low_feature, high_feature)
    out = self.channel_cross_agg(y, agg)
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
            lambda self, x, y, r=recorder, n="AFLB3", h=high, l=low: _forward_with_aflb3_override(
                self, x, y, r, n, h, l), fre3)

    set_override(None, None)  # default: no override, real fft() used
    return model, recorder, set_override, device


def run_normal(model, recorder, set_override, scene_row, device):
    """Cache on CPU -- 25 scenes x 3 degradations x (input+output+raw_high+
    raw_low) at 1024x1024 exceeds 24GB if kept on GPU simultaneously
    (confirmed by an earlier OOM crash). Moved to GPU just-in-time instead."""
    set_override(None, None)
    outputs = {}
    for deg in DEGS:
        img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_path"]), device)
        recorder.start()
        with torch.no_grad():
            out = model(img_t)
        snap = recorder.snapshot_cpu()
        outputs[deg] = {"input": img_t.cpu(), "output": out.detach().cpu(),
                         "raw_high": snap["AFLB3"]["raw_high"],
                         "raw_low": snap["AFLB3"]["raw_low"]}
        del img_t, out
    torch.cuda.empty_cache()
    return outputs


def swap_and_run(model, set_override, recipient_img_cpu, donor_high_cpu, donor_low_cpu, device):
    recipient_img = recipient_img_cpu.to(device)
    donor_high = donor_high_cpu.to(device)
    donor_low = donor_low_cpu.to(device)
    set_override(donor_high, donor_low)
    with torch.no_grad():
        out = model(recipient_img)
    set_override(None, None)
    result = out.detach().cpu()
    del recipient_img, donor_high, donor_low, out
    torch.cuda.empty_cache()
    return result


def metrics_row(swapped, normal, clean=None):
    diff = output_diff(swapped, normal)
    normal_l2 = float(torch.linalg.vector_norm(normal.reshape(-1)).item())
    m = psnr_ssim_mse(swapped, normal)
    row = {"l2": diff["l2"], "mae": diff["mae"], "normalized_l2": diff["l2"] / (normal_l2 + 1e-12),
           "psnr_vs_normal": m["psnr"], "ssim_vs_normal": m["ssim"]}
    return row


def main():
    with open(MANIFEST_PATH) as f:
        scene_rows = list(csv.DictReader(f))

    model, recorder, set_override, device = build_model()
    model.eval()

    baseline_rows, self_swap_rows, cross_deg_rows, control_rows, donor_behavior_rows = [], [], [], [], []
    t_start = time.time()

    # cache normals for ALL scenes first (needed for cross-scene control + donor-behavior)
    normals = {}
    for i, scene_row in enumerate(scene_rows):
        normals[scene_row["scene_id"]] = run_normal(model, recorder, set_override, scene_row, device)
        for deg in DEGS:
            o = normals[scene_row["scene_id"]][deg]
            baseline_rows.append({"scene_id": scene_row["scene_id"], "degradation": deg,
                                   "raw_high_shape": str(tuple(o["raw_high"].shape))})
        if (i + 1) % 5 == 0:
            print(f"[baseline {i+1}/{len(scene_rows)}] elapsed={time.time()-t_start:.0f}s", flush=True)

    scene_ids = list(normals.keys())

    # ---- Phase 8: self-swap control ----
    for scene_id in scene_ids:
        for deg in DEGS:
            o = normals[scene_id][deg]
            swapped = swap_and_run(model, set_override, o["input"], o["raw_high"], o["raw_low"], device)
            row = metrics_row(swapped, o["output"])
            row.update({"scene_id": scene_id, "degradation": deg})
            self_swap_rows.append(row)
    self_swap_df = pd.DataFrame(self_swap_rows)
    max_self_swap_l2 = self_swap_df.l2.max()
    print(f"\nPhase 8 self-swap max L2: {max_self_swap_l2:.8f} "
          f"({'PASS -- near-zero as expected' if max_self_swap_l2 < 1e-4 else 'FAIL -- STOP, debug intervention'})",
          flush=True)
    self_swap_df.to_csv(OUT_DIR / "self_swap.csv", index=False)
    if max_self_swap_l2 >= 1e-4:
        print("ABORTING per Phase 8's explicit instruction: self-swap is not near-identical.")
        return

    # ---- Phase 9: same-scene cross-degradation swap (PRIMARY CAUSAL TEST) ----
    for scene_id in scene_ids:
        for recipient in DEGS:
            for donor in DEGS:
                if donor == recipient:
                    continue
                r = normals[scene_id][recipient]
                d = normals[scene_id][donor]
                swapped = swap_and_run(model, set_override, r["input"], d["raw_high"], d["raw_low"], device)
                row = metrics_row(swapped, r["output"])
                row.update({"scene_id": scene_id, "recipient": recipient, "donor": donor,
                             "condition": "same_scene_cross_degradation"})
                cross_deg_rows.append(row)

                # ---- Phase 11: donor-behavior analysis ----
                dist_to_normal_recipient = metrics_row(swapped, r["output"])["normalized_l2"]
                dist_to_normal_donor = metrics_row(swapped, d["output"])["normalized_l2"]
                donor_behavior_rows.append({
                    "scene_id": scene_id, "recipient": recipient, "donor": donor,
                    "normalized_dist_to_normal_recipient": dist_to_normal_recipient,
                    "normalized_dist_to_normal_donor": dist_to_normal_donor,
                    "moved_toward_donor": dist_to_normal_donor < dist_to_normal_recipient,
                })
    print(f"Phase 9: {len(cross_deg_rows)} cross-degradation swaps done, elapsed={time.time()-t_start:.0f}s",
          flush=True)
    pd.DataFrame(cross_deg_rows).to_csv(OUT_DIR / "frequency_swap.csv", index=False)
    pd.DataFrame(donor_behavior_rows).to_csv(OUT_DIR / "donor_behavior.csv", index=False)

    # ---- Phase 10: swap controls ----
    rng = np.random.RandomState(0)
    for i, scene_id in enumerate(scene_ids):
        recipient_deg = DEGS[i % 3]
        r = normals[scene_id][recipient_deg]

        # B: cross-scene, same-degradation
        other_scene_id = scene_ids[(scene_ids.index(scene_id) + 1) % len(scene_ids)]
        d = normals[other_scene_id][recipient_deg]
        if d["raw_high"].shape == r["raw_high"].shape:
            swapped = swap_and_run(model, set_override, r["input"], d["raw_high"], d["raw_low"], device)
            row = metrics_row(swapped, r["output"])
            row.update({"scene_id": scene_id, "degradation": recipient_deg, "control": "cross_scene_same_degradation"})
            control_rows.append(row)

        # C: random, distribution-matched (CPU tensors -- swap_and_run moves to GPU internally)
        mean_h, std_h = r["raw_high"].mean(), r["raw_high"].std()
        mean_l, std_l = r["raw_low"].mean(), r["raw_low"].std()
        rand_high = torch.from_numpy(rng.randn(*r["raw_high"].shape).astype(np.float32)) * std_h + mean_h
        rand_low = torch.from_numpy(rng.randn(*r["raw_low"].shape).astype(np.float32)) * std_l + mean_l
        swapped = swap_and_run(model, set_override, r["input"], rand_high, rand_low, device)
        row = metrics_row(swapped, r["output"])
        row.update({"scene_id": scene_id, "degradation": recipient_deg, "control": "random_matched"})
        control_rows.append(row)

        # D: zero
        swapped = swap_and_run(model, set_override, r["input"], torch.zeros_like(r["raw_high"]),
                                torch.zeros_like(r["raw_low"]), device)
        row = metrics_row(swapped, r["output"])
        row.update({"scene_id": scene_id, "degradation": recipient_deg, "control": "zero"})
        control_rows.append(row)

        # E: mean (dataset-mean tensor, approximated per-scene since a true dataset mean
        # would require a full pass; using the recipient's own spatial mean as a documented proxy)
        mean_high = torch.full_like(r["raw_high"], float(mean_h.item()))
        mean_low = torch.full_like(r["raw_low"], float(mean_l.item()))
        swapped = swap_and_run(model, set_override, r["input"], mean_high, mean_low, device)
        row = metrics_row(swapped, r["output"])
        row.update({"scene_id": scene_id, "degradation": recipient_deg, "control": "mean"})
        control_rows.append(row)

    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(OUT_DIR / "frequency_swap_controls.csv", index=False)
    print(f"Phase 10: {len(control_rows)} control swaps done, elapsed={time.time()-t_start:.0f}s", flush=True)

    # ---- summary ----
    cross_df = pd.DataFrame(cross_deg_rows)
    print("\n" + "=" * 70)
    print("PRIMARY RESULT SUMMARY (normalized_l2, mean):")
    print(f"  same_scene_cross_degradation (A):  {cross_df.normalized_l2.mean():.6f}")
    for cond in ["cross_scene_same_degradation", "random_matched", "zero", "mean"]:
        sub = control_df[control_df.control == cond]
        if len(sub):
            print(f"  {cond} (control):  {sub.normalized_l2.mean():.6f}")
    donor_df = pd.DataFrame(donor_behavior_rows)
    print(f"\nDonor-behavior: swapped output closer to DONOR's own output than to "
          f"recipient's normal output in {donor_df.moved_toward_donor.mean()*100:.1f}% of swaps")
    print("=" * 70)


if __name__ == "__main__":
    main()
