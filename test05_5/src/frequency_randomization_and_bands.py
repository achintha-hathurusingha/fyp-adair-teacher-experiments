"""TEST05.5 Phase 10-11.

Phase 10 -- FREQUENCY RANDOMIZATION CONTROL: pair the "high"/"low"
frequency-domain tensors extracted from image A's AFLB pass with a
DIFFERENT image B's spatial (post-cross-attention) path, via a paired
teacher variant, to test whether the FMiM/FMoM stages respond
specifically to the true (A,A) frequency-spatial correspondence or would
respond just as strongly to any mismatched (A,B) pairing. This is
distinct from TEST04's/Phase5-6's whole-tensor donor swap: here only the
raw_high/raw_low pair (pre channel_cross_l/h) is swapped, isolating the
frequency branch specifically rather than the whole AFLB output.

Phase 11 -- INPUT vs FEATURE FREQUENCY: for each of input / latent_pre /
AFLB1-3 aflb_out, independently measure the radial FFT low/mid/high energy
distribution (same 3-band method as TEST05, applied per-channel-averaged
magnitude spectrum) and report it WITHOUT assuming any degradation maps to
any band. TEST05's counter-intuitive "Noise -> low frequency" finding is
re-verified here on the fresh TEST03 dataset read (not merely cited).

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python frequency_randomization_and_bands.py [--limit N]
"""
from __future__ import annotations

import argparse
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

TEST05_5 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST05_5.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test01" / "scripts"))
sys.path.insert(0, str(TEACHER_EXP / "test04" / "src"))
from instrument import Recorder, load_adair, TRANSFORMER_STAGES  # noqa: E402
from model_variants import _fft_released  # noqa: E402 (read-only reuse)

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
MANIFEST_PATH = TEACHER_EXP / "test03" / "results" / "manifest" / "scene_manifest.csv"
OUT_DIR = TEST05_5 / "results" / "frequency"
DEGS = ["Rain", "Haze", "Noise"]
CANDIDATE_KEYS = ["input", "latent_pre", "AFLB1_aflb_out", "AFLB2_aflb_out", "AFLB3_aflb_out"]


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(img_u8, device):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


# ---------------------------------------------------------------- Phase 11
def radial_band_energy(gray: np.ndarray, n_bins=3):
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    total = mag.sum() + 1e-12
    fracs = []
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1] if i < n_bins - 1 else r <= bins[i + 1])
        fracs.append(float(mag[mask].sum() / total))
    return fracs  # [low, mid, high]


def tensor_radial_bands(t: torch.Tensor, n_bins=3):
    x = t.detach().float()[0]  # (C,H,W)
    per_channel = []
    for c in range(min(x.shape[0], 8)):  # cap channels for speed on high-dim feature maps
        per_channel.append(radial_band_energy(x[c].cpu().numpy(), n_bins))
    return list(np.mean(per_channel, axis=0))


def phase11_input_vs_feature(model, recorder, scene_rows, device, out_dir):
    rows = []
    for scene_row in scene_rows:
        scene_id = scene_row["scene_id"]
        for deg in DEGS:
            img_t = to_tensor(load_rgb(scene_row[f"{deg.lower()}_image_path"]), device)
            recorder.start()
            with torch.no_grad():
                _ = model(img_t)
            snap = recorder.snapshot_cpu()
            tensors = {"input": img_t, "latent_pre": snap["_stages"]["latent"],
                       "AFLB1_aflb_out": snap["AFLB1"]["aflb_out"],
                       "AFLB2_aflb_out": snap["AFLB2"]["aflb_out"],
                       "AFLB3_aflb_out": snap["AFLB3"]["aflb_out"]}
            for key in CANDIDATE_KEYS:
                low, mid, high = tensor_radial_bands(tensors[key])
                rows.append({"scene_id": scene_id, "degradation": deg, "point": key,
                             "low_frac": low, "mid_frac": mid, "high_frac": high})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "input_vs_feature_frequency.csv", index=False)
    summary = df.groupby(["point", "degradation"])[["low_frac", "mid_frac", "high_frac"]].mean().reset_index()
    summary.to_csv(out_dir / "input_vs_feature_frequency_summary.csv", index=False)
    print("\nPhase 11 -- radial band energy by point x degradation (re-verified, not assumed):")
    print(summary.to_string(index=False))
    return df


# ---------------------------------------------------------------- Phase 10
def _instrumented_forward_randomized(self, x, y, recorder, aflb_name, donor_high, donor_low):
    _, _, H, W = y.size()
    x_r = F.interpolate(x, (H, W), mode="bilinear")
    conv_feat = self.conv1(x_r)
    real_high, real_low = _fft_released(self, x_r, recorder, aflb_name)

    use_high = donor_high if donor_high is not None else real_high
    use_low = donor_low if donor_low is not None else real_low
    if use_high.shape != real_high.shape:
        use_high, use_low = real_high, real_low  # shape mismatch guard

    high_feature = self.channel_cross_l(use_high, y)
    low_feature = self.channel_cross_h(use_low, y)
    recorder.put(aflb_name, "mined_high", high_feature)
    recorder.put(aflb_name, "mined_low", low_feature)
    spatial_weight = self.frequency_refine.SpatialGate(high_feature)
    channel_weight = self.frequency_refine.ChannelGate(low_feature)
    agg = low_feature * spatial_weight + high_feature * channel_weight
    agg = self.frequency_refine.proj(agg)
    out = self.channel_cross_agg(y, agg)
    aflb_out = out * self.para1 + y * self.para2
    recorder.put(aflb_name, "aflb_out", aflb_out)
    return aflb_out


def phase10_frequency_randomization(scene_rows, device, out_dir, n_pairs=30):
    """For a set of image pairs (A,B), run image A normally, capture its
    raw_high/raw_low at AFLB1; then re-run A but with AFLB1's raw_high/low
    REPLACED by B's raw_high/low (extracted from a separate forward pass on
    B), and compare the resulting aflb_out / final output against A's own
    normal run. A large change => the frequency branch materially uses the
    correct (A,A) pairing; near-zero change => FMiM/FMoM are not sensitive
    to which image's frequency content they receive (evidence against
    "useful frequency information", consistent with generic statistics)."""
    from metrics_utils import psnr_ssim_mse, output_diff  # noqa: E402 (test04, read-only)

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    net = model.net if hasattr(model, "net") else model
    recorder = Recorder()
    fre1 = net.fre1

    def _capture_high_low(img_t):
        rec = Recorder()
        rec.start()
        orig_forward = fre1.forward
        fre1.forward = types.MethodType(
            lambda self, x, y, r=rec, n="AFLB1": _instrumented_forward_randomized(self, x, y, r, n, None, None),
            fre1)
        with torch.no_grad():
            _ = model(img_t)
        fre1.forward = orig_forward
        snap = rec.snapshot_cpu()
        return snap["AFLB1"]["raw_high"].to(device), snap["AFLB1"]["raw_low"].to(device)

    results = []
    rng = np.random.RandomState(0)
    pairs = [(i, (i + 1) % len(scene_rows)) for i in range(min(n_pairs, len(scene_rows)))]
    for ia, ib in pairs:
        row_a, row_b = scene_rows[ia], scene_rows[ib]
        for deg in DEGS:
            img_a = to_tensor(load_rgb(row_a[f"{deg.lower()}_image_path"]), device)
            img_b = to_tensor(load_rgb(row_b[f"{deg.lower()}_image_path"]), device)
            high_b, low_b = _capture_high_low(img_b)

            fre1.forward = types.MethodType(
                lambda self, x, y, r=recorder, n="AFLB1": _instrumented_forward_randomized(
                    self, x, y, r, n, None, None), fre1)
            with torch.no_grad():
                out_normal = model(img_a)

            fre1.forward = types.MethodType(
                lambda self, x, y, r=recorder, n="AFLB1", hb=high_b, lb=low_b: _instrumented_forward_randomized(
                    self, x, y, r, n, hb, lb), fre1)
            with torch.no_grad():
                out_swapped = model(img_a)

            od = output_diff(out_swapped, out_normal)
            m = psnr_ssim_mse(out_swapped, out_normal)
            results.append({"scene_a": row_a["scene_id"], "scene_b": row_b["scene_id"], "degradation": deg,
                             "l2": od["l2"], "mae": od["mae"], "psnr_vs_normal": m["psnr"], "ssim_vs_normal": m["ssim"]})

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "frequency_randomization_control.csv", index=False)
    print("\nPhase 10 -- frequency-branch-only randomization effect on final output:")
    print(df[["l2", "mae", "psnr_vs_normal", "ssim_vs_normal"]].mean().to_string())
    return df


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

    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    recorder = Recorder()
    net = model.net if hasattr(model, "net") else model
    from instrument import attach_instrumentation, attach_stage_hooks
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)

    t0 = time.time()
    phase11_input_vs_feature(model, recorder, scene_rows, device, OUT_DIR)
    print(f"Phase 11 done, elapsed={time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    phase10_frequency_randomization(scene_rows, device, OUT_DIR, n_pairs=min(30, len(scene_rows)))
    print(f"Phase 10 done, elapsed={time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
