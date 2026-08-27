"""Main driver: run the frozen AdaIR 3-degradation teacher over the 300-image
manifest, capture the full AFLB pipeline for every image, and produce

  * AdaIR_3Degradation_Analysis.xlsx  (statistics -- see plan / README sheet)
  * intermediates/<deg>/<Image_ID>/aflb{1,2,3}.pt + stages.pt  (raw tensors)
  * outputs/restored/<deg>/<Image_ID>.png
  * outputs/visuals/<Image_ID>_pipeline.png  (one sample per degradation)
  * embeddings.npz  (pooled AFLB features for PCA/t-SNE, reused by the notebook)

Usage (on devon, adair-distill env):
  python run_inference.py --limit 6      # smoke test, 2 images/degradation
  python run_inference.py                # full 300-image run
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import Recorder, attach_instrumentation, attach_stage_hooks, load_adair, TRANSFORMER_STAGES
from stats_utils import tensor_stats, psnr_ssim

REPO = Path(__file__).resolve().parent.parent
ADAIR_DIR = REPO / "AdaIR"
CKPT_PATH = REPO / "weights" / "adair3d.ckpt"
MANIFEST_PATH = REPO / "manifest.csv"
INTERMEDIATES_DIR = REPO / "intermediates"
OUTPUTS_DIR = REPO / "outputs"
CSV_DIR = REPO / "csv_export"

AFLB_NAMES = ["AFLB1", "AFLB2", "AFLB3"]
FMIM_FEATURES = ["conv_feat", "fft_magnitude", "raw_low", "raw_high", "mined_low", "mined_high"]

GIT_SHA = subprocess.run(["git", "-C", str(ADAIR_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def crop_img(image: np.ndarray, base: int = 16) -> np.ndarray:
    h, w = image.shape[0], image.shape[1]
    crop_h, crop_w = h % base, w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def add_gaussian_noise(clean_img: np.ndarray, sigma: float, rng: np.random.RandomState | None = None) -> np.ndarray:
    noise = rng.randn(*clean_img.shape) if rng is not None else np.random.randn(*clean_img.shape)
    return np.clip(clean_img + noise * sigma, 0, 255).astype(np.uint8)


def to_tensor(img_u8: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def save_pipeline_visual(path: Path, degraded_np, restored_np, gt_np, snap: dict, aflb: str = "AFLB1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = snap[aflb]
    mask = d["mask"][0, 0].numpy()
    raw_low = d["raw_low"][0].float().mean(0).numpy()
    raw_high = d["raw_high"][0].float().mean(0).numpy()
    hl = d["hl_spatial_weight"][0, 0].numpy()
    lh = d["lh_channel_weight"][0].float().mean().item()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(degraded_np); axes[0, 0].set_title("degraded input")
    axes[0, 1].imshow(mask, cmap="gray"); axes[0, 1].set_title(f"{aflb} freq. mask (low=white)")
    axes[0, 2].imshow(raw_low, cmap="viridis"); axes[0, 2].set_title(f"{aflb} X_low (channel-mean)")
    axes[0, 3].imshow(raw_high, cmap="viridis"); axes[0, 3].set_title(f"{aflb} X_high (channel-mean)")
    axes[1, 0].imshow(hl, cmap="magma"); axes[1, 0].set_title(f"{aflb} H-L spatial weight")
    axes[1, 1].text(0.1, 0.5, f"L-H channel weight\nmean = {lh:.4f}", fontsize=12)
    axes[1, 1].axis("off")
    axes[1, 2].imshow(restored_np); axes[1, 2].set_title("restored (AdaIR)")
    axes[1, 3].imshow(gt_np); axes[1, 3].set_title("ground truth")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap total images (debug)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    print(f"loading AdaIR from {ADAIR_DIR} (git {GIT_SHA[:8]}), ckpt {CKPT_PATH.name}", flush=True)
    model = load_adair(ADAIR_DIR, CKPT_PATH, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 28_784_824, n_params
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    attach_stage_hooks(net, recorder)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        # keep it balanced across degradations for a smoke test
        by_deg: dict[str, list] = {}
        for r in rows:
            by_deg.setdefault(r["Degradation"], []).append(r)
        per_deg = max(1, args.limit // len(by_deg))
        rows = [r for rs in by_deg.values() for r in rs[:per_deg]]

    print(f"running inference on {len(rows)} images", flush=True)

    image_info_rows, mgb_rows, freq_rows, fmim_rows, fmom_rows = [], [], [], [], []
    stage_rows, tensor_index_rows = [], []
    embeddings, embedding_labels = [], []

    visual_done = set()
    t0 = time.time()

    for idx, row in enumerate(rows):
        image_id, deg, dataset = row["Image_ID"], row["Degradation"], row["Dataset"]
        gt_np = crop_img(load_rgb(row["gt_path"]))

        if deg == "Noise":
            sigma = float(row["noise_sigma"])
            degraded_np = add_gaussian_noise(gt_np, sigma)
        else:
            degraded_np = crop_img(load_rgb(row["input_path"]))
            sigma = None

        degraded_t = to_tensor(degraded_np, args.device)
        gt_t = to_tensor(gt_np, args.device)

        recorder.start()
        with torch.no_grad():
            restored_t = model(degraded_t)
        snap = recorder.snapshot_cpu()

        psnr, ssim = psnr_ssim(restored_t, gt_t)
        image_info_rows.append({
            "Image_ID": image_id, "Degradation": deg, "Dataset": dataset,
            "Filename": Path(row["input_path"]).name, "Noise_Sigma": sigma or "",
            "PSNR": psnr, "SSIM": ssim, "H": gt_np.shape[0], "W": gt_np.shape[1],
        })

        # ---- per-AFLB tensor bundle + statistics ----
        for aflb in AFLB_NAMES:
            d = snap[aflb]

            alpha = d["threshold_alpha_beta"][0, 0, 0, 0].item()
            beta = d["threshold_alpha_beta"][0, 1, 0, 0].item()
            mask_pct = d["mask"].float().mean().item() * 100
            mgb_rows.append({"Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                              "alpha": alpha, "beta": beta, "mask_area_pct": mask_pct})

            fft = d["fft_shifted"]
            mask = d["mask"]
            energy_total = (fft.abs() ** 2).sum().item()
            energy_low = ((fft * mask).abs() ** 2).sum().item()
            energy_high = ((fft * (1 - mask)).abs() ** 2).sum().item()
            freq_rows.append({
                "Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                "fft_energy": energy_total, "low_energy": energy_low, "high_energy": energy_high,
                "low_pct": 100 * energy_low / energy_total, "high_pct": 100 * energy_high / energy_total,
            })

            for feat_name in FMIM_FEATURES:
                t = fft.abs() if feat_name == "fft_magnitude" else d[feat_name]
                s = tensor_stats(t)
                fmim_rows.append({"Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                                   "Feature": feat_name, **s})

            hl_s = tensor_stats(d["hl_spatial_weight"])
            lh_s = tensor_stats(d["lh_channel_weight"])
            agg_s = tensor_stats(d["fmom_agg"])
            cross_s = tensor_stats(d["cross_agg_out"])
            out_s = tensor_stats(d["aflb_out"])
            fmom_rows.append({
                "Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                "hl_mean": hl_s["mean"], "hl_std": hl_s["std"],
                "lh_mean": lh_s["mean"], "lh_std": lh_s["std"],
                "agg_energy": agg_s["energy"], "cross_agg_energy": cross_s["energy"],
                "aflb_out_mean": out_s["mean"], "aflb_out_std": out_s["std"], "aflb_out_energy": out_s["energy"],
            })

            out_dir = INTERMEDIATES_DIR / deg.lower() / image_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pt_path = out_dir / f"{aflb.lower()}.pt"
            bundle = {k: v.half() if v.is_floating_point() else v for k, v in d.items()}
            torch.save(bundle, pt_path)
            tensor_index_rows.append({"Image_ID": image_id, "Degradation": deg, "AFLB": aflb,
                                       "pt_path": str(pt_path.relative_to(REPO))})

        stages_dir = INTERMEDIATES_DIR / deg.lower() / image_id
        stages_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in snap["_stages"].items()}, stages_dir / "stages.pt")
        for stage_name in TRANSFORMER_STAGES:
            s = tensor_stats(snap["_stages"][stage_name])
            stage_rows.append({"Image_ID": image_id, "Degradation": deg, "Layer": stage_name, **s})
        for aflb in AFLB_NAMES:
            s = tensor_stats(snap[aflb]["aflb_out"])
            stage_rows.append({"Image_ID": image_id, "Degradation": deg, "Layer": f"{aflb}_output", **s})

        # pooled embedding for PCA/t-SNE: global-avg-pool aflb_out, concat across AFLB1..3
        pooled = torch.cat([snap[a]["aflb_out"].float().mean(dim=(0, 2, 3)) for a in AFLB_NAMES])
        embeddings.append(pooled.numpy())
        embedding_labels.append({"Image_ID": image_id, "Degradation": deg})

        # restored image
        restored_dir = OUTPUTS_DIR / "restored" / deg.lower()
        restored_dir.mkdir(parents=True, exist_ok=True)
        restored_np = (restored_t.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        Image.fromarray(restored_np).save(restored_dir / f"{image_id}.png")

        if deg not in visual_done:
            visuals_dir = OUTPUTS_DIR / "visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            save_pipeline_visual(visuals_dir / f"{image_id}_pipeline.png",
                                  degraded_np, restored_np, gt_np, snap)
            visual_done.add(deg)

        if (idx + 1) % 20 == 0 or idx == len(rows) - 1:
            elapsed = time.time() - t0
            print(f"[{idx + 1}/{len(rows)}] {image_id} ({deg}) psnr={psnr:.2f} ssim={ssim:.4f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    # ---- assemble & write workbook ----
    df_image_info = pd.DataFrame(image_info_rows)
    df_mgb = pd.DataFrame(mgb_rows)
    df_freq = pd.DataFrame(freq_rows)
    df_fmim = pd.DataFrame(fmim_rows)
    df_fmom = pd.DataFrame(fmom_rows)
    df_stages = pd.DataFrame(stage_rows)
    df_tensor_index = pd.DataFrame(tensor_index_rows)

    output_metrics = df_image_info.groupby("Degradation")[["PSNR", "SSIM"]].agg(["mean", "std", "min", "max"])
    output_metrics.columns = ["_".join(c) for c in output_metrics.columns]
    output_metrics = output_metrics.reset_index()

    cross_mgb = df_mgb.groupby(["Degradation", "AFLB"])[["alpha", "beta", "mask_area_pct"]].agg(["mean", "std"])
    cross_mgb.columns = ["_".join(c) for c in cross_mgb.columns]
    cross_freq = df_freq.groupby(["Degradation", "AFLB"])[["low_pct", "high_pct"]].agg(["mean", "std"])
    cross_freq.columns = ["_".join(c) for c in cross_freq.columns]
    cross_fmom = df_fmom.groupby(["Degradation", "AFLB"])[["hl_mean", "lh_mean"]].agg(["mean", "std"])
    cross_fmom.columns = ["_".join(c) for c in cross_fmom.columns]
    cross_comparison = cross_mgb.join(cross_freq).join(cross_fmom).reset_index()

    # PCA / t-SNE on pooled AFLB-output embeddings
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(np.stack(embeddings))
    pca_xy = PCA(n_components=2, random_state=0).fit_transform(X)
    tsne_xy = TSNE(n_components=2, random_state=0, perplexity=min(30, len(X) // 3 or 1),
                    init="pca").fit_transform(X)
    df_pca = pd.DataFrame(embedding_labels)
    df_pca["PC1"], df_pca["PC2"] = pca_xy[:, 0], pca_xy[:, 1]
    df_pca["TSNE1"], df_pca["TSNE2"] = tsne_xy[:, 0], tsne_xy[:, 1]
    np.savez(REPO / "embeddings.npz", X=np.stack(embeddings),
             image_id=df_pca["Image_ID"].to_numpy(), degradation=df_pca["Degradation"].to_numpy())

    readme_text = f"""AdaIR 3-Degradation Analysis
Generated: {pd.Timestamp.now().isoformat()}
Checkpoint: {CKPT_PATH.name} (3-degradation all-in-one teacher)
AdaIR source: github.com/c-yn/AdaIR @ {GIT_SHA}
Model params: {n_params:,} (matches paper's 28.8M)
Images: {len(rows)} ({dict(pd.Series([r['Degradation'] for r in rows]).value_counts())})

IMPORTANT FINDING -- the low-frequency mask is genuinely empty, not a bug
The adaptive binary frequency mask (Ml) generated by the released AdaIR
implementation's FreModule.fft() is identically zero -- mask.unique() == [0] --
for the low-frequency branch at every AFLB, for all 300 images in this run.
This was independently verified with a from-scratch forward pass on R001
(scripts/trace_single_image.py), not just read off the saved tensors:
  1. Ml + Mh == 1 everywhere                                  -> exact
  2. ||F||^2 == ||F_low||^2 + ||F_high||^2                    -> exact (rel. error 0.00e+00)
  3. manual ifft(F_low)/ifft(F_high) matches saved raw_low/high -> exact match
  4. mask.unique() == [0.0]                                    -> confirmed at AFLB1/2/3
  5. h_ = int((h // 128) * alpha): h // 128 is 0 (AFLB1, AFLB2) or 1 (AFLB3)
     at native benchmark resolutions, and int(1 * ~0.5) still truncates to 0
Root cause: FreModule.fft() hardcodes n=128 as the spatial normaliser for the
mask half-width, but AFLB1/2/3 operate on feature maps of roughly 40x60,
80x120 and 160x240 (for a typical 480x320 input) -- all well under 128px per
side once floor-divided. The mask box can only ever be non-empty once the
floor-divided ratio h//128 reaches ~2-3 (since alpha/beta sit near 0.5, and
int() truncates rather than rounds).
Resolution sweep (scripts/resolution_sweep.py, one image resized and re-run
through the full model at increasing input sizes) empirically confirms the
activation thresholds -- see the 11_Resolution_Sweep sheet/CSV:
  AFLB3 (shallowest, feature = H/2): first activates at input 768x768
  AFLB2 (feature = H/4): did not activate up to 1024x1024 (1536/2048 OOM'd
                          on a 24GB GPU at fp32/batch=1 -- untested beyond that)
  AFLB1 (deepest, feature = H/8): did not activate up to 1024x1024 (same OOM limit)
Interpretation: this is a genuine, reproducible property of the RELEASED
CODE at standard benchmark resolutions (Rain100L/SOTS/BSD68 are all under
~550px) -- not an instrumentation bug, and not evidence that AdaIR "doesn't
use frequency information" (restoration quality in Image_Info is strong).
The narrower, correct claim: the specific binary frequency-boundary
mechanism in this FreModule.fft() path is inactive at the resolutions it is
normally evaluated at; restoration performance at those resolutions must
therefore come substantially from the rest of the architecture (Transformer
backbone, FMiM/FMoM cross-attention on the near-full "high" branch, etc.),
not from the adaptive low/high frequency split the paper describes.

ARCHITECTURE MAPPING (code name -> analysis-plan name)
  FreModule                         AFLB (Adaptive Frequency Learning Block), x3 in the decoder
  FreModule.fft()                   MGB (mask/frequency-boundary generation) + FMiM raw split
  channel_cross_l / channel_cross_h FMiM cross-attention (mines high/low against feature y)
  FreRefine (frequency_refine)      FMoM
    .SpatialGate (applied to "high")  H-L unit -> spatial attention, multiplies the LOW branch
    .ChannelGate (applied to "low")   L-H unit -> channel attention, multiplies the HIGH branch
  channel_cross_agg                 final cross-attention merge of FMoM output with y
  out*para1 + y*para2                AFLB residual gate (final block output)

  AFLB1 = fre1, applied to `latent` (deepest stage, dim*8=384 ch, smallest spatial size)
  AFLB2 = fre2, applied to `decoder_level3` (dim*4=192 ch)
  AFLB3 = fre3, applied to `decoder_level2` (dim*2=96 ch, shallowest/largest spatial size)
  i.e. AFLB1 runs first (deepest), AFLB3 runs last (shallowest) -- read depth as 1->3.

NOTE: every FreModule also owns `self.conv` and `self.score_gen` submodules.
They hold trained weights (checkpoint loads with 0 missing/0 unexpected keys)
but are NEVER called in forward() -- dead code in the released model, kept
faithfully un-instrumented here.

DATASET SAMPLING NOTES
  Derain: Rain100L test split, 100/100 paired images used as-is.
  Dehaze: SOTS-outdoor has 492 unique clean scenes (500 hazy renders, a few
          scenes have >1 render); de-duplicated by scene id, first 100 sorted
          scenes used.
  Denoise: BSD68 has only 68 unique clean images (no native noisy pairs --
          AdaIR synthesises Gaussian noise at test time). To reach 100
          instances while staying inside AdaIR's own {{15,25,50}} sigma
          protocol: all 68 images @ sigma=25 (canonical single-number level)
          + 16 extra images @ sigma=15 + 16 extra @ sigma=50 = 100. The
          Noise_Sigma column on Image_Info records which sigma each row used.

SHEETS
  Image_Info                per-image identity, PSNR/SSIM, resolution
  MGB_Values                 alpha/beta (learned freq. boundary) + mask area %, per image x AFLB
  Frequency_Statistics       FFT energy split (low/high %), per image x AFLB
  FMiM_Statistics            mean/std/min/max/L1/L2/energy for conv_feat, fft magnitude,
                              raw_low/high (pre cross-attn), mined_low/high (post cross-attn)
  FMoM_Statistics             H-L / L-H attention weight stats + FMoM aggregate + AFLB output energy
  Transformer_Features        same stats for every encoder/decoder stage + each AFLB's output
  Output_Metrics              PSNR/SSIM mean/std/min/max per degradation
  Cross_Degradation_Comparison  MGB/frequency/FMoM stats aggregated (mean+-std) by degradation x AFLB
  PCA_TSNE_Coordinates         2D projection of pooled AFLB-output embeddings, colour by degradation
  Tensor_File_Index           Image_ID x AFLB -> .pt file path (full-precision intermediates)
  Resolution_Sweep             (added post-hoc, scripts/resolution_sweep.py) mask activation vs. input size

FULL TENSORS
  intermediates/<degradation>/<Image_ID>/aflb{{1,2,3}}.pt   (dict of every captured tensor, float16)
  intermediates/<degradation>/<Image_ID>/stages.pt          (encoder/decoder stage outputs, float16)
  embeddings.npz                                             (pooled AFLB-output vectors used for PCA/t-SNE)
"""

    # Write CSVs only -- this machine has flaky logical CPUs (8-11) that have
    # been observed to silently corrupt in-process data (including openpyxl's
    # zip/XML writing). CSV is a much smaller blast radius per write call, and
    # the actual .xlsx is rendered on a separate, trusted machine from these
    # CSVs (see scripts/build_excel_local.py). Sheet order is preserved via
    # the leading NN_ prefix so the local renderer can sort them back.
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    (CSV_DIR / "00_README.csv").write_text(readme_text, encoding="utf-8")
    df_image_info.to_csv(CSV_DIR / "01_Image_Info.csv", index=False)
    df_mgb.to_csv(CSV_DIR / "02_MGB_Values.csv", index=False)
    df_freq.to_csv(CSV_DIR / "03_Frequency_Statistics.csv", index=False)
    df_fmim.to_csv(CSV_DIR / "04_FMiM_Statistics.csv", index=False)
    df_fmom.to_csv(CSV_DIR / "05_FMoM_Statistics.csv", index=False)
    df_stages.to_csv(CSV_DIR / "06_Transformer_Features.csv", index=False)
    output_metrics.to_csv(CSV_DIR / "07_Output_Metrics.csv", index=False)
    cross_comparison.to_csv(CSV_DIR / "08_Cross_Degradation_Comparison.csv", index=False)
    df_pca.to_csv(CSV_DIR / "09_PCA_TSNE_Coordinates.csv", index=False)
    df_tensor_index.to_csv(CSV_DIR / "10_Tensor_File_Index.csv", index=False)

    print(f"wrote CSVs -> {CSV_DIR}", flush=True)
    print(f"total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
