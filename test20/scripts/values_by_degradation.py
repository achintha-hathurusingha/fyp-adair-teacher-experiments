"""TEST20 -- what internal AFLB values actually change with degradation type,
using Himeth's REPAIRED checkpoint (runs/finetune/C_full_soft/final.pt),
not the original released adair3d.ckpt.

Motivation: TEST01/06/18 found the RELEASED checkpoint's mask is degenerate
(active_fraction = 0.0 for every degradation, at every practical resolution)
-- so tracking "what changes by degradation" on that checkpoint is
uninformative by construction; there is nothing to track. Himeth's freq_fix.py
(finetune/freq_fix.py) independently found and fixed the exact same root
cause (h//128 floor + non-differentiable index-write mask), and its own
mask-ablation already showed the repaired mask is doing real, structured,
statistically significant work on dehaze. This asks the natural follow-up:
now that the mask can vary, does it actually vary BY DEGRADATION?

Non-invasive: reuses Himeth's own freq_fix.py utilities directly
(apply_freq_fix, load_adair_state) via a read-only import, rather than
re-deriving the repair. His FreModule.fft patch already stores
`{alpha, beta, coverage, hw}` on every forward (`_freqfix_last`) -- this
script just reads that, plus para1/para2 directly off the module
parameters, for each of the 3 AFLBs x 3 degradations, into one table.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIMETH_ADAIR = Path("/home/minura/FYP/Workspace/Himeth/AdaIR")
sys.path.insert(0, str(HIMETH_ADAIR / "AdaIR"))
sys.path.insert(0, str(HIMETH_ADAIR / "finetune"))
from net.model import AdaIR  # noqa: E402
from freq_fix import apply_freq_fix, load_adair_state  # noqa: E402

CKPT = HIMETH_ADAIR / "runs/finetune/C_full_soft/final.pt"
OUT_DIR = Path("/home/minura/teacher-experiments/test20/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = {
    "Rain": Path("/home/minura/teacher-experiments/test18/data/Train/Derain/rainy/rain-106.png"),
    "Haze": Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/input/0001_0.8_0.2.jpg"),
}
NOISE_SOURCE = Path("/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/target/0001.png")


def load_image_tensor(path, device, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def make_noisy(device, size=256, sigma=25):
    clean = load_image_tensor(NOISE_SOURCE, device, size)
    noise = torch.randn_like(clean) * (sigma / 255.0)
    return (clean + noise).clamp(0, 1)


def main():
    device = "cuda"
    model = AdaIR(decoder=True).to(device)
    mods = apply_freq_fix(model, mode="soft", tau=0.05)  # matches C_full_soft's own training config
    sd = load_adair_state(str(CKPT))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"repaired ckpt load: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model.eval()

    aflb_names = {id(model.fre1): "AFLB1", id(model.fre2): "AFLB2", id(model.fre3): "AFLB3"}
    para_means = {
        name: {"para1": float(getattr(model, f"fre{i+1}").para1.mean()),
               "para2": float(getattr(model, f"fre{i+1}").para2.mean())}
        for i, name in enumerate(["AFLB1", "AFLB2", "AFLB3"])
    }

    images = {
        "Rain": load_image_tensor(IMAGES["Rain"], device),
        "Haze": load_image_tensor(IMAGES["Haze"], device),
        "Noise": make_noisy(device),
    }

    rows = []
    with torch.no_grad():
        for deg, x in images.items():
            print(f"=== {deg} ===", flush=True)
            _ = model(x)  # single forward populates _freqfix_last on all 3 FreModules
            for m in mods:
                name = aflb_names[id(m)]
                last = m._freqfix_last
                rows.append({
                    "degradation": deg, "aflb": name,
                    "alpha": last["alpha"], "beta": last["beta"],
                    "coverage": last["coverage"], "feature_hw": list(last["hw"]),
                    "para1_mean": para_means[name]["para1"],  # fixed weights, not per-input
                    "para2_mean": para_means[name]["para2"],
                })
                print(f"  {name}: alpha={last['alpha']:.4f} beta={last['beta']:.4f} "
                      f"coverage={last['coverage']:.4f}", flush=True)

    with open(OUT_DIR / "values_by_degradation.json", "w") as f:
        json.dump(rows, f, indent=2)

    # CSV for quick inspection
    with open(OUT_DIR / "values_by_degradation.csv", "w") as f:
        f.write("degradation,aflb,alpha,beta,coverage,para1_mean,para2_mean\n")
        for r in rows:
            f.write(f"{r['degradation']},{r['aflb']},{r['alpha']:.6f},{r['beta']:.6f},"
                    f"{r['coverage']:.6f},{r['para1_mean']:.6f},{r['para2_mean']:.6f}\n")

    # ---- Grouped bar chart: coverage (the mask's actual active fraction) by AFLB x degradation ----
    aflbs = ["AFLB1", "AFLB2", "AFLB3"]
    degs = ["Rain", "Haze", "Noise"]
    colors = {"Rain": "#5b8def", "Haze": "#e8a33d", "Noise": "#5fbf8f"}
    cov = {(r["aflb"], r["degradation"]): r["coverage"] for r in rows}

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0a0d13")
    ax.set_facecolor("#0a0d13")
    x = np.arange(len(aflbs))
    width = 0.25
    for i, deg in enumerate(degs):
        vals = [cov[(a, deg)] for a in aflbs]
        ax.bar(x + (i - 1) * width, vals, width, label=deg, color=colors[deg])
    ax.set_xticks(x)
    ax.set_xticklabels(aflbs, color="#e8eaf0")
    ax.set_ylabel("mask coverage (active fraction)", color="#8891a8")
    ax.set_title("Himeth's REPAIRED mask -- coverage by AFLB x degradation", color="#e8eaf0")
    ax.tick_params(colors="#5c6478")
    for spine in ax.spines.values():
        spine.set_color("#262c3d")
    ax.legend(facecolor="#12161f", edgecolor="#262c3d", labelcolor="#e8eaf0")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_by_aflb_degradation.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nwrote {OUT_DIR}/values_by_degradation.{{json,csv}} and coverage_by_aflb_degradation.png")


if __name__ == "__main__":
    main()
