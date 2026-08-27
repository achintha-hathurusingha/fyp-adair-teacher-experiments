"""TEST04 Phase 20: qualitative panels (clean/inputs/normal outputs/6
cross-degradation latent swaps) for the 10 scenes whose images were saved
during run_interventions.py. Qualitative only -- not primary evidence.

Usage: python build_visualizations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TEST04 = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = TEST04 / "results" / "tensors" / "output_images"
VIZ_DIR = TEST04 / "results" / "visualizations"
DEGS = ["rain", "haze", "noise"]
PAIRS = [("rain", "haze"), ("rain", "noise"), ("haze", "rain"),
         ("haze", "noise"), ("noise", "rain"), ("noise", "haze")]


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    scene_ids = sorted({p.stem.replace("_clean", "") for p in OUTPUTS_DIR.glob("scene_*_clean.png")})

    for scene_id in scene_ids:
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        clean_path = OUTPUTS_DIR / f"{scene_id}_clean.png"
        axes[0, 0].imshow(mpimg.imread(clean_path)); axes[0, 0].set_title(f"{scene_id}\nclean")
        for i, deg in enumerate(DEGS):
            axes[0, i + 1].imshow(mpimg.imread(OUTPUTS_DIR / f"{scene_id}_{deg}_input.png"))
            axes[0, i + 1].set_title(f"{deg} input")
        axes[0, 4].axis("off")

        for i, deg in enumerate(DEGS):
            axes[1, i].imshow(mpimg.imread(OUTPUTS_DIR / f"{scene_id}_{deg}_normal_output.png"))
            axes[1, i].set_title(f"{deg} normal output")
        axes[1, 3].axis("off"); axes[1, 4].axis("off")

        for i, (recipient, donor) in enumerate(PAIRS):
            path = OUTPUTS_DIR / f"{scene_id}_{recipient}+{donor}_latent.png"
            row, col = 2, i if i < 5 else None
            if col is not None:
                axes[2, col].imshow(mpimg.imread(path))
                axes[2, col].set_title(f"recipient={recipient}\ndonor={donor} (latent)")

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{scene_id} -- normal vs. same-scene cross-degradation latent-swap interventions")
        fig.tight_layout()
        fig.savefig(VIZ_DIR / f"{scene_id}_intervention_panel.png", dpi=105)
        plt.close(fig)
        print(f"saved {scene_id}_intervention_panel.png")


if __name__ == "__main__":
    main()
