"""TEST10-R Phase 15: verify the final student inference graph contains NO
teacher projection heads and Model G's output is bit-identical whether or
not traj_heads exists at all.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python verify_inference_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TEST10R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST10R / "src"))
from models import MODELS  # noqa: E402

CKPT_DIR = TEST10R / "results" / "checkpoints"


def main():
    device = "cuda"
    ckpt_path = CKPT_DIR / "model_G_seed0.pt"
    g_state = torch.load(ckpt_path, map_location=device, weights_only=True)

    g_keys = set(g_state.keys())
    traj_keys = {k for k in g_keys if k.startswith("traj_heads.")}
    non_traj_keys = g_keys - traj_keys
    print(f"Model G checkpoint: {len(g_keys)} total param tensors, "
          f"{len(traj_keys)} belong to traj_heads (training-only), "
          f"{len(non_traj_keys)} shared with the deployable backbone.")

    stripped = MODELS["F"]().to(device)
    stripped_keys = set(stripped.state_dict().keys())
    assert stripped_keys == non_traj_keys, "param name mismatch between stripped F and G-minus-traj_heads!"
    print("Confirmed: ModelF's parameter set == ModelG's parameter set minus traj_heads.* exactly.")

    stripped.load_state_dict({k: v for k, v in g_state.items() if k in non_traj_keys})
    stripped.eval()

    full_g = MODELS["G"]().to(device)
    full_g.load_state_dict(g_state)
    full_g.eval()

    torch.manual_seed(0)
    x = torch.randn(4, 3, 128, 128).to(device)
    with torch.no_grad():
        out_stripped, e_s_stripped = stripped(x)
        out_full, e_s_full = full_g(x)

    max_diff_out = float((out_stripped - out_full).abs().max())
    max_diff_es = float((e_s_stripped - e_s_full).abs().max())
    print(f"max |out_stripped - out_full| = {max_diff_out:.3e}")
    print(f"max |e_s_stripped - e_s_full| = {max_diff_es:.3e}")
    assert max_diff_out == 0.0 and max_diff_es == 0.0, "traj_heads is NOT causally inert!"
    print("CONFIRMED: Model G's restoration output and e_S are BIT-IDENTICAL with or without "
          "traj_heads present. No AdaIR, no PCA, no teacher projections, no trajectory projections "
          "are required at inference -- the deployable graph is self-contained.")

    n_params_stripped = sum(p.numel() for p in stripped.parameters())
    n_params_full_g = sum(p.numel() for p in full_g.parameters())
    print(f"\nDeployable (stripped) param count: {n_params_stripped:,}")
    print(f"Training-time G param count (incl. traj_heads): {n_params_full_g:,}")
    print(f"Discarded at inference: {n_params_full_g - n_params_stripped:,} traj_heads params, "
          f"the entire frozen 28,784,824-param AdaIR teacher (never saved to any student checkpoint), "
          f"and the fixed StandardScaler+PCA transforms used only to build the training-time targets.")


if __name__ == "__main__":
    main()
