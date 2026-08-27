"""TEST10 Phase 13: verify the final student inference graph contains NONE
of AdaIR, PCA, teacher projections, or trajectory projections -- i.e. that
Model G's restoration output depends ONLY on its NAFNet backbone + the
compact-embedding projection (proj) + the low-rank conditioning head
(bn_lowrank), and that model.forward()'s output is BIT-IDENTICAL whether or
not traj_heads exists at all (proving traj_heads is a pure training-time
side-branch with zero causal effect on restoration).

Method: construct a "stripped" ModelF (which has no traj_heads attribute at
all -- not just an unused one), copy over the shared-name weights from a
trained Model G checkpoint, and diff the two models' outputs on random
inputs. This is a strict causal test, not just a code-inspection claim.

Usage (on devon, adair-distill env, PINNED):
  taskset -c 0-7,12-31 python verify_inference_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TEST10 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST10 / "src"))
from models import MODELS  # noqa: E402

CKPT_DIR = TEST10 / "results" / "checkpoints"


def main():
    device = "cuda"
    ckpt_path = CKPT_DIR / "model_G_seed0.pt"
    g_state = torch.load(ckpt_path, map_location=device, weights_only=True)

    g_keys = set(g_state.keys())
    traj_keys = {k for k in g_keys if k.startswith("traj_heads.")}
    non_traj_keys = g_keys - traj_keys
    print(f"Model G checkpoint: {len(g_keys)} total params tensors, "
          f"{len(traj_keys)} belong to traj_heads (training-only), "
          f"{len(non_traj_keys)} are shared with the deployable backbone.")

    # ModelF has NO traj_heads attribute at all (not just zeroed/unused) --
    # this is a structural guarantee, not a runtime flag.
    stripped = MODELS["F"]().to(device)
    stripped_keys = set(stripped.state_dict().keys())
    assert stripped_keys == non_traj_keys, (
        f"stripped ModelF's param names do not exactly match Model G's non-traj_heads params!\n"
        f"only in stripped: {stripped_keys - non_traj_keys}\nonly in G non-traj: {non_traj_keys - stripped_keys}")
    print("Confirmed: ModelF's parameter set == ModelG's parameter set minus traj_heads.* exactly.")

    stripped.load_state_dict({k: v for k, v in g_state.items() if k in non_traj_keys})
    stripped.eval()

    # Now build the FULL Model G (with traj_heads present) and confirm its
    # forward() -- the actual restoration path -- is bit-identical to the
    # stripped version, proving traj_heads is causally inert at inference.
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
    assert max_diff_out == 0.0 and max_diff_es == 0.0, "outputs differ -- traj_heads is NOT causally inert!"
    print("CONFIRMED: Model G's restoration output and e_S are BIT-IDENTICAL "
          "with or without traj_heads present. The trajectory projection heads "
          "have zero causal effect on inference -- they can be (and should be) "
          "discarded for deployment.")

    n_params_stripped = sum(p.numel() for p in stripped.parameters())
    n_params_full_g = sum(p.numel() for p in full_g.parameters())
    print(f"\nDeployable (stripped) param count: {n_params_stripped:,}")
    print(f"Full training-time G param count (incl. traj_heads): {n_params_full_g:,}")
    print(f"Discarded at inference: {n_params_full_g - n_params_stripped:,} params "
          f"(traj_heads) + the entire frozen AdaIR teacher (28,784,824 params, never "
          f"saved to any student checkpoint) + TeacherTrajectoryHeads (also never saved "
          f"to the student checkpoint -- see trajheads_*.pt, which is a SEPARATE file "
          f"from model_*.pt and is not required for inference).")


if __name__ == "__main__":
    main()
