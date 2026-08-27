"""TEST10: loads the frozen AdaIR teacher (read-only reuse of
teacher-experiments/scripts/instrument.py, unmodified) and extracts the 3
chosen trajectory stages (AFLB1/AFLB2/AFLB3 aflb_out, see
report/teacher_stage_audit.md), GAP+GMP pooled. `TeacherTrajectoryHeads` is a
small trainable module (3 Linear heads, one per stage) that projects those
pooled features into the shared 32-dim trajectory space -- jointly optimized
with the student (per the task: "the goal is to learn a trainable stage
representation", not a fixed PCA). Both the teacher forward pass and these
heads are TRAINING-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

TEST10 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST10.parent
sys.path.insert(0, str(TEACHER_EXP / "scripts"))
from instrument import Recorder, attach_instrumentation, load_adair  # noqa: E402

ADAIR_DIR = TEACHER_EXP / "AdaIR"
CKPT_PATH = TEACHER_EXP / "weights" / "adair3d.ckpt"
TRAJ_DIM = 32

# AFLB name -> raw pooled dim (GAP+GMP over channel count from the audit)
TEACHER_STAGE_CHANNELS = {"AFLB1": 384, "AFLB2": 192, "AFLB3": 96}
# maps student stage_idx (0,1,2) -> teacher AFLB name, matched by spatial resolution
STUDENT_TO_TEACHER_STAGE = {0: "AFLB1", 1: "AFLB2", 2: "AFLB3"}


def pooled_gap_gmp(x: torch.Tensor) -> torch.Tensor:
    gap = x.mean(dim=(2, 3))
    gmp = x.amax(dim=(2, 3))
    return torch.cat([gap, gmp], dim=1)


def load_frozen_teacher(device: str):
    model = load_adair(ADAIR_DIR, CKPT_PATH, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    recorder = Recorder()
    net = attach_instrumentation(model, recorder)
    return model, net, recorder


@torch.no_grad()
def extract_teacher_stage_pooled(model, recorder: Recorder, x: torch.Tensor) -> dict:
    """Runs the frozen teacher forward (no grad), returns
    {stage_idx (0,1,2): pooled_raw_tensor} for the 3 chosen AFLB stages."""
    recorder.start()
    _ = model(x)
    snap = recorder._store  # stay on-device, no .cpu() -- used immediately
    out = {}
    for stage_idx, aflb_name in STUDENT_TO_TEACHER_STAGE.items():
        raw = snap[aflb_name]["aflb_out"]
        out[stage_idx] = pooled_gap_gmp(raw.detach())
    return out


class TeacherTrajectoryHeads(nn.Module):
    """3 trainable Linear heads, one per stage, projecting the teacher's
    pooled raw stage features into the shared 32-dim trajectory space."""

    def __init__(self, traj_dim: int = TRAJ_DIM):
        super().__init__()
        self.heads = nn.ModuleDict({
            str(stage_idx): nn.Linear(TEACHER_STAGE_CHANNELS[aflb_name] * 2, traj_dim)
            for stage_idx, aflb_name in STUDENT_TO_TEACHER_STAGE.items()
        })

    def forward(self, pooled_by_stage: dict) -> dict:
        return {idx: self.heads[str(idx)](feat) for idx, feat in pooled_by_stage.items()}
