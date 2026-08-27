"""TEST17: the four models under hardware x restoration co-design.

A    -- TEST12's baseline NAFNet (layernorm2d backbone, no conditioning).
N    -- TEST16's normalization-surgery baseline (affine_clamp backbone,
         no conditioning). Same architecture as A except norm_type.
F2   -- TEST12's validated rank-2 low-rank conditional operator on the
         ORIGINAL layernorm2d backbone (compact degradation embedding
         e_D + feature conditioning phi(F), fixed U/V basis).
N+F2 -- F2's EXACT mechanism (same class, same _condition/forward code,
         inherited unchanged) applied to N's affine_clamp backbone. The
         only difference from F2 is which backbone `self.net` is built
         from -- no operator/rank/coefficient-generator changes, per
         TEST17 Phase 12's "no new architecture" rule.

All four are trained this pass (unlike TEST16 where N/S were
architecture-only) -- TEST17's whole point is a real quality number for
the normalization-surgery + F2 combination.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TEST17 = Path(__file__).resolve().parent.parent
TEACHER_EXP = TEST17.parent
TEST12_SCRIPTS = TEACHER_EXP / "test12" / "scripts"
TEST16_SCRIPTS = TEACHER_EXP / "test16" / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_test12_models = _load_module("test12_models", TEST12_SCRIPTS / "models.py")
_test16_models = _load_module("test16_models", TEST16_SCRIPTS / "models.py")

ModelA = _test12_models.ModelA
ModelF2 = _test12_models.ModelF2
PilotNAFNetBase = _test12_models.PilotNAFNetBase
LOCKED_CFG = _test12_models.LOCKED_CFG
BOTTLENECK_CHAN = _test12_models.BOTTLENECK_CHAN
POOLED_DIM = _test12_models.POOLED_DIM
PCA_DIM = _test12_models.PCA_DIM
RANK = _test12_models.RANK
pooled_gap_gmp = _test12_models.pooled_gap_gmp
zero_init_linear = _test12_models.zero_init_linear
build_base_nafnet = _test12_models.build_base_nafnet

ModelN = _test16_models.ModelN
N_CFG = _test16_models.N_CFG
build_norm_surgery_nafnet = _test16_models.build_norm_surgery_nafnet


class ModelNF2(ModelF2):
    """F2's mechanism, unchanged, on the affine_clamp (norm-surgery)
    backbone. Only __init__ differs from ModelF2 -- it swaps `self.net`
    to the norm-surgery NAFNet after the parent constructor runs. All
    conditioning code (_condition, forward, forward_diagnostics) is
    inherited from ModelF2 verbatim.
    """

    def __init__(self, rank: int = RANK):
        super().__init__(rank=rank)
        self.net = build_norm_surgery_nafnet()


MODELS = {"A": ModelA, "N": ModelN, "F2": ModelF2, "NF2": ModelNF2}
CONDITIONED_MODELS = {"F2", "NF2"}  # use KD loss; expose e_D/e_S
NORM_SURGERY_MODELS = {"N", "NF2"}  # affine_clamp backbone
