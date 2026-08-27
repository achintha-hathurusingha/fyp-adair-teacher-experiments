"""Writes results/README.csv for the TEST03 workbook. Run LOCALLY."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST03 = Path(__file__).resolve().parent.parent

readme_text = """TEST03 -- Controlled Same-Scene Degradation Representation Study
Research question: for the SAME clean scene, does changing ONLY the
degradation cause a systematic, measurable, scene-independent change in
AdaIR's internal representations?
Full report: report/test03_report.md
Design doc: report/test03_design.md
Data validation: report/synthetic_data_validation.md

DIRECTORY RULE
Everything for this experiment lives under test03/. test01/, test02/, the
original AdaIR source, and the checkpoint are read-only references and
were NOT modified.

WHY TEST02 WAS INSUFFICIENT
TEST02's Rain/Haze/Noise came from three disjoint source datasets
(Rain100L/SOTS-outdoor/BSD68) -- degradation type was perfectly confounded
with dataset domain. Its 71.7% input-only classification accuracy proved
substantial separability already existed in raw pixels before any AdaIR
computation.

WHAT TEST03 CHANGES
100 clean scenes (Rain100L ground-truth norain-*.png images) are used as a
SHARED base pool. All three degradations are synthesized by TEST03 itself,
deterministically, from the SAME clean image per scene -- see
degradation_synthesis.py / Degradation_Parameters sheet for exact,
documented parameters (rain: synthetic streaks; haze: atmospheric
scattering model with a documented synthetic depth PROXY, not a real depth
estimate; noise: additive Gaussian sigma=25). Evaluation unit is the SCENE:
all cross-validation is GroupKFold(group=scene_id) -- no fold ever trains
on one degraded version of a scene while testing on another version of the
SAME scene (leakage-impossible by construction, asserted in code).

MODEL
Released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt, 28,784,824 params,
0 missing/0 unexpected keys -- IDENTICAL checkpoint and loader to
test01/test02. No retraining, no architecture change. No degradation label
ever enters AdaIR.

HEADLINE RESULT -- THE CONTROLLED TRAJECTORY
(logistic regression, 5-fold GROUPED CV accuracy, group=scene_id; compare
directly against TEST02 in the TEST02_vs_TEST03 sheet)

  Input                    66.7%   (TEST02: 71.7%)
  Shallow (Y0)              99.3%   (TEST02: 93.3%)
  Encoder L1/L2/L3         100.0% / 100.0% / 100.0%   (TEST02: 98.0% / 99.3% / 99.3%)
  Latent                   100.0%   (TEST02: 100.0%)
  AFLB 1/2/3               100.0% / 100.0% / 99.7%   (TEST02: 99.7% / 99.0% / 99.7%)
  Decoder L3/L2/L1          99.7% / 99.7% / 100.0%   (TEST02: 99.0% / 99.7% / 99.7%)
  Refinement               100.0%   (TEST02: 99.7%)
  Output (restored)         37.0%   (TEST02: 54.7%)

OBSERVATION: under a controlled same-scene design with scene-grouped
cross-validation (leakage impossible by construction), a linear classifier
achieves 100.0% accuracy with ZERO variance across Encoder L1 through
Refinement, using features from scenes it has NEVER seen any version of
during training. Accuracy at Output drops to 37.0% -- even closer to the
33.3% random baseline than TEST02's 54.7%, and LOWER than TEST03's own
input-only baseline (66.7%).
INFERENCE: this is strong evidence that AdaIR's internal representations
encode genuine degradation-specific information, not merely dataset/domain
identity -- the confound TEST02 could not rule out is directly addressed
here and the effect survives (in fact strengthens) under the controlled
design.

SAME-SCENE VS CROSS-SCENE DISTANCE (Scene_vs_Degradation sheet) -- the
second, independent line of evidence
degradation_ratio = mean(same-scene, different-degradation distance) /
                     mean(same-degradation, different-scene distance)
Top features: AFLB3_lh_channel_weight (2.57), AFLB3_mined_low (2.39),
AFLB3_fmom_agg (2.14), AFLB1_mined_low (2.06) -- degradation moves the
representation 2-2.6x MORE than scene content does.
STRIKING: `output` scores 0.30 (<1) -- for the FINAL RESTORED IMAGE, scene
identity dominates over degradation type (restorations of the SAME scene
under different degradations are more similar to each other than
restorations of DIFFERENT scenes under the same degradation) -- exactly
the signature of successful, content-preserving restoration.

RAW_LOW -- FOURTH INDEPENDENT CONFIRMATION
Exactly 33.33% classification accuracy (zero variance) and
degradation_ratio undefined (0/0, both distances exactly zero) for all 3
AFLBs. TEST01 (direct forward-pass trace) + TEST02 (dataset-confounded
linear probe) + TEST03 linear probe + TEST03 distance analysis now FOUR
independently-derived confirmations that this tensor is a constant zero at
benchmark resolution.

ALPHA/BETA
62.0-62.7% (vs. 33.3% random) under grouped CV -- consistent with TEST02's
64-66%, confirming alpha/beta carry real but non-dominant degradation
signal even with the dataset confound removed.

SHEETS
  README                        this sheet
  Scene_Manifest                 100-scene manifest (clean/rain/haze/noise paths, dimensions)
  Degradation_Parameters         exact, documented synthesis parameters for all 300 images
  Data_Validation                Phase-4 same-scene validation checks (all 100 scenes passed)
  Feature_Index                   index of raw .pt tensors (10 representative scenes x 3 degradations x 41 features)
  Feature_Statistics              mean/std/min/max/L1/L2/energy for all 41 features x 300 images
  Linear_Probe                    grouped-CV accuracy/balanced-accuracy/macro-F1/precision/recall, 41 features x 2 classifiers
  Feature_Trajectory               the headline controlled trajectory table
  Paired_Distances                 raw same-scene and cross-scene pairwise distances (24,600 rows)
  Scene_vs_Degradation             degradation_ratio per feature -- the core new TEST03 evidence
  Alpha_Beta                       raw alpha/beta per image x AFLB (900 rows)
  AFLB_Analysis                    focused ranking of AFLB-internal sub-features (grouped-CV + degradation_ratio)
  Raw_Low_Check                    explicit raw_low degeneracy re-verification (4th confirmation)
  PSNR_SSIM                        restoration quality vs. ORIGINAL CLEAN image (not synthetic degraded image)
  TEST02_vs_TEST03                 direct trajectory comparison (TEST02 read-only reference, not modified)
  Environment                      reproducibility record (git SHA, checkpoint hash, versions, seeds, synthesis method)
  Tensor_Index                     index of all raw .pt tensors + representation-swap-prep index (Phase 16, NOT executed)

FULL TENSORS
  results/tensors/<feature_name>/<degradation>/<scene_id>.pt (float16), 10 representative
  scenes x 3 degradations x 41 features. A dedicated subset (5 scenes x
  {latent, AFLB1/2/3 output} x 3 degradations = 60 tensors) is indexed
  separately in representation_swap_prep_index.csv as infrastructure for a
  FUTURE representation-swap intervention -- NOT executed in TEST03 per
  the task's explicit instruction.

REPRODUCE
  cd test03/scripts && conda activate adair-distill
  (run build_scenes.py, validate_synthetic_data.py, then the src/ scripts in order --
   see test03_design.md and the script docstrings for the exact sequence)
"""

out_path = TEST03 / "results" / "README.csv"
pd.DataFrame({"README": readme_text.split("\n")}).to_csv(out_path, index=False)
print(f"wrote {out_path}")
