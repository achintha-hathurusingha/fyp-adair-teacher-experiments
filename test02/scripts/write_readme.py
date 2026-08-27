"""Writes results/README.csv for the TEST02 workbook. Run LOCALLY."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TEST02 = Path(__file__).resolve().parent.parent

readme_text = """TEST02 -- Degradation Representation Analysis of AdaIR
Research question: where does the released 3-degradation AdaIR checkpoint
encode degradation-specific (Rain/Haze/Noise) information, and how strongly,
at each network stage?
Full report: report/test02_report.md
Source audit: report/source_audit.md

DIRECTORY RULE
Everything for this experiment lives under test02/. The original AdaIR
source, checkpoint, and test01 were read-only references and were NOT
modified, overwritten, or deleted.

MODEL
Released, UNMODIFIED AdaIR (decoder=True), adair3d.ckpt (3-degradation
all-in-one teacher), 28,784,824 params, 0 missing/0 unexpected keys. No
retraining, no architecture change, no preprocessing change -- the only
addition is non-intrusive forward hooks (identical mechanism to test01's
instrumentation, reused unmodified).

EXPLICIT DEGRADATION MECHANISM SEARCH (source_audit.md)
AdaIR.forward(self, inp_img, noise_emb=None) -- `noise_emb` is the ONLY
parameter besides the image, appears exactly ONCE in the entire 475-line
source file (its own signature), and is NEVER read or used anywhere in the
forward pass. NO explicit degradation label, embedding, classifier, prompt,
or routing mechanism exists anywhere in the released implementation --
confirmed by direct source inspection, not inferred from the paper. Any
degradation-type information the network uses must be implicitly derived
from the image content itself.

DATASET
Identical 300-image manifest to test01/the original analysis (100 Rain100L
/ 100 SOTS-outdoor / 100 BSD68). Degradation labels (Rain=0/Haze=1/Noise=2)
are used ONLY for external post-hoc analysis -- never fed into AdaIR.

METHOD
41 intermediate representations extracted per image (11 global pipeline
stages [input/shallow/encoder1-3/latent/decoder1-3/refinement/output] + 3
AFLB x 10 internal sub-features [y_in, raw_low/high, mined_low/high, H-L,
L-H, FMoM-agg, cross-attn-out, AFLB-output]), pooled via GAP+GMP
concatenation into compact vectors. For every feature: a Logistic
Regression AND a Linear SVM (both EXTERNAL to AdaIR -- not part of the
model) probe Rain/Haze/Noise separability via 5-fold stratified
cross-validation. Random baseline = 33.3%.

HEADLINE RESULT -- THE DEGRADATION-INFORMATION TRAJECTORY
(logistic regression, 5-fold CV accuracy; see Feature_Trajectory sheet /
results/visualizations/degradation_information_trajectory.png)

  Input                    71.7%  (+-3.8)   -- raw degraded pixels already carry substantial signal
  Shallow (Y0)             93.3%  (+-1.5)
  Encoder L1               98.0%  (+-2.4)
  Encoder L2               99.3%  (+-0.8)
  Encoder L3               99.3%  (+-1.3)
  Latent                  100.0%  (+-0.0)   <- PEAK, perfect separation
  AFLB 1                   99.7%  (+-0.7)
  Decoder L3 (post-AFLB1)  99.0%  (+-1.3)
  AFLB 2                   99.0%  (+-1.3)
  Decoder L2 (post-AFLB2)  99.7%  (+-0.7)
  AFLB 3                   99.7%  (+-0.7)
  Decoder L1 (post-AFLB3)  99.7%  (+-0.7)
  Refinement               99.7%  (+-0.7)
  Output (restored)        54.7%  (+-4.5)   <- COLLAPSES back down near input level

OBSERVATION: a linear classifier can distinguish Rain/Haze/Noise from
essentially every internal representation from Encoder L2 onward with
>99% accuracy, but from the final restored output with only 54.7%.
INFERENCE: internal representations carry near-perfect degradation-
discriminative information; the final output layer removes most (not all)
of it. INFERENCE, not proven causally: this is consistent with AdaIR using
strong internal degradation-awareness to condition its restoration
behavior, then converging toward a degradation-agnostic "clean image"
representation at the output -- exactly what a well-behaved blind
restoration network should do. We do NOT claim AdaIR "detects" or "knows"
the degradation type in any symbolic sense; we observe that a simple
external linear probe can decode it from the feature statistics.

CONTROL: INPUT-ONLY BASELINE
71.7% from raw pixels ALONE (before any AdaIR computation) confirms the
three degradation types are visually distinct enough that some separability
is expected even trivially -- but every internal stage from Encoder L1
onward substantially exceeds this baseline (98%+), showing AdaIR actively
AMPLIFIES the input's latent degradation signal through its encoder, not
merely passing it through.

ALPHA/BETA (MGB) -- PARTIAL, NOT DOMINANT, SIGNAL
[alpha, beta] alone (2 numbers per AFLB) classify degradation at 64-66%
accuracy (vs. 33.3% random) -- meaningfully above chance, but far below the
93-100% every full feature representation achieves. Alpha/beta are NOT
degradation labels and do not on their own explain how AdaIR conditions
restoration; see Alpha_Beta_Probe sheet.

AFLB-INTERNAL FINDING (corroborates test01)
`raw_low` (the pre-cross-attention low-frequency FFT-split feature) scores
EXACTLY 33.33% accuracy with ZERO variance across all 3 AFLBs -- the
mathematically expected result for a constant zero vector (test01
independently proved raw_low is exactly zero at every AFLB, every image, at
these resolutions). This is a clean, mutually-corroborating null result
between test01 and test02, using entirely different analysis methods.
Every OTHER AFLB sub-feature (mined_low/high, FMoM-agg, cross-attn-out,
AFLB-output, y_in) scores 88-100%.

SHEETS
  README                    this sheet
  Dataset                    300-image manifest (Rain=0/Haze=1/Noise=2 labels, analysis-only)
  Feature_Index               index of raw .pt tensors (15 representative images x 41 features)
  Feature_Statistics          mean/std/min/max/L1/L2/energy for all 41 features x 300 images (12,300 rows)
  Linear_Probe                accuracy/balanced-accuracy/macro-F1/precision/recall, all 41 features x 2 classifiers
  Distance_Analysis           intra/inter-class Euclidean+cosine distances, separation ratios, all 41 features
  Alpha_Beta                  raw alpha/beta per image x AFLB (900 rows)
  Alpha_Beta_Probe            [alpha,beta]-only classifier accuracy (Phase 13)
  AFLB_Analysis                focused ranking of AFLB-internal sub-features by degradation separability
  PSNR_SSIM                   per-image restoration quality (needed for Phase 14 correlation)
  PSNR_SSIM_Correlation        exploratory Pearson/Spearman correlation, feature stats vs PSNR/SSIM (NOT causal)
  Feature_Trajectory           the headline trajectory table (see above)
  Confusion_Matrices           per-class confusion for input / best feature / output
  Tensor_Index                 index of all raw .pt tensors saved
  Environment                  reproducibility record (git SHA, checkpoint hash, versions, seeds)

FULL TENSORS
  results/tensors/<feature_name>/<degradation>/<Image_ID>.pt (float16), 15 representative
  images (5/degradation) x 41 features. Indexed by Feature_Index / Tensor_Index sheets.

REPRODUCE
  cd test02/scripts && conda activate adair-distill && ./run_test02.sh
"""

out_path = TEST02 / "results" / "README.csv"
pd.DataFrame({"README": readme_text.split("\n")}).to_csv(out_path, index=False)
print(f"wrote {out_path}")
