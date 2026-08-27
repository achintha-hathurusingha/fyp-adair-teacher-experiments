#!/usr/bin/env bash
# Reproduces the complete TEST02 degradation-representation analysis.
# Run FROM test02/scripts inside the adair-distill conda env.
#
#   cd teacher-experiments/test02/scripts
#   conda activate adair-distill
#   ./run_test02.sh
#
# devon (192.248.10.68) has unreliable logical CPUs 8-11 -- every step is
# pinned with `taskset -c 0-7,12-31` (harmless on other hosts).

set -euo pipefail
TASKSET="taskset -c 0-7,12-31"

echo "== Phase 17: environment record =="
$TASKSET python write_environment.py

echo "== Phase 2: dataset manifest =="
$TASKSET python build_manifest.py

echo "== Phase 3/4/5: feature extraction (300 images, pooled + 15-image raw tensors) =="
$TASKSET python ../src/extract_features.py

echo "== Phase 13a: alpha/beta scalar extraction =="
$TASKSET python ../src/extract_alpha_beta.py

echo "== Phase 6: distance analysis =="
$TASKSET python ../src/distance_analysis.py

echo "== Phase 7-9/15: linear probes, feature trajectory, confusion matrices =="
$TASKSET python ../src/linear_probe.py

echo "== Phase 12: AFLB-specific summary =="
$TASKSET python ../src/build_aflb_summary.py

echo "== Phase 13b: alpha/beta distributions + classifier =="
$TASKSET python ../src/alpha_beta_analysis.py

echo "== Phase 10-11: PCA / t-SNE / UMAP =="
$TASKSET python ../src/dimensionality_reduction.py

echo "== Phase 14: correlate features with PSNR/SSIM =="
$TASKSET python ../src/correlate_psnr_ssim.py

echo "Done. CSVs are in ../results/{statistics,classifiers}; render the"
echo ".xlsx on a separate (non-flaky) machine with build_excel_local.py."
