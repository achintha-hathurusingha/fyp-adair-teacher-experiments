#!/usr/bin/env bash
# Reproduces the complete ablation study end to end. Run FROM this directory
# (test01/scripts) inside the adair-distill conda env, on a host where the
# flaky-core pin below is either applicable or a no-op (harmless elsewhere).
#
#   cd teacher-experiments/test01/scripts
#   conda activate adair-distill
#   ./run_all_experiments.sh
#
# devon (192.248.10.68) has unreliable logical CPUs 8-11 that silently
# corrupt in-process data -- every step is pinned with `taskset -c 0-7,12-31`.
# On a different machine this taskset invocation is harmless (just pins to
# a wide, always-valid core range) but can be removed if truly unnecessary.

set -euo pipefail
TASKSET="taskset -c 0-7,12-31"

echo "== git / environment record =="
git -C ../../AdaIR rev-parse HEAD > ../results/git_sha_adair.txt
python -V > ../results/python_version.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" >> ../results/python_version.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ../results/gpu_info.txt

echo "== Phase 1/2/6/12: three-condition ablation (300 images x 3 variants) =="
$TASKSET python run_ablation.py

echo "== Phase 7: paired statistical analysis =="
$TASKSET python analyze_results.py

echo "== Phase 9: resolution sweep (3 variants, square + rectangular) =="
$TASKSET python resolution_sweep_variants.py

echo "== Phase 11 (derived sheets): diff tables, mechanism audit, tensor index =="
$TASKSET python build_derived_sheets.py

echo "== Phase 10: visualizations =="
$TASKSET python build_visualizations.py

echo "Done. CSVs are in ../csv_export and ../results -- render the .xlsx on a"
echo "separate (non-flaky) machine with build_ablation_excel_local.py."
