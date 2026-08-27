#!/bin/bash
# TEST18: train all 5 AdaIR ablation variants SEQUENTIALLY (AdaIR is
# ~28.8M params, far larger than this project's usual ~7.4M-param NAFNet-M
# student -- concurrent multi-variant training, the pattern used
# throughout TEST07-17, does not fit on one RTX 4090 for a model this
# size). Real 3-in-1 degradation data (dehaze: 10k-image real OTS
# subsample; derain: full real Rain100L/RainTrainL 200 pairs; denoise:
# 100 real DIV2K clean images w/ online Gaussian noise), 8 epochs each
# (rescoped down from an initial 30-epoch target after a smoke test
# showed the full real-scale epoch would take ~21hrs/variant even with
# AMP -- see TEST18_PLAN.md's timing-calibration note), batch_size=8,
# AMP enabled, checkpointed every epoch so nothing is lost if this needs
# to be interrupted.
set -e
cd ~/teacher-experiments/test18/scripts
source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill
mkdir -p ~/teacher-experiments/test18/logs

for variant in A_baseline B_fixed_mask C_learned_mask D_plus_lh E_full; do
  echo "=== STARTING $variant at $(date) ==="
  python3 train_variant.py --variant "$variant" --epochs 8 --batch_size 8 --num_workers 8 \
    > ~/teacher-experiments/test18/logs/train_${variant}.log 2>&1
  echo "=== FINISHED $variant at $(date), exit=$? ==="
done
echo "ALL 5 VARIANTS COMPLETE at $(date)"
