#!/bin/bash
# TEST17: train all 12 runs (A/N/F2/NF2 x seeds 0,1,2) in two waves of 6
# concurrent processes each, mirroring TEST12's established-safe 9-way
# concurrency pattern (OMP/MKL=3 threads/proc) but more conservative since
# this is 12 runs total, and TEST11 showed 18-way concurrency OOMs on the
# RTX 4090 while TEST12's 9-way was safe.
set -e
cd ~/teacher-experiments/test17/scripts
source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill
mkdir -p ~/teacher-experiments/test17/logs

run_one() {
  local model=$1
  local seed=$2
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python3 train.py --model "$model" --seed "$seed" \
    > ~/teacher-experiments/test17/logs/train_${model}_seed${seed}.log 2>&1
  echo "FINISHED $model seed=$seed exit=$?"
}

echo "=== WAVE 1: A x {0,1,2}, N x {0,1,2} ==="
run_one A 0 &
run_one A 1 &
run_one A 2 &
run_one N 0 &
run_one N 1 &
run_one N 2 &
wait
echo "=== WAVE 1 COMPLETE ==="

echo "=== WAVE 2: F2 x {0,1,2}, NF2 x {0,1,2} ==="
run_one F2 0 &
run_one F2 1 &
run_one F2 2 &
run_one NF2 0 &
run_one NF2 1 &
run_one NF2 2 &
wait
echo "=== WAVE 2 COMPLETE ==="
echo "ALL 12 TRAINING RUNS DONE"
