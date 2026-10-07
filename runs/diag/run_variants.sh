#!/usr/bin/env bash
# F-032 diagnosis: Gimbal's early instability at lr 3e-3 (tuning code snapshot, seed 0, the tuning
# schedule of 2,500 steps stopped after 300 steps). One setting changed per variant.
set -u
cd /home/tasma/gimbal
export GIMBAL_CODE=/home/tasma/gimbal_snapshots/tuning
BASE="python -m gimbal.train.train --optimizer gimbal --lr 0.003 --seed 0 --steps 2500 --stop-after 300 --no-full-eval"
run() {
  name=$1; shift
  echo "$(date +%H:%M:%S) start $name"
  scripts/tpu/launch.sh runs/logs/diag/$name "$BASE --run-dir runs/diag/$name $*"
  echo "$(date +%H:%M:%S) end $name exit=$?"
}
run d1_frozen_frame --set optimizers.gimbal.max_rotation=0.0
run d2_b1_095 --set optimizers.gimbal.b1=0.95
run d3_maxrot_01 --set optimizers.gimbal.max_rotation=0.1
run d4_every_step --set optimizers.gimbal.frame_schedule=fixed --set optimizers.gimbal.frame_every=1
echo "all done"
