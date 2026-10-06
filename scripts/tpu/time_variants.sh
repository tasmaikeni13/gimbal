#!/usr/bin/env bash
# Short runs (400 steps of the confirmatory configuration) to measure steady-state step times.
set -uo pipefail
cd /home/tasma/gimbal
export GIMBAL_CODE=$(scripts/tpu/snapshot.sh timing_$(date +%Y%m%d_%H%M%S))
T="python -m gimbal.train.train --steps 10172 --stop-after 400 --no-full-eval --seed 0"
run() { local name=$1; shift; scripts/tpu/launch.sh runs/logs/timing/$name "$T --run-dir runs/timing/$name $*"; echo "$name exit=$?"; }
run adamw --optimizer adamw
run soap --optimizer soap
run gimbal_k4_a02 --optimizer gimbal
run gimbal_k4_a05 --optimizer gimbal --set optimizers.gimbal.rot_rate=0.05
run gimbal_adapt_a05 --optimizer gimbal --set optimizers.gimbal.rot_rate=0.05 --set optimizers.gimbal.frame_schedule=adaptive
run gimbal_k2_a05 --optimizer gimbal --set optimizers.gimbal.rot_rate=0.05 --set optimizers.gimbal.frame_every=2
run gimbal_k1_a05 --optimizer gimbal --set optimizers.gimbal.rot_rate=0.05 --set optimizers.gimbal.frame_every=1
.venv/bin/python benchmarks/tpu/step_times.py runs/timing/adamw runs/timing/soap runs/timing/gimbal_k4_a02 runs/timing/gimbal_k4_a05 runs/timing/gimbal_adapt_a05 runs/timing/gimbal_k2_a05 runs/timing/gimbal_k1_a05
