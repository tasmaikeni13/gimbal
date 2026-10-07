#!/usr/bin/env bash
# F-032 diagnosis, variant D5: transport of Adam's and the flow's second moments at every frame
# move (gimbal.jax transport=True; snapshot "diag" = commit b7b70c6). Waits for D1-D4.
set -u
cd /home/tasma/gimbal
until grep -q "all done" runs/logs/diag/variants.log; do sleep 10; done
export GIMBAL_CODE=/home/tasma/gimbal_snapshots/diag
BASE="python -m gimbal.train.train --optimizer gimbal --lr 0.003 --seed 0 --steps 2500 --stop-after 300 --no-full-eval"
echo "$(date +%H:%M:%S) start d5_transport"
scripts/tpu/launch.sh runs/logs/diag/d5_transport "$BASE --run-dir runs/diag/d5_transport --set optimizers.gimbal.transport=true"
echo "$(date +%H:%M:%S) end d5_transport exit=$?"
