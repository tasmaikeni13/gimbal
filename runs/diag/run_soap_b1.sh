#!/usr/bin/env bash
# F-032 diagnosis, variant D6: SOAP with Gimbal's momentum (b1 = 0.9) at lr 3e-3 (tuning code
# snapshot, seed 0, 300 of 2,500 steps). Waits for D5.
set -u
cd /home/tasma/gimbal
until grep -q "end d5_transport" runs/logs/diag/transport.log; do sleep 10; done
export GIMBAL_CODE=/home/tasma/gimbal_snapshots/tuning
echo "$(date +%H:%M:%S) start d6_soap_b1_09"
scripts/tpu/launch.sh runs/logs/diag/d6_soap_b1_09 "python -m gimbal.train.train --optimizer soap --lr 0.003 --seed 0 --steps 2500 --stop-after 300 --no-full-eval --run-dir runs/diag/d6_soap_b1_09 --set optimizers.soap.b1=0.9"
echo "$(date +%H:%M:%S) end d6_soap_b1_09 exit=$?"
