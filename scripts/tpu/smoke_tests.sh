#!/usr/bin/env bash
# Phase 05 smoke tests on the full slice; results checked by scripts/tpu/check_smoke.py.
set -uo pipefail
# Run once on a clean runs/smoke (logs are appended).
cd /home/tasma/gimbal
L=runs/logs/smoke
T="python -m gimbal.train.train"
run() { local name=$1; shift; scripts/tpu/launch.sh $L/$name "$T $* --run-dir runs/smoke/$name"; echo "$name exit=$?"; }
# 1. overfit 977 sequences (1.0M tokens) with the full model
run overfit --optimizer adamw --lr 1e-3 --steps 400 --no-full-eval --set data.subset_seqs=977 --set eval.every=400
# 2. zero learning rate: the validation loss must not change
run zero_lr --optimizer adamw --lr 0 --steps 20 --no-full-eval --eval-at-start --set eval.every=20
# 3. 200 AdamW steps at the full configuration; 4. determinism (two identical runs)
run adamw_200a --optimizer adamw --seed 0 --stop-after 200 --no-full-eval
run adamw_200b --optimizer adamw --seed 0 --stop-after 200 --no-full-eval
# 5. resume: checkpoint, then restart from it and compare the next 20 steps
run resume_adamw --optimizer adamw --seed 0 --stop-after 40 --no-full-eval --set log.ckpt_every=20
run resume_adamw --optimizer adamw --seed 0 --stop-after 40 --no-full-eval --set log.ckpt_every=20 --resume
run resume_gimbal --optimizer gimbal --seed 0 --stop-after 80 --no-full-eval --set log.ckpt_every=60
run resume_gimbal --optimizer gimbal --seed 0 --stop-after 80 --no-full-eval --set log.ckpt_every=60 --resume
