#!/usr/bin/env bash
# Run one command on all four hosts of the v4-32 slice at once (multi-host JAX needs every host).
# Usage: scripts/tpu/launch.sh <log-dir> <command...>
# Worker 0 runs locally; workers 1-3 over ssh (aliases w1-w3 in ~/.ssh/config). Each host writes
# <log-dir>/host<k>.log. The command runs from the repository root with the project venv.
set -euo pipefail
ROOT=/home/tasma/gimbal
LOGDIR=$1; shift
CMD="$*"
mkdir -p "$LOGDIR"
for h in w1 w2 w3; do ssh "$h" "mkdir -p $LOGDIR"; done
ENVSET="cd $ROOT && export PATH=$ROOT/.venv/bin:\$PATH && export PYTHONPATH=$ROOT/src:\${PYTHONPATH:-}"
pids=()
for k in 1 2 3; do
  ssh "w$k" "$ENVSET && export GIMBAL_WORKER=$k && $CMD" > "$LOGDIR/host$k.log" 2>&1 &
  pids+=($!)
done
bash -c "$ENVSET && export GIMBAL_WORKER=0 && $CMD" > "$LOGDIR/host0.log" 2>&1 &
pids+=($!)
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
