#!/usr/bin/env bash
# Keep one periodic checkpoint of a running TPU job past the trainer's retention (it keeps the two
# newest): wait until every host has finished writing <run>/ckpt/step_<n>, then hard-link it to
# <run>/ckpt_keep/step_<n> on every host (no extra disk; the trainer's later deletion of the
# original leaves the links). Used for the mid-training frame probe (Phase 08 diagnostics).
# Usage: scripts/tpu/keep_checkpoint.sh runs/main/gimbal/seed2 1000
set -u
RUN=/home/tasma/gimbal/$1
STEP=$(printf "step_%06d" "$2")
done_on() {  # host -> 0 when that host's process directory has its done marker
  if [ "$1" = local ]; then ls "$RUN"/ckpt/"$STEP"/process_*/done >/dev/null 2>&1
  else ssh "$1" "ls $RUN/ckpt/$STEP/process_*/done >/dev/null 2>&1"; fi
}
[ -d "$RUN/ckpt_keep/$STEP" ] && { echo "already kept: $1 $STEP"; exit 0; }
[ -f "$RUN/final.json" ] && { echo "run finished before $STEP was kept: $1"; exit 1; }
until done_on local && done_on w1 && done_on w2 && done_on w3; do sleep 20; done
mkdir -p "$RUN/ckpt_keep" && cp -al "$RUN/ckpt/$STEP" "$RUN/ckpt_keep/"
for h in w1 w2 w3; do ssh "$h" "mkdir -p $RUN/ckpt_keep && cp -al $RUN/ckpt/$STEP $RUN/ckpt_keep/"; done
echo "$(date -u +%H:%M:%S) kept $1 $STEP"
