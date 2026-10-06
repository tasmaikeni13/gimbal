#!/usr/bin/env bash
# Freeze the code for a set of TPU runs: copy src/, configs/, scripts/ and benchmarks/ to
# ~/gimbal_snapshots/<name> on all four hosts (data/, runs/ and the compilation cache are
# symlinked to the repository), and record the commit and diff it came from. Runs launched from a
# snapshot are not affected by later edits of the working tree.
# Usage: scripts/tpu/snapshot.sh <name>   (prints the snapshot path)
set -euo pipefail
NAME=$1
REPO=/home/tasma/gimbal
SNAP=/home/tasma/gimbal_snapshots/$NAME
mkdir -p "$SNAP"
rsync -a --exclude __pycache__ "$REPO"/src "$REPO"/configs "$REPO"/scripts "$REPO"/benchmarks \
  "$REPO"/pyproject.toml "$SNAP"/
for d in data runs .jax_cache; do ln -sfn "$REPO/$d" "$SNAP/$d"; done
git -C "$REPO" rev-parse HEAD > "$SNAP/SNAPSHOT_COMMIT"
git -C "$REPO" diff HEAD -- src configs > "$SNAP/SNAPSHOT_DIFF"
for h in w1 w2 w3; do
  ssh "$h" "mkdir -p $SNAP" && rsync -a "$SNAP"/ "$h":"$SNAP"/ &
done
wait
echo "$SNAP"
