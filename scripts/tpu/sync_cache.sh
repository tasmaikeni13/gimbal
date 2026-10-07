#!/usr/bin/env bash
# Copy new entries of the persistent compilation cache from worker 0 to workers 1-3.
# In a multi-process job only process 0 writes cache entries, so with host-local cache directories
# workers 1-3 would recompile every program in every run (minutes per run for SOAP and Gimbal) while
# worker 0 waits for them at the first collective. Entries are immutable once written, and rsync
# renames each file into place only when complete, so a reader never sees a partial entry.
set -euo pipefail
CACHE=/home/tasma/gimbal/.jax_cache
for h in w1 w2 w3; do
  rsync -a --ignore-existing "$CACHE/" "$h:$CACHE/" &
done
wait
