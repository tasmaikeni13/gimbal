#!/usr/bin/env bash
# Copy the code (not data, results or the venv) from worker 0 to workers 1-3 before a launch.
set -euo pipefail
cd /home/tasma/gimbal
for h in w1 w2 w3; do
  rsync -a src configs scripts tests pyproject.toml "$h":/home/tasma/gimbal/ &
done
wait
