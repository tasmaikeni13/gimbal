#!/bin/sh
# Grid extension for the four E2.1 suites (same seeds and streams; see MEMORY_EXTENSION).
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 --extend \
    > "results/e21_${suite}_ext.log" 2>&1
done
