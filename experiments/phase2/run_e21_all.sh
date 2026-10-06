#!/bin/sh
# Runs the four E2.1 suites sequentially with single-threaded BLAS (avoids oversubscription).
cd "$(dirname "$0")" || exit 1
mkdir -p results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 > "results/e21_${suite}.log" 2>&1
done
