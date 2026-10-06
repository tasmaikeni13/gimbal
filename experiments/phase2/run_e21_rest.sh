#!/bin/sh
# E2.1 suites after the main suite (tie, drift, tails), then the equal-treatment grid extension of
# all four suites. Single-threaded BLAS per worker.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for suite in tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 \
    > "results/e21_${suite}.log" 2>&1
done
./run_e21_ext.sh
