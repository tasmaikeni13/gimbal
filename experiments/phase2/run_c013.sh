#!/bin/sh
# Re-runs after change C-013 (frame statistics on the empirical-Bayes innovation; F-020): every
# Phase 02 experiment that executes the optimizer, outputs tagged _c013 (earlier outputs are kept
# and stay in the analyses as the "before C-013" ablation). E2.5 first (it found F-020) on fresh
# evaluation seeds; the CPU benchmark last, alone on the machine. E2.10 and E2.11 do not execute
# the optimizer's update (static shrinkage function, population flow) and are not re-run.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "E2.5 noisy quadratics, evaluation seeds 300-311"
python e25_noisy_quadratic.py --workers 4 --eval-offset 300 --tag _c013 > results/e25_c013.log 2>&1
log "E2.1-E2.4 Gimbal rows"
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 \
    --methods gimbal,gimbal_k4 --tag _c013 > "results/e21_${suite}_c013.log" 2>&1
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 --extend \
    --methods gimbal,gimbal_k4 --tag _c013 > "results/e21_${suite}_ext_c013.log" 2>&1
done
log "E2.2b"; python e22b_tie_consistency.py --workers 4 --tag _c013 > results/e22b_c013.log 2>&1
log "E2.9";  python e29_ablations.py --workers 4 --tag _c013 > results/e29_c013.log 2>&1
log "E2.12"; python e212_numerics.py --tag _c013 > results/e212_c013.log 2>&1
log "E2.8";  OMP_NUM_THREADS=4 python e28_cost_model.py --bench --tag _c013 \
  > results/e28_c013.log 2>&1
log "done"
