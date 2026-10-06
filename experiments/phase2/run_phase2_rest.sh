#!/bin/sh
# Phase 02 experiments after E2.1-E2.4 (run_e21_all.sh / run_e21_rest.sh), sequentially; each
# uses all cores. No language-model training here (C-006).
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "E2.10 theory vs simulation"; python e210_theory_vs_simulation.py > results/e210.log 2>&1
log "E2.11 landscape";            python e211_landscape.py --workers 4 > results/e211.log 2>&1
log "E2.12 numerics";             python e212_numerics.py > results/e212.log 2>&1
log "E2.5 noisy quadratics";      python e25_noisy_quadratic.py --workers 4 > results/e25.log 2>&1
log "E2.9 ablations";             python e29_ablations.py --workers 4 > results/e29.log 2>&1
log "E2.8 cost model";            OMP_NUM_THREADS=4 python e28_cost_model.py --bench > results/e28.log 2>&1
log "done"
