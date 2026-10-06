#!/bin/sh
# Re-runs after change C-015 (sign-equivariant start of the spectral-norm power iteration; F-024):
# every Phase 02 experiment that executes Gimbal's update, outputs tagged _c015, with the seeds of
# the C-013 runs so that the before/after comparison is paired (the repair was found by the Phase 03
# cross-framework test, not by these seeds). Earlier outputs are kept. E2.10 and E2.11 do not
# execute the optimizer's update; E2.8's cost model does not depend on the estimate.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-60}
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "E2.5 noisy quadratics, evaluation seeds 300-311"
python e25_noisy_quadratic.py --workers "$W" --eval-offset 300 --tag _c015 > results/e25_c015.log 2>&1
log "E2.1-E2.4 Gimbal rows"
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers "$W" \
    --methods gimbal,gimbal_k4 --tag _c015 > "results/e21_${suite}_c015.log" 2>&1
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers "$W" --extend \
    --methods gimbal,gimbal_k4 --tag _c015 > "results/e21_${suite}_ext_c015.log" 2>&1
done
log "E2.2b"; python e22b_tie_consistency.py --workers "$W" --tag _c015 > results/e22b_c015.log 2>&1
log "E2.9";  python e29_ablations.py --workers "$W" --tag _c015 > results/e29_c015.log 2>&1
log "E2.12"; python e212_numerics.py --tag _c015 > results/e212_c015.log 2>&1
log "done"
