#!/bin/sh
# Re-runs after change C-019 (default momentum beta_1 = 0.95, SOAP's; F-032): every Phase 02
# experiment that executes Gimbal with its default momentum, outputs tagged _c019, with the seeds of
# the earlier re-runs so that the before/after comparison is paired (the repair was found by the
# Phase 06 TPU runs, not by these seeds). E2.5 sets beta = (0.9, 0.95) explicitly for every method
# (matched momentum), so its _c018 outputs stand; E2.10 and E2.11 do not execute the update; E2.8's
# cost model does not depend on beta_1. Earlier outputs are kept.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-60}
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "E2.1-E2.4 Gimbal rows"
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers "$W" \
    --methods gimbal,gimbal_k4 --tag _c019 > "results/e21_${suite}_c019.log" 2>&1
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers "$W" --extend \
    --methods gimbal,gimbal_k4 --tag _c019 > "results/e21_${suite}_ext_c019.log" 2>&1
done
log "E2.2b"; python e22b_tie_consistency.py --workers "$W" --tag _c019 > results/e22b_c019.log 2>&1
log "E2.9";  python e29_ablations.py --workers "$W" --tag _c019 > results/e29_c019.log 2>&1
log "E2.12"; python e212_numerics.py --tag _c019 > results/e212_c019.log 2>&1
log "done"
