#!/bin/sh
# Re-runs after C-011 (KL-SOAP initialization): KL rows of E2.1-E2.4, then E2.5 and E2.10, then
# the analyses and the report.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "E2.1 KL rows";  ./run_e21_peerfix.sh
log "E2.5";          python e25_noisy_quadratic.py --workers 4 > results/e25.log 2>&1
log "E2.10";         python e210_theory_vs_simulation.py > results/e210.log 2>&1
log "analyses";      python analyze_e21.py > results/analyze_e21.log 2>&1
python analyze_e25.py > results/analyze_e25.log 2>&1
python make_report.py > results/make_report.log 2>&1
log "done"
