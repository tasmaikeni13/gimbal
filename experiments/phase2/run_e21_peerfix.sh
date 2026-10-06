#!/bin/sh
# Re-run the KL-SOAP rows and the exact KL-factor control of every E2.1 suite (base grid and
# extension) after the peer fix C-011 (identity initialization of the KL factors, F-018).
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for suite in main tie drift tails; do
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 \
    --methods klsoap,kl_eigh --tag _peerfix > "results/e21_${suite}_peerfix.log" 2>&1
  python e21_frame_efficiency.py --suite "$suite" --seeds 10 --steps 800 --workers 4 --extend \
    --methods klsoap,kl_eigh --tag _peerfix > "results/e21_${suite}_ext_peerfix.log" 2>&1
done
