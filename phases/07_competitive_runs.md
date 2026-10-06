# Phase 07 — Competitive runs: 125M parameters, 2.5B FineWeb-Edu tokens, 2 seeds per optimizer

## Purpose

Run the confirmatory comparison under the frozen protocol and collect everything Phase 08 needs.

## Depends on

Phase 06 (`configs/frozen/`, tag `tuning-frozen`), Phase 05 pipeline, Phase 04 implementation.

## Produces

`runs/main/<optimizer>/seed<k>/` (JSONL logs, final metrics, step-time logs, environment, git hash),
checkpoints in GCS, `runs/main/manifest.csv` (one row per planned run with status).

## Run matrix

* **After D-003 (user, 2026-10-06):** AdamW, SOAP (f=10) and Gimbal, seeds 2 and 3, 10,172 steps of
  240 × 1024 tokens (2.49987B tokens) on all 16 chips. The lists below are the original plan.
* **Core (original):** AdamW, SOAP (f=10), SOAP real-time, KL-SOAP, Muon, NorMuon, SPlus, Gimbal.
* **Extended (if budget allows, decided before starting):** KL-Shampoo, ARO, Shampoo + grafting,
  COSMOS, PSGD-Kron.
* **Seeds:** 2 per optimizer, the same two seeds for every optimizer (paired initialization and data
  order). These seeds were not used in Phase 06.
* **Order:** interleave optimizers (e.g. seed 1 of all, then seed 2 of all) so slow hardware drift or
  pre-emptions do not align with one optimizer.

## Procedure per run

1. Launch from the frozen config and the tagged commit; write the manifest row as `running`.
2. Monitor: loss, NaN/Inf, gradient-norm spikes, step time, HBM.
3. On pre-emption or host failure: resume from the last checkpoint with identical config; log it.
4. On divergence (NaN/Inf, or loss > 2× its 100-step minimum for 50 steps): restart once from the last
   good checkpoint with identical config to rule out hardware faults. If it diverges again, the run is
   a valid, reported failure for that optimizer. Do not change hyperparameters of a frozen config.
5. On completion: evaluate the final checkpoint on the full validation set; write final metrics; mark
   the manifest row `done`.

## Gimbal-specific failure (the self-correcting path)

If Gimbal diverges, spikes, or is clearly behind on validation loss after both seeds:

1. Diagnose with logged diagnostics ($\|\Omega\|$, orthogonality defect, $\kappa$, $D$ floor hits) and by
   replaying saved gradients offline (protocol §4 steps 1–3).
2. If the cause is a bug: fix, re-run Phase 03 tests, re-run Gimbal's seeds only.
3. If the cause is the method: research and repair through protocol §4; the repair is a change-id
   under protocol §5 — rewrite the theory document, re-check affected Lean statements (Phase 01),
   re-run the affected Phase 02 gates on fresh seeds, re-tune Gimbal with the same Phase 06 budget,
   rewrite Phases 03–08 where they are affected, then re-run Gimbal's two seeds.
4. Competitor runs are reused unless the pipeline or evaluation changed.
5. The original failed Gimbal runs stay in the record and are reported.

## Exit gate

* G7.1 Every core run is `done` or a documented, twice-reproduced divergence.
* G7.2 Logs, configs, environment files and checkpoints are uploaded and listed in the manifest.
* G7.3 Total compute per optimizer (TPU-hours, including tuning) is recorded.
