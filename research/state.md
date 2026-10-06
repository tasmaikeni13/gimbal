# Research state — Gimbal — updated 2026-10-06

## Contract
Target claim: Gimbal reaches lower validation loss/perplexity than SOAP and its peers at 125M
parameters / 2.5B FineWeb-Edu tokens, with step time no worse than SOAP's, on TPU v4-32.
Metrics and trade-offs: final validation loss (primary), wall-clock to target, step time, memory.
Regime: dense decoder LM, ≈1× Chinchilla, batch ≈ 0.5M tokens.
Success / kill thresholds: Phase 08 decision rule; Phase 02 gates G2.1–G2.6.
Locked final test: FineWeb-Edu test split (Phase 05), read once in Phase 08.

## Baseline
Not yet reproduced at scale (Phase 05). Phase 02 uses CPU-scale baselines.

## Current evidence
- Phase 01 passed: theory v0.2, 37 Lean theorems, counterexample battery clean (E1.1).

## Live hypotheses
- H1 Gimbal (main), H2 transport (ablation), H5 Lie momentum, H6 robust score (reserves).

## Killed hypotheses
- none

## Next actions
1. Phase 02: E2.1 frame efficiency Monte Carlo with faithful SOAP/KL-SOAP frames.
2. Phase 02: E2.5 noisy-quadratic optimization with tuned learning rates.
3. Phase 02: E2.6/E2.7 small LM on FineWeb-Edu text (CPU).
