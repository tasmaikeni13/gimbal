# Research state — Gimbal — updated 2026-10-06 (end of Phase 02)

## Contract
Target claim: Gimbal reaches lower validation loss/perplexity than SOAP and its peers at 125M
parameters / 2.5B FineWeb-Edu tokens, with step time no worse than SOAP's, on TPU v4-32.
Metrics and trade-offs: final validation loss (primary), wall-clock to target, step time, memory.
Regime: dense decoder LM, ≈1× Chinchilla, batch ≈ 0.5M tokens.
Success / kill thresholds: Phase 08 decision rule; Phase 02 gates G2.1–G2.6 (formal, numerical,
statistical and Monte Carlo evidence, no model training — C-006).
Locked final test: FineWeb-Edu test split (Phase 05), read once in Phase 08.

## Baseline
Not yet reproduced at scale (Phase 05). Phase 02 compares against reference implementations of
every peer on synthetic problems with known ground truth.

## Current evidence
- Phase 01 passed: theory, Lean formalization, counterexample battery (E1.1).
- Phase 02 passed (2026-10-06): algorithm C-004 + C-008 + C-012 (`frame_every = 4`) + C-013
  (frame statistics on the empirical-Bayes innovation); theory v0.7; 53 Lean theorems on standard
  axioms. All gates G2.1–G2.6 pass (`experiments/phase2/report.md`); claims C4–C7 in
  `research/ledger/claims.md`. Failures F-009 … F-023 recorded with mechanisms.
- Known limits: KL-SOAP's frame is slightly better on the flattest separable spectra (within the
  non-inferiority margin; theory §7); the default's optimizer state exceeds SOAP's by 2mn per
  layer (F-023); no language-model evidence yet.

## Live hypotheses
- H1 Gimbal (main). Reserves: H2 transport (ablation in E2.9), H5 Lie-algebra momentum, H6 robust
  (Student-t) score.

## Killed hypotheses
- none

## Next actions (user-run phases)
1. Phase 03: JAX/Optax implementations with golden tests against the PyTorch reference; then the
   small-LM validation moved there by C-006: E3.1 real-gradient premise check (G3.4) and E3.2
   small-LM benchmark (G3.5). Scripts ready in `experiments/phase3/`.
2. Phases 04–10 as specified in `phases/`.
