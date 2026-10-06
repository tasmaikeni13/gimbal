# Research state — Gimbal — updated 2026-10-06 (Phases 03–06)

## Contract
Target claim: Gimbal reaches lower validation loss than SOAP (and AdamW) at 125M parameters /
2.5B FineWeb-Edu tokens, with step time no worse than SOAP's, on TPU v4-32 (D-003: the TPU study
compares AdamW, SOAP f = 10 and Gimbal). Decision rule: Phase 08 (D-005 definitions).

## Current evidence
- Phases 01–02 passed; re-verified after C-015 and C-018 (`experiments/phase2/report.md`).
- Phase 03: JAX implementations verified (float64 agreement ~1e-12 per step); C-015 fixed a gauge
  dependence of the trust region (F-024); the default lost the small-LM gate G3.5 (F-027) and was
  repaired by C-018 (α = 1 − β₂ = 0.05, adaptive amortization), after which Gimbal is lower than
  SOAP and AdamW on every fresh seed (small margin over SOAP). **Premise failure** (G3.4, F-030):
  κ ≈ 0.005 on real small-LM gradients; the likelihood frame still beats SOAP's pooled frame on
  held-out gradients (efficiency under separable spectra), but KL-SOAP beats Gimbal and SOAP
  real-time ties it: H1's kill criterion is met at small scale (D-006).
- Phase 04: Gimbal's training step is 1.9% slower than SOAP's on the v4-32 (G4.2 fails).
- Phase 05 passed (pipeline validated against GPT-2's published validation loss).
- Phase 06 sweep running.

## Live hypotheses
- H1 at 125M vs AdamW and SOAP (the confirmatory test). Reserves: H2 transport, H5 Lie momentum,
  H6 robust score.

## Killed hypotheses
- H1's non-separability mechanism at small scale (kill criterion met; see portfolio).

## Next actions
1. Finish Phase 06; freeze configs; tag `tuning-frozen`.
2. Phase 07 runs (seeds 2, 3); Phase 08 analysis by D-005, decision, test split once.
3. Phase 09 paper (claims bounded by F-030 and the decision), Phase 10 release.
