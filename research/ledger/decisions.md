# Decisions and change log (protocol §5)

| ID | Date | Decision / change | Reason | Affected set | Re-verification |
|---|---|---|---|---|---|
| D-001 | 2026-10-06 | Select hypothesis H1 (Gimbal) as the main line | literature frontier (no overlap found), two prototype checks | all phases | Phase 01–02 |
| D-002 | 2026-10-06 | Repository layout and phase protocol (`phases/README.md`) | user request | — | — |
| C-001 | 2026-10-06 | Algorithm §5: spectral trust region ‖Ω‖₂ ≤ 1 per step; Newton–Schulz polish every step, repeated until the orthogonality defect is below working precision; transport rows renormalized | F-001, F-002 | `src/gimbal/torch/gimbal.py`, theory §5 and Thm 5 note, Phase 02 defaults, Phase 03–04 (cost of the polish) | unit tests pass; Lean results unaffected (they concern the exact retraction identities, which are unchanged); phases 03/04 already list the polish period as a tunable cost option |
