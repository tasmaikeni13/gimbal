# Phase status

| Phase | State | Date | Commit | Gate evidence | Open failures |
|---|---|---|---|---|---|
| 01 Formal theory and Lean | **passed** | 2026-10-06 | see git log ("Phase 01") | G1.1 `lake build` ok, 37 theorems on standard axioms only (`formal/audit/axioms_output.txt`); G1.2 labels in `theory/gimbal_theory.md`; G1.3 `factor_estimators_strictly_inefficient`, `pooled_ge_mle`, `tie_but_identifiable`; G1.4 `experiments/phase1/results/check_identities.json` all pass; G1.5 theory §6 | none (F-001…F-005 closed) |
| 02 Math, statistics, Monte Carlo | **passed** (re-verified after C-015) | 2026-10-06 | see git log ("Phase 02", "C-015") | G2.1–G2.6 pass for the default after C-013 (`experiments/phase2/report.md`, generated); G2.4 failed before C-013 and passed on fresh seeds after it (F-020); 53 Lean theorems on standard axioms (`formal/audit/axioms_output.txt`); theory v0.7 | F-023 (state of the default above SOAP's; not a Phase 02 gate, for Phases 03–04); open item: flat separable spectra (theory §7) |
| 03 Reference implementations | **in_progress** | 2026-10-06 | 6c91c81 (entry) | entry state reproduced (22 tests pass); JAX AdamW/SOAP/Gimbal with golden and invariant tests (C-016); C-015 fix with Phase 02 re-verified | F-024, F-025, F-026 closed |
| 04 TPU kernels and distribution | pending | | | | |
| 05 Data and pipeline | pending | | | | |
| 06 Fair tuning | pending | | | | |
| 07 Competitive runs | pending | | | | |
| 08 Analysis and decision | pending | | | | |
| 09 Paper | pending | | | | |
| 10 Release | pending | | | | |
