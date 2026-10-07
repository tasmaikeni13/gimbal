# Changelog

Changes to the method, its defaults and the research protocol. Every entry has a change-id
(C-…) or decision (D-…) in `research/ledger/decisions.md` with its reason, the affected phases and
the re-verification; failures that triggered a change are in `research/ledger/failures.md`.

## 1.0.0 — 2026-10-07

Release of the completed research program (phases 01–10): theory with machine-checked
statements, Monte Carlo study, PyTorch and JAX implementations, the 125M-parameter / 2.5B-token
comparison of AdamW, SOAP and Gimbal on a TPU v4-32, the analysis and the paper. Outcome: Gimbal
beats AdamW and matches SOAP within seed noise with a slightly slower step, so the headline claim
is not supported (`analysis/decision.md`); results in `README.md`, `analysis/results/` and
`paper/`.

### Method and defaults

* C-019 — in the 125M study Gimbal uses SOAP's momentum β₁ = 0.95; with Adam's 0.9 against
  SOAP's 0.95 the comparison was confounded, and both optimizers are unstable early at high
  learning rates with 0.9 (F-032). Library default unchanged.
* C-018 — rotation rate α = 1 − β₂ = 0.05 (one estimation window for the frame, its variances and
  Adam's second moment) and adaptive amortization of the frame moves (F-027).
* C-015 — the trust region's spectral-norm estimate starts from the largest column of Ω, so the
  update does not depend on the eigenvector sign gauge (F-024).
* C-013 — the frame is fitted to the innovation G − c·M with a James–Stein factor c (F-020).
* C-012 — amortized flow: the frame moves every 4 steps with the mean score (F-019).
* C-004 — empirical-Bayes shrinkage of the flow's variances toward their separable fit.
* C-003 — separate flow variances with the frame's memory; pooled warm start at step 50 (F-010).
* C-002 — bias-corrected rotation schedule; damping 0.003.
* C-001 — spectral trust region and Newton–Schulz polish of the frames (F-001, F-002).

### Implementation and infrastructure

* JAX implementations of AdamW, SOAP and Gimbal for the TPU study, verified against the PyTorch
  references (float64) and against the official SOAP; distributed step checked against
  per-matrix steps (`tests/test_distributed.py`).
* Gimbal's optional moment transport (Theorem 7) available in JAX as in the reference.
* Compilation cache shared across hosts (F-031); frozen code snapshots for every TPU run (F-028).

### Protocol

* D-003 — the TPU study compares AdamW, SOAP and Gimbal only, with the budgets set by the owner.
* D-005 — Phase 08 definitions fixed before the confirmatory runs.
* D-006 — the 125M comparison proceeds although H1's kill criterion was met at small scale.
* D-007 — MIT license.
* D-008 — repair rule for F-032, written before the last diagnostics were seen.
* D-009 — mid-training frame probe, declared before the confirmatory runs.
* D-010 — Stage C edge extension, written before the second Stage C seed was seen.
* D-011 — no further repair cycle after the Phase 08 decision; the negative result is reported.
* D-012 — humanize pass with a faithfulness audit.
