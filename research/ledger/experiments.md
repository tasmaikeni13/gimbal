# Experiment records

Template: `ml-research` skill, `references/research-loop.md`. Raw outputs live next to the scripts.

## E1.1 — Counterexample battery (Phase 01)
* Question: do the identities and inequalities of the theory survive random and adversarial
  instances, and do the asymptotic formulas (delta method, Fisher, flow stationary variance) match
  simulation?
* Code: `experiments/phase1/check_identities.py` (quick and full modes).
* Result: all nine checks pass in both modes after fixing four defects in the checking code
  (F-002…F-005). Pooled-factor angle variance: predicted 1.636e-4, simulated 1.602e-4; weighted
  factor: 8.603e-4 vs 8.493e-4; Cramér–Rao 1.875e-5 (8.7× and 46× below). Flow stationary
  variance: 4.717e-4 predicted, 4.815e-4 simulated.
* Raw: `experiments/phase1/results/check_identities.json`, `check_identities_full.log`.
* Decision: Phase 01 gate G1.4 passed.

## E2.1–E2.4 — Frame-estimation efficiency, ties, drift, heavy tails (Phase 02, confirmatory)
* Question: does Gimbal's frame estimate beat every peer's (SOAP, SOAP real-time, KL-SOAP and
  exact pooled/KL eigenvector controls) on KRD streams, at each method's best memory?
* Design: paired streams, 10 seeds (10–19), 800 steps; every method at a memory grid, extended
  equally for all methods (`run_e21_ext.sh`); score = frame KL averaged over the second half;
  one-sided paired Wilcoxon with Holm correction per cell; bootstrap intervals; random-effects
  pooling across cells. Algorithm: C-004 defaults (C-008 fix bit-identical, 141/141 checks).
* Code: `experiments/phase2/e21_frame_efficiency.py`, `analyze_e21.py`.
* Result as first analyzed (KL-SOAP before its fix C-011, default before C-013): Gimbal below every
  peer and control in every cell of every suite. Superseded by the re-runs below.
* Result after C-011 (KL rows re-run) and C-013 (Gimbal rows re-run on the same seeds; earlier rows
  kept as `*_c012`): the default (`frame_every = 4`) is below every peer and both exact-eigenvector
  controls (Holm-corrected) in every non-separable, tie, drift and heavy-tail cell, at best and at
  matched memory. In the six separable cells it is below SOAP and SOAP real-time everywhere and
  below KL-SOAP except in the two flattest (γ = 0, s = 0.5), where KL-SOAP and the exact KL control
  are slightly better: on separable arrays KL weights are efficient (Theorem 3.3); the gap is inside
  G2.1's non-inferiority margin and recorded as an open item (theory §7). C-013 moves frame KL on
  these zero-mean streams by a few percent at most, in both directions (centering-effect tables).
  Exploratory runs (seeds 0–9, C-002) and pilots (seeds 21–35) are archived and not used.
* Raw: `results/e21_*.jsonl.gz` (re-runs `*_peerfix`, `*_c013`), report `results/e21_report.md`,
  gates `results/e21_gate.json`.
* Decision: G2.1, G2.3 pass; G2.2 needed E2.2b (F-016).

## E2.2b — Consistency in a tied plane (Phase 02)
* Question: does Gimbal's error vanish with the horizon in tied cells, where pooled factors cannot
  identify the frame?
* Design: E2.2 tie cell, seeds 10–14, horizons 800 / 4,000 / 16,000 steps, memory matched to the
  horizon; criterion fixed before the run (C-010).
* Result and raw: `results/e22b_report.md`, `results/e22b_tie_consistency.jsonl.gz`. Gimbal's frame
  KL falls about as 1/T; pooled-factor frames fall less than 2×, passing that clause narrowly.
  Re-run with the C-013 default (`results/e22b_report_c013.md`): the same picture, criterion met.
* Decision: G2.2 passes (before and after C-013).

## E2.5 — Optimization on noisy quadratics (Phase 02, gate G2.4)
* Question: does Gimbal reach a lower loss than every peer when each optimizer is tuned the same
  way, on quadratics whose curvature and gradient noise share a KRD frame?
* Design: γ ∈ {0, 1, 2} × σ ∈ {0.1, 1}, 32×48, 400 steps, warm-up and cosine schedule; every method
  gets the same 7-point learning-rate grid (factor 2 around its own centre), selected on seeds 0–2
  and evaluated on 12 fresh paired seeds; Adam in the true frame is the ceiling, not a peer.
* Code: `experiments/phase2/e25_noisy_quadratic.py`, `analyze_e25.py`.
* Run 1 (after C-011, default of C-012, evaluation seeds 100–111): G2.4 failed in the separable
  low-noise configuration (F-020); outputs `results/e25_*_full.*`.
* Run 2 (after C-013, evaluation seeds 300–311, two centering ablations of ours): the default has
  the lowest mean final loss in all six configurations and every paired bootstrap interval against
  every peer excludes 0; no selected learning rate on a grid edge; centering gains most where the
  deterministic signal dominates (low noise) and is neutral at σ = 1, and the adaptive factor stays
  within about 1% of always-on centering here. Outputs `results/e25_*_full_c013.*`, report
  `results/e25_report_full_c013.md`.
* Decision: G2.4 passes on run 2; F-020 closed by C-013.

## E2.9 — Ablations (Phase 02, reported)
* Question: does every component of the default earn its place, and what do the hyper-parameters
  do?
* Design: one setting changed at a time on seven representative E2.1–E2.3 cells, seeds 40–47, best
  of three rotation rates per variant; paired Wilcoxon against the default.
* Run 1 (baseline `frame_every = 1`, before C-013): `results/e29_report.md`.
* Run 2 (baseline the default: `frame_every = 4`, C-013; adds `k1`, `no_center`, `center_always`):
  `results/e29_report_c013.md`. Adaptive centering matches no centering on these zero-mean
  streams while always-on centering costs a few percent in most cells; shrinkage, the tied flow
  variance, the pooled warm start and the bias-corrected schedule each have large measured
  benefits; `frame_every = 1` is slightly better and 10 worse than the default; a smaller damping
  helps the separable cells (open item, theory §7).
* Decision: defaults kept; no component removed.

## E2.10 — Theory–simulation agreement (Phase 02)
* Question: do the theory's quantitative predictions match simulation?
* Parts: (a) Theorem 3 + small-angle expansion vs measured E2.1 frame KL; (b) Lemma 5.4.2 vs exact
  plug-in inflation; (c) Proposition 5.5 split-sample noise and shrinkage factor vs truth;
  (d) Theorem 4.2 one-step contraction vs prediction; (e) Hessian of J = Fisher, gradient of J =
  expected score.
* Result: all parts pass after one theory revision: (b) failed as pre-registered with the
  leading-order formula (F-013) and passed on fresh pairs with the derived second-order term.
  (a) passes in the six separable cells (the only ones in the linear regime, C-009); pooled factors
  are saturated in the non-separable cells, as the theory predicts.
* Part (f), added with C-013 (criteria fixed first, C-014): Proposition 5.7's noise factor η, the
  (1 − c)² law for the score bias of a plug-in mean, and the optimizer's plug-in factor against the
  oracle at five signal-to-noise ratios; a decaying mean is reported (the factor lags slightly
  above the oracle, as the proposition's limits say). Part (a) now reads the E2.1 rows through the
  analysis loader, so its KL columns are the C-011 re-runs and its Gimbal columns the C-013 re-runs
  (F-022). E2.10 re-run in full after the E2.1 re-runs.
* Raw: `results/e210_theory_vs_simulation.json`, report `results/e210_report.md`.

## E2.11 — Landscape and global convergence (Phase 02)
* Question: does the noise-free flow reach the global minimum from random frames (Conjecture 4.1)?
  Are other critical points strict saddles?
* Design: 4,500 Haar-random starts over five array types and three shapes; 180 constructed critical
  points (45° rotations of one or two pairs) with Hessian eigenvalues and escape runs.
* Result: every start reaches the global minimum; every constructed critical point is a strict
  saddle and is escaped (F-017 corrected an over-specified sub-check).
* Raw: `results/e211_landscape.jsonl`, `results/e211_saddles.jsonl`, report `results/e211_report.md`.
* Decision: G2.6 (i) passes; Conjecture 4.1 has numerical support (L1), still unproved.

## E2.12 — Numerical behaviour (Phase 02)
* Question: float32 vs float64, long-run orthogonality, scale invariance, degenerate inputs.
* Result: passes after the F-014 fix (absolute constants) and the F-015 test redesign. Re-run with
  the C-013 default and a new degenerate case, a constant (noise-free) gradient, for which the
  centered statistic is pure rounding residue: every part passes, the frames stay finite and
  orthogonal.
* Raw: `results/e212_numerics.json`, report `results/e212_report.md`; after C-013
  `results/e212_numerics_c013.json`, `results/e212_report_c013.md`.

## Phase 02 re-runs after C-015 (sign-equivariant spectral-norm estimate)
* Question: does C-015 change any Phase 02 result? (The change only affects frame moves whose
  trust-region cap is active, mostly the first few.)
* Design: every experiment that executes Gimbal's update re-run with the seeds of the C-013 runs
  (paired before/after): E2.1–E2.4 Gimbal rows (`gimbal`, `gimbal_k4`), E2.2b, E2.5 (evaluation
  seeds 300–311, full tuning protocol), E2.9, E2.12; E2.10 re-run because part (a) reads E2.1.
  Command: `experiments/phase2/run_c015.sh` (60 workers), then `analyze_e21.py`,
  `analyze_e25.py --tag full_c015`, `e210_theory_vs_simulation.py`, `make_report.py`.
* Result: all gates G2.1–G2.6 pass (`experiments/phase2/report.md`); paired effect on the frame KL
  at best memory within seed noise in nearly every cell (E2.1 report, "effect of C-015").
* Raw: `experiments/phase2/results/*_c015*`.

## E3.0 — JAX implementations against the PyTorch references (Phase 03)
* Question: do the JAX AdamW, SOAP and Gimbal implement the reference rules?
* Design: `tests/test_jax_optimizers.py` (criteria C-016): float64 50-step golden sequences,
  float32 single steps from the reference's state for every step kind, float32 accuracy relative
  to the float32 reference, invariants (orthogonality over 10⁴ steps, equivariance, eigenvector
  sign gauge, scale invariance, descent), toy problems, edge cases.
* Result: all pass after F-024 (C-015), F-025 (C-016) and F-026; float64 agreement ≈1e-12 over
  60 steps for Gimbal.
* Raw: pytest output (51 tests, CPU).
