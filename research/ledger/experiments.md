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
* Result: Gimbal below every peer and control in every cell of every suite, at best and at matched
  memory; separable cells included (bootstrap upper bound of the ratio to the best peer below 1).
  The amortized variant (`frame_every=4`) also wins every cell. Exploratory runs (seeds 0–9, C-002)
  and pilots (seeds 21–32) are archived and not used.
* Raw: `results/e21_*.jsonl.gz`, report `results/e21_report.md`, gates `results/e21_gate.json`.
* Decision: G2.1 and G2.3 pass; G2.2 needed E2.2b (F-016).

## E2.2b — Consistency in a tied plane (Phase 02)
* Question: does Gimbal's error vanish with the horizon in tied cells, where pooled factors cannot
  identify the frame?
* Design: E2.2 tie cell, seeds 10–14, horizons 800 / 4,000 / 16,000 steps, memory matched to the
  horizon; criterion fixed before the run (C-010).
* Result and raw: `results/e22b_report.md`, `results/e22b_tie_consistency.jsonl`. Gimbal's frame KL
  falls about as 1/T; pooled-factor frames fall less than 2×, passing that clause narrowly.
* Decision: G2.2 passes.

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
* Result: passes after the F-014 fix (absolute constants) and the F-015 test redesign.
* Raw: `results/e212_numerics.json`, report `results/e212_report.md`.

