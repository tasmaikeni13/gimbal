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
