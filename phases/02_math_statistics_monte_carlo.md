# Phase 02 — Mathematical, numerical, statistical and Monte Carlo analysis (iterate until Gimbal wins)

## Purpose

Test the theory of Phase 01 quantitatively, against every peer, with several independent methods
before any language model is trained or any accelerator is used: formal proofs (Lean), mathematical
derivations, deterministic numerical analysis, Monte Carlo simulation and statistical inference.
The phase iterates on the mathematics (protocol §4) until Gimbal matches or beats all peers on the
predeclared criteria, or until the evidence shows it cannot (protocol §9).

Scope (decision C-006, at the user's instruction): this phase trains no language model. Everything
that needs language-model training (the real-gradient premise check and the small-LM benchmark,
formerly E2.6 and E2.7) belongs to Phase 03 and is executed there.

## Depends on

Phase 01 (theory, peer table, Lean), `src/gimbal/torch/` (reference optimizers).

## Produces

* `experiments/phase2/*.py` — every experiment as a script with a fixed seed list and CLI config
* `experiments/phase2/results/` — raw outputs (JSON/JSONL, gzipped when large), never edited by hand
* `experiments/phase2/report.md` — generated tables with uncertainty (`make_report.py`)
* `formal/` — Lean statements for every new exact claim used by this phase
* `research/ledger/experiments.md`, `failures.md`, `claims.md`, `decisions.md` updated

## Skills

`theory-research` (predictions, proof audit, refutation), `experimental-research` (design,
measurement model, uncertainty), `ml-research` (controls, tuning fairness, claims),
`literature-frontier` (novelty checks for any new mechanism), `mechanism-transfer` (repairs).

## Methods

| Method | Used for | Where |
|---|---|---|
| Formal proof (Lean 4 + Mathlib) | exact identities and inequalities the experiments rely on | `formal/` |
| Mathematical derivation | asymptotic predictions with explicit assumptions and labels | `theory/gimbal_theory.md` |
| Deterministic numerics | population (noise-free) flows, exact asymptotic variances, Hessians, cost counts, floating-point behaviour | E2.8, E2.10–E2.12 |
| Monte Carlo | estimator and optimizer behaviour on synthetic streams with known ground truth | E2.1–E2.5, E2.9, E2.10 |
| Statistical inference | paired non-parametric tests with multiplicity control, bootstrap intervals, non-inferiority bounds, random-effects pooling across cells | all comparative experiments |

## Experiments

All comparative experiments are paired: for a given seed every method sees the same problem
instance and the same noise stream. The experimental unit is the seed.

| ID | Question | Design | Primary metric |
|---|---|---|---|
| E2.1 | Frame-estimation efficiency | Gradient streams from KRD$(U^\star,D)$ with $\log D_{ij}=a_i+b_j+\gamma c_{ij}$, spectra slope $s$; grid $\gamma\in\{0,0.5,1,2\}$, $s\in\{0.5,1,1.5\}$, shapes $(32,48),(64,64)$; every method at a grid of memories, extended equally for all methods while any best memory is on a grid edge; 10 confirmatory seeds (10–19) | time-averaged frame KL $J(\hat U)$ (Prop. 1) over the second half |
| E2.2 | Identifiability stress | Crossing profiles with tied row sums (Example 1 family); E2.2b: horizons 800, 4,000, 16,000 steps with memory matched to the horizon (seeds 10–14) | $J$; $J$ against the horizon |
| E2.3 | Tracking under drift | $U^\star(t)$ rotates at angular velocity $\omega$; best memory per method | time-averaged $J$ |
| E2.4 | Heavy tails | Student-$t$ noise ($\nu\in\{3,5,\infty\}$) | $J$ |
| E2.5 | Noisy quadratic optimization | Loss $\tfrac12\sum H_{ij}(Q_L^{\star\top}(W-W^\star)Q_R^\star)_{ij}^2$, gradient noise with KRD covariance $\propto H$, noise levels × $\gamma$ grid; LR tuned per method on an equal 7-point grid (seeds 0–2), evaluated on fresh seeds 100–111 (re-test after C-013: 300–311, with two centering ablations of ours) | final loss |
| E2.8 | Cost model | Multiply–accumulate counts and state for the 125M configuration (Prop. 9), QR/eigh counted separately; CPU microbenchmarks per matrix shape | % of model compute, state, ms per step |
| E2.9 | Ablations | One setting of the default changed at a time: transport, initialization, warm-start length, shrinkage, flow variance source, $\delta$, $\rho$, trust region, schedule, amortization $k$, polish period, centering of the frame statistic (none, always; C-013); 8 seeds (40–47) | $J$ relative to the default |
| E2.10 | Theory–simulation agreement | (a) predicted frame KL from the asymptotic variances of Theorem 3 and the expansion $J\approx\tfrac12\sum_pF_p\theta_p^2$ against measured E2.1 values (pooled factors exactly; Gimbal against the Cramér–Rao value); (b) Lemma 5.4.2's plug-in inflation against its exact value on random pairs; (c) Proposition 5.5's split-sample noise estimate and shrinkage factor against the truth; (d) Theorem 4.2's gap-independent contraction against SOAP's power-iteration rate, by deterministic one-step perturbation; (e) Hessian of $J$ at $U^\star$ against the Fisher information; (f) Proposition 5.7: the momentum's noise factor, the $(1-c)^2$ score bias of a plug-in mean, the plug-in factor against the oracle (C-014) | ratios measured/predicted |
| E2.11 | Landscape and global convergence (Conjecture 4.1) | Noise-free population flow (exact expected score at the current frame) from Haar-random starts on separable, non-separable, tied and near-degenerate arrays; Riemannian Hessian at every end point that is not the global minimum | fraction converged to $J<10^{-8}$; type of other end points |
| E2.12 | Numerical behaviour | float32 vs float64 on E2.1 cells; 10⁴-step orthogonality; gradient scales $10^{\pm30}$ (float64) and $10^{\pm15}$ (float32); zero gradients, dead rows, rank-one gradients, variance range $10^{12}$, a constant (noise-free) gradient | NaN/Inf count, orthogonality defect, $J$ ratio fp32/fp64, frame deviation across scales |

Peers: AdamW, SOAP (f=10), SOAP real-time (per-step frame from factors including the current
gradient), KL-SOAP (F=1), Muon, NorMuon, SPlus, ARO; exact pooled-factor and KL-factor
eigenvectors at every step as controls in E2.1–E2.4.

## Exit gate (predeclared)

* **G2.1 (efficiency).** In every E2.1 cell with $\gamma>0$, Gimbal's frame KL is lower than SOAP,
  SOAP real-time and KL-SOAP (paired one-sided Wilcoxon over seeds, Holm-corrected, $p<0.05$). At
  $\gamma=0$ Gimbal is within 25% of the best peer (theory predicts parity with KL-Shampoo there);
  C-006 adds that the paired bootstrap 95% upper bound of the geometric-mean ratio to the best peer
  is also below 1.25.
* **G2.2 (identifiability).** In E2.2 Gimbal identifies the tied plane while pooled-factor frames do
  not, and Gimbal is below every peer in every tie cell. "Identifies" means consistency: in E2.2b,
  with memory matched to the horizon, Gimbal's frame KL falls monotonically and at least 4× from
  800 to 16,000 steps while the exact pooled-factor frame falls less than 2× (C-010; the C-006
  wording "keeps falling as memory lengthens" at a fixed 800-step horizon failed, F-016).
* **G2.3 (tracking).** In E2.3 Gimbal's best-memory tracking error ≤ every peer's best-memory error.
* **G2.4 (optimization).** In E2.5 Gimbal has the lowest mean final loss in every configuration with
  $\gamma\ge1$ (paired bootstrap 95% CI of the difference to each peer excludes 0) and is not
  significantly worse than the best peer anywhere.
* **G2.5 (theory–simulation agreement; C-006).**
  (i) Pooled-factor control: measured/predicted frame KL in $[0.75,1.33]$ in every E2.1 main cell
  where the prediction is in the linear regime: pairs whose predicted angle s.d. exceeds 0.15 rad
  carry at most 15% of the predicted frame KL (C-009; as first written — largest per-pair s.d.
  ≤ 0.15 rad — the criterion excluded every cell, because nearly degenerate pairs always exceed it
  while carrying under 11% of $J$).
  (ii) Lemma 5.4.2: the second-order inflation formula is within 15% of the exact excess
  $V_w F-1$ for pairs with predicted excess ≤ 1 and $\bar\varepsilon^2\le0.02$.
  (iii) Proposition 5.5: the split-sample noise estimate has relative bias within ±15% for Gaussian
  streams at every tested time and memory, and the mean shrinkage factor is within 0.1 of the oracle
  factor $S/(S+\nu)$.
  (iv) Theorem 4.2: the measured one-step contraction of the population flow is within 10% of
  $1-\alpha F_{ik}/(F_{ik}+\delta n)$ for every tested pair, across eigen-gaps spanning at least 100×.
  (v) The Hessian of $J$ at $U^\star$ equals the Fisher information to relative error $10^{-4}$.
  (vi) Proposition 5.7 (added with C-013 by C-014, criteria fixed before the check ran): the Monte
  Carlo variance of the bias-corrected momentum of unit-variance noise is within 3% of $\eta$ for
  $T\in\{2,5,20,100,1000\}$; the Monte Carlo mean of the skew score at the true frame with the plug-in
  mean $c\hat M_{t-1}$ is within 5% (Frobenius norm, relative to the uncentred bias) of
  $(1-c)^2$ times the uncentred bias for $c\in\{0,0.5,1\}$; and the mean of the optimizer's plug-in
  factor (its own code path, 32×48 Gaussian streams, $t=200$) is within 0.1 of the oracle $c^\star$ at
  signal-to-noise ratios $\|\mu\|^2/(\eta\,\mathrm{tr}\,\Sigma)\in\{0,0.1,1,10,100\}$. A decaying mean is reported,
  not gated.
* **G2.6 (landscape, cost, numerics; C-006).**
  (i) E2.11: for arrays with pairwise-distinct profiles, ≥ 99% of random starts reach the global
  minimum; every other end point found is a strict saddle. A spurious local minimum refutes
  Conjecture 4.1 and triggers the protocol loop (the theory, not the gate, changes first).
  (ii) E2.8: the analytic optimizer cost of the default configuration, with QR/eigh at 10× matmul
  cost, is no more than SOAP's (C-005).
  (iii) E2.12: no NaN/Inf anywhere; float32 orthogonality defect ≤ 1e-5 over 10⁴ steps; float32
  frame KL within 10% of float64 in every tested cell; float64 frames at scales $10^{\pm30}$ equal
  those at scale 1 to 1e-8.

Reported alongside (not gates): matched-memory comparisons; Gimbal's measured frame KL relative to
the Cramér–Rao prediction; random-effects pooling across cells of the paired log-ratios; E2.9
ablation effects.

## Iteration rule

If a gate item fails, run the protocol loop. Candidate repairs already in the portfolio: damping and
floor schedules, rotation-rate schedule, Lie-algebra momentum (H5), robust score (H6), variance
transport (H2), eigenvalue-rule changes (H3). Every repair is first justified in the theory document
and, where it changes a theorem, re-checked in Lean (Phase 01 files become stale through protocol
§5). Statistical gates are re-run on fresh seeds after a repair.

## Invalidation triggers

Changes to the Gimbal update rule or defaults, to any peer implementation, or to the theory items
that define the metrics (Prop. 1, Thm. 3, Lemma 5.4, Prop. 5.5, Prop. 5.7).
