# Phase 02 — Mathematical, statistical and Monte Carlo analysis (iterate until Gimbal wins)

## Purpose

Test the predictions of Phase 01 quantitatively, against every peer, before any expensive
hardware is used. The phase iterates on the mathematics (protocol §4) until Gimbal matches or beats
all peers on the predeclared metrics, or until the evidence shows it cannot (protocol §9).

## Depends on

Phase 01 (theory, peer table, Lean), `src/gimbal/torch/` (reference optimizers).

## Produces

* `experiments/phase2/*.py` — every experiment as a script with a fixed seed list and CLI config
* `experiments/phase2/results/` — raw outputs (JSON/CSV), never edited by hand
* `experiments/phase2/report.md` — generated tables and figures with uncertainty
* `research/ledger/experiments.md`, `failures.md`, `claims.md` updated

## Skills

`experimental-research` (design, measurement model, uncertainty), `ml-research` (controls,
tuning fairness, claims), `theory-research` (when a prediction fails).

## Experiments

All experiments are paired: for a given seed, every method sees the same problem instance and the
same noise/data stream. The experimental unit is the seed (Monte Carlo) or the training run (LM).

| ID | Question | Design | Primary metric |
|---|---|---|---|
| E2.1 | Frame-estimation efficiency | Gradient streams from KRD$(U^\star,D)$ with $\log D_{ij}=a_i+b_j+\gamma c_{ij}$, spectra slope $s$; grid $\gamma\in\{0,0.5,1,2\}$, $s\in\{0.5,1,1.5\}$, shapes $(32,48),(64,64)$; memories matched by effective sample size; ≥10 seeds | time-averaged frame KL $J(\hat U)$ (Prop. 1) |
| E2.2 | Identifiability stress | Crossing profiles with tied row sums (Example 1 family) | $J(\hat U)$, angle error in the tied plane |
| E2.3 | Tracking under drift | $U^\star(t)$ rotates at angular velocity $\omega$; best memory per method | time-averaged $J$ |
| E2.4 | Heavy tails | Student-$t$ noise ($\nu\in\{3,5,\infty\}$) | $J$; tests hypothesis H6 |
| E2.5 | Noisy quadratic optimization | Loss $\tfrac12\sum H_{ij}(Q_L^{\star\top}(W-W^\star)Q_R^\star)_{ij}^2$, gradient noise with KRD covariance $\propto H$, noise levels (batch sizes) × $\gamma$ grid; LR tuned per method on an equal grid; ≥8 seeds | final loss, steps to target |
| E2.6 | Premise check on real gradients | Train a small LM; at checkpoints, collect gradients and measure the non-separability index $\kappa$ of $D$ in SOAP's frame and the frame KL of SOAP vs Gimbal on held-out gradient snapshots | $\kappa$, $J$ |
| E2.7 | Small-LM benchmark (CPU) | Byte-level Llama-style LM on a FineWeb-Edu sample; equal LR-grid budget per optimizer; ≥3 seeds at the chosen LR | validation loss at equal steps; loss vs wall-clock |
| E2.8 | Cost model | FLOPs/memory per optimizer for the 125M config (Prop. 9) and CPU microbenchmarks per matrix shape; projected TPU step time | optimizer-step time |
| E2.9 | Ablations | $\alpha$, $\delta$, $\rho$, transport on/off, init eigh vs identity, retraction order | E2.1/E2.5/E2.7 metrics |

Peers: AdamW, SOAP (f=10), SOAP real-time (per-step frame from factors including the current
gradient), KL-SOAP (F=1), KL-Shampoo (frame only, in E2.1–E2.3), Muon, NorMuon, SPlus; ARO and
Shampoo-with-grafting when time allows (record if omitted).

## Exit gate (predeclared)

* **G2.1 (efficiency).** In every E2.1 cell with $\gamma>0$, Gimbal's frame KL is lower than SOAP,
  SOAP real-time and KL-SOAP (paired one-sided Wilcoxon over seeds, Holm-corrected, $p<0.05$). At
  $\gamma=0$ Gimbal is within 25% of the best peer (theory predicts parity with KL-Shampoo there).
* **G2.2 (identifiability).** In E2.2 Gimbal identifies the tied plane (angle error → 0 with samples)
  while pooled-factor frames do not.
* **G2.3 (tracking).** In E2.3 Gimbal's best-memory tracking error ≤ every peer's best-memory error.
* **G2.4 (optimization).** In E2.5 Gimbal has the lowest mean final loss in every configuration with
  $\gamma\ge1$ (paired bootstrap 95% CI of the difference to each peer excludes 0) and is not
  significantly worse than the best peer anywhere.
* **G2.5 (premise).** E2.6 finds $\kappa$ clearly above 0 on real LM gradients, at a level where E2.1
  predicts a material advantage (operationalized by C-005: median noise-corrected $\kappa\ge0.05$
  over snapshot matrices and checkpoints).
* **G2.6 (small LM).** In E2.7 Gimbal's mean final validation loss is lower than every peer's at equal
  steps (paired over seeds; Holm-corrected one-sided paired test $p<0.05$, or lower on every seed when
  only 3 seeds are affordable), and its optimizer step costs no more than SOAP's in E2.8 (C-005:
  analytic cost with QR/eigh at 10× matmul cost, for the same configuration as the loss
  comparison; CPU microbenchmarks reported alongside).

## Iteration rule

If a gate item fails, run the protocol loop. Candidate repairs already in the portfolio: damping and
floor schedules, rotation-rate schedule (Robbins–Monro start, constant tail), Lie-algebra momentum
(H5), robust score (H6), variance transport (H2), eigenvalue-rule changes (H3). Every repair is first
justified in the theory document and, where it changes a theorem, re-checked in Lean (Phase 01 files
become stale through protocol §5). Statistical gates are re-run on fresh seeds after a repair.

## Invalidation triggers

Changes to the Gimbal update rule or defaults, to any peer implementation, or to the theory items
that define the metrics (Prop. 1, Thm. 3).
