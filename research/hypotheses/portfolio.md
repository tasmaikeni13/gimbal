# Hypothesis portfolio

Maintained under the `ml-research` skill. Status values: proposed | implemented | screening |
confirming | killed | supported. A hyperparameter variant stays inside its parent branch unless it
changes the mechanism.

| H-ID | Causal delta vs SOAP | Load-bearing assumption | Unique prediction | Status | Next test |
|---|---|---|---|---|---|
| H1 Gimbal | basis = online MLE of the free-diagonal rotated model (joint diagonalization) by natural-gradient flow on O(m)×O(n); no L, R; no QR after init | real gradient second moments are not Kronecker-separable in their eigenvalues, and pooled estimators waste samples | advantage over SOAP and KL-SOAP grows with non-separability; disappears vs KL-Shampoo for separable D | screening (Phase 2) | MC basis efficiency; NQM; small LM |
| H2 Transport | rotate V with the doubly-stochastic map (P∘P) when the basis moves | basis motion per step is large enough for V mismatch to matter | gains concentrated early in training and at high rotation rates | proposed (ablation of H1) | ablation in Phase 2 |
| H3 Wiener–secant | eigenvalue rule from regression curvature Cov(g̃,w̃)/Var(w̃) with SNR shrinkage | curvature is approximately diagonal in the tracked basis and identifiable from the trajectory | larger gains at large batch (low noise) | reserve | only if H1's ceiling (oracle basis) is reached |
| H4 Shared residual basis | one basis per residual-stream side shared by all layers | residual-side statistics are shared across depth | amortized cost; loss neutral or better at small batch | reserve | measure cross-layer basis overlap in Phase 2 |
| H5 Lie momentum | heavy-ball in the Lie algebra for the basis flow | eigenvectors drift smoothly | lower tracking lag under drift at equal noise | proposed (sub-mechanism) | drift MC in Phase 2 |
| H6 Robust score | Student-t score z/(D + z²/ν) in the generator | heavy-tailed gradient noise biases the Gaussian score | gains only with heavy-tailed noise | proposed | heavy-tail MC in Phase 2 |

## H1 — Gimbal (selected)

* **Parent / inspiration:** SOAP (Adam in a Kronecker eigenbasis); common principal components
  (Flury 1984); nonstationary blind source separation (Pham & Cardoso 2001); equivariant serial
  updates (Cardoso & Laheld 1996); natural gradient (Amari 1998).
* **Exact causal delta:** replace SOAP's `L, R` accumulators and periodic `QR(L Q)` refresh by the
  per-step Lie-algebra update
  `Ω_L = −α · (S_L − S_Lᵀ) ⊘ (F_L + δ n)`, `S_L = Z (Z ⊙ D^{-1})ᵀ`,
  `F_L = D (D^{-1})ᵀ + D^{-1} Dᵀ − 2n`, with `Z = Q_Lᵀ G Q_R` and `D` the bias-corrected Adam
  second moment (plus a relative floor); same on the right; retraction
  `Q ← Q (I + Ω + Ω²/2)` with periodic Newton–Schulz re-orthonormalization.
* **Mechanism hypothesis:** the basis converges to the KL projection of the gradient second moment
  onto SOAP's own preconditioner family; pair-dependent, gap-aware weights make the estimator
  efficient (Cramér–Rao) where pooled factors are not.
* **Unique prediction:** the gap to SOAP widens with the non-separability index
  `κ = ‖log D − rank-1-additive fit‖² / ‖log D‖²` and with the steepness of the other side's spectrum;
  for exactly separable D, Gimbal ≈ KL-Shampoo's basis quality, both better than SOAP.
* **Simplest alternative explanation:** freshness (per-step updates) rather than the estimator;
  control = SOAP real-time (per-step QR including the current gradient).
* **Closest precedent:** KL-SOAP (2509.03378, 2607.20548); TurboSoap (blog).
* **Smallest discriminating experiment:** synthetic gradient streams from the KRD model with
  controlled κ, compare basis KL at matched memory time-constants.
* **Required controls:** oracle basis; identity basis (AdamW); SOAP f=10; SOAP real-time; KL-SOAP.
* **Estimated cost:** Phase 2 CPU budget (hours).
* **Success threshold (Phase 2):** lower median final loss than every peer on the NQM suite and on
  the small-LM benchmark with non-overlapping 95% bootstrap intervals or a paired test p < 0.05
  across seeds; step time ≤ SOAP's in the FLOP/memory model.
* **Kill criterion:** measured κ on real LM gradients ≈ 0 **and** SOAP real-time ≥ Gimbal on loss.

## Decision record

* 2026-10-06: H1 selected as the main line after the literature pass (no overlap found) and two
  prototype checks (basis KL Monte Carlo; noisy quadratic). Prototype numbers are exploratory and
  are superseded by the Phase 2 runs recorded under `experiments/phase2/`.

## Repair cards for F-027 (G3.5: the default lost to SOAP on the small LM)

Selection seeds 10–12 (not used by the gate), learning rate 4e-3 (Gimbal's selected one), 800
steps of E3.2; raw `experiments/phase3/results/lm/*_f027.*`. The gate is re-run on fresh seeds
with the full E3.2 protocol after a choice is made.

| Card | Change | Mechanism hypothesis | Unique prediction | Kill criterion | Cost on TPU |
|---|---|---|---|---|---|
| R1 adaptive amortization | move every `k_t = clamp(round(K α/α_t), 1, K)` steps (K = 4) | the amortized flow matches the per-step flow to first order in kα; with the bias-corrected rate (α_t ≈ 1/t early) a fixed k = 4 is far outside that regime for the first ~100 steps, which costs a deficit that never closes | removes most of the k = 4 vs k = 1 gap at step 100, part of it at the end | final loss not lower than k = 4 by ≥ half the k = 4 → k = 1 gap | none after ~100 steps |
| R2 per-step flow | k = 1 (undo C-012) | amortization also costs in the steady state (staleness of a frame that is still moving) | best loss | — | every step pays a frame move (F-019) |
| R3 shorter warm start | T_w = 20 | the deficit comes from the pre-warm-start phase | early gap closes | no gain over k = 4 | none |
| R4 shorter memory | rot_rate α = 1 − β₂ = 0.05 (frame, flow variances and Adam's V share one window of effective size 39; SOAP's default `shampoo_beta = β₂` is the same convention) | the default α = 0.02 (C-002) was chosen on stationary synthetic streams; LM gradient statistics drift, and a 100-step memory lags | larger α lowers the loss for every k | larger α not better at k = 4 and k = 1 | none |

Outcome so far (mean final validation loss over seeds 10–12; SOAP 1.4981): k = 4, α = 0.02:
1.5108; R3: 1.5117 (killed); R1: 1.5018; R2: 1.4949; R4 at k = 4: α = 0.04 1.5029, α = 0.08 1.5001;
R1 + R4 (α = 0.04): 1.4968; R2 + R4 (α = 0.04): 1.4900.
