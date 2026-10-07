# Optimizers: update rules, defaults and sources

Every optimizer acts on one matrix parameter `W ∈ R^{m×n}` with gradient `G`. In all comparisons the
hidden matrices (attention q, k, v, o; MLP gate, up, down) use the optimizer under test and every
other parameter (embeddings, output head, norms) uses one AdamW configuration that is identical for
all methods (Phase 03 routing; in the TPU study its learning rate is the method's tuned rate, its
weight decay is zero, and for the AdamW baseline, one optimizer, its tuned β₂ applies to every
parameter). Decoupled weight decay applies to hidden matrices only. `lr` is the
scheduled learning rate of the step. Implementations: PyTorch `src/gimbal/torch/`, JAX
`src/gimbal/jax/` (AdamW, SOAP and Gimbal only, the optimizers of the TPU study, decision D-003).
The JAX versions agree with the PyTorch ones to about 1e-12 (Gimbal) and 1e-8 (SOAP) per step in
float64 (`tests/test_jax_optimizers.py`).

Notation: `Z = Q_Lᵀ X Q_R` (rotated coordinates), `⊙`, `⊘` entrywise, `t` the step count.

## AdamW

Source: Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019; rule as in
`torch.optim.AdamW`.

    W ← W (1 − lr·λ)
    M ← β₁ M + (1 − β₁) G,   V ← β₂ V + (1 − β₂) G⊙G
    W ← W − lr · (M / (1 − β₁ᵗ)) ⊘ (sqrt(V / (1 − β₂ᵗ)) + ε)

Defaults here: β = (0.9, 0.95), ε = 1e-8, λ = 0.1 on hidden matrices (TPU study), learning-rate grid
centre 1e-3 (the `torch.optim.AdamW` default). Secondary knob in tuning: β₂ ∈ {0.95, 0.98, 0.999}.

## SOAP (f = 10)

Source: Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam*, arXiv:2409.11321, and the
official implementation `nikhilvyas/SOAP` (vendored unmodified in `tests/third_party/soap_official.py`;
`tests/test_optimizers.py::test_soap_matches_official`).

* First call: `L = (1 − β_s) G Gᵀ`, `R = (1 − β_s) Gᵀ G`, `Q_L, Q_R` = eigenvectors (descending);
  no parameter update.
* Every later call (Adam counter `t`): `Z = Q_Lᵀ G Q_R`; `M ← β₁ M + (1 − β₁) Z`,
  `V ← β₂ V + (1 − β₂) Z⊙Z` (both in rotated coordinates);
  `W ← W − lr·sqrt(1 − β₂ᵗ)/(1 − β₁ᵗ) · Q_L (M ⊘ (sqrt V + ε)) Q_Rᵀ`, then `W ← W − lr·λ W`.
* Factors after the step: `L ← β_s L + (1 − β_s) G Gᵀ`, `R ← β_s R + (1 − β_s) Gᵀ G`.
* Every f = 10 steps: order each side by the estimated eigenvalues `diag(Qᵀ L Q)`, permute `V`
  accordingly, one power-iteration step and a QR decomposition `Q ← qr(L Q)`; re-express `M` in
  the new frame. (The official code also rotates `M` back and forth at every other step; that is
  the identity up to rounding and is done only at refreshes here, the same economy that Gimbal
  gets.)

Defaults: β = (0.95, 0.95), `shampoo_beta` β_s = β₂, ε = 1e-8, f = 10, `max_precond_dim` 10000;
λ = 0.1 in the TPU study (official default 0.01; the study uses one value for all methods);
learning-rate grid centre 3e-3 (official default). Secondary knob: β_s ∈ {0.9, 0.95, 0.99}.
On the TPU the QR is XLA's Householder `jnp.linalg.qr`.

## Gimbal

Source: `theory/gimbal_theory.md` §5 (this project). Defaults as of change C-018.

* Frames: `Q_L, Q_R` from the eigenvectors of the first gradient's factors; after
  `T_w = 50` steps they are replaced by the eigenvectors of the pooled factors of those steps
  (Remark 5.6), and `V`, `V^F` are transported to the new frame (Theorem 7).
* Adam in the frame: as SOAP's step (momentum kept in rotated coordinates), with
  `W ← W (1 − lr·λ)` before the step (AdamW convention).
* Frame statistic: the innovation `G − c·M̂_{t−1}` with the positive-part James–Stein factor
  `c = (1 − η·tr Σ̂ / ‖M̂_{t−1}‖²)_+` (Proposition 5.7).
* Flow variances: `V^F ← β_D V^F + (1 − β_D) Z⊙Z` with `β_D = 1 − α` and an odd-step half;
  `D` = exp(additive fit of log V̂^F + shrink · residual), shrink = `(1 − noise/‖residual‖²)_+`
  with the noise measured from the odd/even split (Proposition 5.5), relative floor ρ.
* Flow: the skew score `S_L − S_Lᵀ`, `S_L = Z (Z ⊙ D^{⊙−1})ᵀ` (and the right side), is accumulated
  over `k_t` steps; the frame then moves by `Ω = −rate · score ⊘ (F + δ·n)` (`F` the Fisher matrix
  of Theorem 2, rate `1 − Π(1 − α_s)` over the block, `α_t = α/(1 − (1 − α)ᵗ)` capped at 0.5),
  entries clipped to ±θ_max, spectral norm capped at 1 (8 power iterations started from the
  largest column of Ω, C-015), retraction `Q ← Q (I + Ω + Ω²/2)`, Newton–Schulz polish until the
  orthogonality defect is below 1e-6 (float32), at most 4 iterations.
* Block length: `k_t = clamp(round(K·α/α_t), 1, K)` (adaptive amortization, C-018).

Defaults: β = (0.9, 0.95) (the 125M study uses SOAP's β₁ = 0.95, C-019), ε = 1e-8, α (`rot_rate`) = 1 − β₂ = 0.05, δ (`damping`) = 0.003,
ρ (`floor`) = 1e-8, θ_max = 0.25, spectral cap 1, K (`frame_every`) = 4 adaptive, T_w = 50,
λ = 0.1 (TPU study); learning-rate grid centre 3e-3 (the reference implementation's default,
equal to SOAP's). Secondary knob: α ∈ {0.025, 0.05, 0.1}. History of the defaults:
`research/ledger/decisions.md` (C-001 … C-018).

## Peers implemented in PyTorch only (Phase 02 and the CPU benchmark E3.2)

| Optimizer | Source | Rule (summary) | Defaults in this repository |
|---|---|---|---|
| SOAP real-time | arXiv:2607.20548 §5.4.2 | SOAP whose factors include the current gradient and whose frame is refreshed every step before the projection | as SOAP, f = 1 |
| KL-SOAP | Lin et al., arXiv:2509.03378; arXiv:2607.20548 Alg. 2 | factors by the KL rule `L ← β_k L + ((1 − β_k)/n) G R⁻¹ Gᵀ` (and R), inverses in the current frame with relative damping; frame refreshed by power step + QR; Adam in the frame; factors initialized as σI with σ the first gradient's RMS (C-011) | lr 3e-3, β = (0.9, 0.95), β_k = 0.95, damping 1e-4, refresh every step |
| Muon | Jordan et al. 2024; arXiv:2502.16982 | Nesterov momentum, quintic Newton–Schulz orthogonalization (5 steps, coefficients 3.4445, −4.7750, 2.0315), RMS-matched scale | lr 0.02, momentum 0.95 |
| NorMuon | arXiv:2510.05491 | Muon followed by a neuron-wise second-moment normalization | lr 3e-3, β = (0.95, 0.95) |
| SPlus | Frans et al., arXiv:2506.07254, `kvfrans/splus` | sign of the momentum in a stale Shampoo frame, shape scaling, iterate averaging for evaluation | lr 0.1, β = (0.9, 0.999), EMA 0.999, eigh every 100 steps |
| ARO (Sinkhorn) | arXiv:2602.09006 | one-sided rotation from the QR of `M f(RᵀM)ᵀ` with Sinkhorn-normalized base update | lr 3e-3, momentum 0.95, 5 Sinkhorn iterations |

Shampoo with Adam grafting, COSMOS, PSGD-Kron and KL-Shampoo (the extended set of Phase 03) are
not implemented; the TPU study is limited to AdamW, SOAP and Gimbal (D-003).
