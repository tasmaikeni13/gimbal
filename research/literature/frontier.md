# Literature frontier: SOAP, its weaknesses, and the attempts to improve it

Search current through **2026-10-06**. Maintained under the `literature-frontier` skill
(https://github.com/tasmaikeni13/skills). Evidence labels: **[R]** reported by source,
**[V]** verified by reading the primary text in this project, **[I]** inferred here,
**[S]** open speculation.

## 1. Research contract

| Field | Value |
|---|---|
| Question | Which mechanism, not already published, can make a matrix-preconditioned optimizer beat SOAP on validation loss/perplexity **and** wall-clock at the 125M-parameter / 2.5B-token FineWeb-Edu scale on a TPU v4-32 pod? |
| Target phenomenon | Token efficiency and step cost of SOAP-family optimizers in LM pre-training |
| Objects and regime | Dense decoder-only transformer, ~125M params, ~20 tokens/param (≈1x Chinchilla), batch ≈ 0.5M tokens, bf16 compute on TPU v4 |
| Progress criterion | Lower final validation loss than SOAP (and peers) at equal tokens **and** lower or equal time per step; equal-tuning-budget protocol; 2 seeds per optimizer |
| Comparison class | AdamW; SOAP (f=10 and per-step "real-time" variant); KL-SOAP / KL-Shampoo; Shampoo (grafted); Muon; NorMuon; SPlus; ARO; COSMOS (optional); Kron/PSGD (optional) |
| Constraints | TPU-friendly kernels (matmul-dominant), memory ≤ SOAP's, ≤ 1 new essential hyperparameter |
| Assumptions introduced by this framing | That "beats SOAP" is decided by matched-budget runs at a single scale; that 1x Chinchilla is the operating regime (SOAP's relative standing changes with the overtraining factor, see E-cards below) |

## 2. Search ledger

| Date | Site | Query or traversal | Useful hits | New vocabulary | Coverage gap |
|---|---|---|---|---|---|
| 2026-10-06 | web, arXiv | "SOAP Improving and Stabilizing Shampoo using Adam" follow-ups | SOAP (2409.11321), NeurIPS'24 talk | preconditioning frequency | — |
| 2026-10-06 | web | KL-Shampoo / KL-SOAP | 2509.03378 (v1–v10) | augmented eigenvalues, two-sided KL estimation | — |
| 2026-10-06 | web | Purifying Shampoo | 2506.03595 | eigenvalue correction, adaptive QR termination criterion | — |
| 2026-10-06 | web | COSMOS hybrid SOAP/Muon | 2502.17410 (ICLR'26) | leading eigensubspace | — |
| 2026-10-06 | web | Fantastic optimizers; Benchmarking optimizers | 2509.02046, 2509.01440 | token multiplier, scalar vs matrix optimizers | — |
| 2026-10-06 | web | SPlus stable whitening | 2506.07254 | instant-sign, iterate averaging, shape-aware scaling | — |
| 2026-10-06 | web (extended) | SOAP 2026 arXiv improvements | 2607.20548 (NVIDIA), 2609.04577, 2608.28557, 2509.22938 | "slingshot" instability, real-time eigenbasis | — |
| 2026-10-06 | web (extended) | Shampoo 2026 eigendecomposition-free | KL-Shampoo QR variant, Structured preconditioners (2503.10537) | — | — |
| 2026-10-06 | web (extended) | Muon vs SOAP 2026 | Newton-Muon 2604.01472, HTMuon 2603.10067, 2509.11983 | — | — |
| 2026-10-06 | web | NorMuon, PolarGrad, Dion, PolarAdamW | 2510.05491, 2505.21799, 2504.05295, 2605.07067 | gauge equivariance | — |
| 2026-10-06 | web | SOAP memory / one-sided / low-rank 2026 | Clean 2610.04204, AdaDiag 2502.07488, ICLR'26 efficient optimizer design | Nyström sketch | — |
| 2026-10-06 | web | Kronecker eigenbasis rotation learning 2026 | Bregman/Kronecker 2606.00542, Adam-or-GN 2510.13680 | divergence-weighted residuals, BregTop | — |
| 2026-10-06 | web | eigenbasis staleness, Oja, Givens, Riemannian | TurboSoap (blog), basis-rotation async PP 2602.03515, ARO 2602.09006, Pion 2605.12492 | Brockett direction, symmetry teleportation | TurboSoap is a blog, not peer reviewed |
| 2026-10-06 | web | **novelty checks**: joint diagonalization / common principal components / EASI / Pham-Cardoso / Flury FG + optimizer | none applying them to optimizer bases | CPC, FG algorithm, nonstationary BSS | patents and non-English venues not searched |
| 2026-10-06 | web | TPU eigh/QR cost; Gram-Newton-Schulz; Polar Express | JAX eigh docs (QDWH default on TPU), 2505.16932, 2606.00371 | QDWH | no public TPU benchmark of SOAP's QR |

## 3. Evidence cards (compact)

| ID | Work | Claim used here | Regime | Label |
|---|---|---|---|---|
| E1 | SOAP, Vyas et al. 2024, arXiv:2409.11321 | Shampoo(power 1/2) = Adafactor in Shampoo's eigenbasis; SOAP = Adam in that basis; eigenbasis via one power-iteration step + QR every f steps; V is *re-ordered* (not rotated) when the basis changes; ≥40% fewer steps / ≥35% less wall-clock than AdamW at 360M/660M, 2M-token batch | 360M–660M, Chinchilla, H100 | [V] |
| E2 | KL-Shampoo / KL-SOAP, Lin et al., arXiv:2509.03378 | Shampoo's factor estimator is not the KL/MLE solution for a Kronecker *product* covariance; two-sided update S_a ← (1-β)S_a + β G S_b^{-1} Gᵀ/d_b; SOAP's d cannot be a Kronecker product (Claim 5) but no basis estimator for the free-diagonal model is derived; KL-Shampoo > SOAP at 123M–227M with 1B–2.5B tokens | 123M–227M, batch 0.5M | [V] |
| E3 | SOAP, Muon and Beyond (NVIDIA), arXiv:2607.20548 | Stale eigenbasis causes a "slingshot" instability at large batch (divergence at 8B); fix = per-step QR **and** current gradient in the basis; KL-SOAP further stabilizes; KL-SOAP slightly ahead of Muon at 30B-A3B | multi-B params, 24M–100M token batches | [V] |
| E4 | Purifying Shampoo, Eschenhagen et al., arXiv:2506.03595 | Grafting compensates stale and mis-scaled eigenvalues; correcting eigenvalues removes grafting; adaptive basis-refresh criterion via warm-started QR termination | AlgoPerf-style | [V] |
| E5 | Clarifying Shampoo, Eschenhagen et al., arXiv:2602.09314 | Shampoo = adapted Muon; KL-Shampoo 24.95 vs Muon 25.68 vs AdamW 27.74 val. ppl (Llama-320M, C4) | 320M–1.5B | [V] |
| E6 | SPlus, Frans et al., arXiv:2506.07254 | Shampoo diverges with cached inverses; U = Q_L sign(Q_Lᵀ Ḡ Q_R) Q_Rᵀ · 2/(m+n) + iterate averaging; reaches Adam's loss in ~44% of steps | ViT/LM, constant LR | [V] |
| E7 | COSMOS, Liu et al., arXiv:2502.17410 | SOAP on top-r eigensubspace + Muon on the rest; memory reduction | ≤ 1B | [V] |
| E8 | Clean, Rakhshan et al., arXiv:2610.04204 | Nyström sketch of SOAP's factors, linear memory; matches AdamW 26% faster | up to 13B single GPU | [V] (abstract + method section) |
| E9 | ARO, Gong et al., arXiv:2602.09006 | Rotation chosen by a Procrustes "norm-informed" policy R_t = QR(M f(R_{t-1}ᵀM)ᵀ), one-sided, per step; 1.3–1.35× over AdamW, 1.1–1.15× over Muon | ≤ 8B active, ≤ 8× overtrain | [V] |
| E10 | Fantastic optimizers, Wen et al., arXiv:2509.02046 | Matrix optimizers ≈1.4× over AdamW at 0.1B, ≈1.1× at 1.2B; Muon best at 1× Chinchilla, SOAP/Kron best at ≥ 8× | 0.1B–1.2B | [R] |
| E11 | Benchmarking optimizers, Semenov et al., arXiv:2509.01440 | Matrix-based (Kron, Muon, SOAP) ≈1.3× over AdamW < 520M | ≤ 720M | [R] |
| E12 | Optimizer memory schedules, Everett & Qiu, arXiv:2609.04577 | At 124M, Muon and SOAP lead at short horizons; SOAP overtakes Muon at long horizons; optimal memory grows with horizon | 51M–253M, OT 1–256× | [V] |
| E13 | Hyperparameter transfer, Qiu et al., arXiv:2512.05620 | μP-style LR scaling + 1/width WD gives consistent ≈1.4× for Muon/SOAP/Shampoo | 190M–1.4B | [R] |
| E14 | Gradient-whitening view, arXiv:2509.22938 | Idealized SOAP = Shampoo iff the whitening matrix is a Kronecker product; small nanoGPT shows little gain | 10M | [V] |
| E15 | AdaDiag, arXiv:2502.07488 | Adam in the basis of an SVD of the current gradient, refreshed every 200–500 steps | LLaMA ≤ 1B | [V] |
| E16 | Bregman/Kronecker, arXiv:2606.00542 | Frobenius / von Neumann / LogDet factor estimators distribute Kronecker error differently; top eigenspace aligns better with Hessian | small | [V] |
| E17 | TurboSoap (blog) | Matmul-only tangent/Brockett tracking of eigenvectors of L,R + Newton–Schulz re-orthogonalization; ~10× faster refresh, slightly worse loss | 50k steps toy | [R], not peer reviewed |
| E18 | NorMuon, arXiv:2510.05491 | Muon + neuron-wise second moments; +11% over Muon at 1.1B | ≤ 1.1B | [R] |
| E19 | Dion, arXiv:2504.05295 | Low-rank power-iteration orthonormalization, shard-friendly | 120M–3B | [R] |
| E20 | Muon (Jordan 2024), Muon is scalable (arXiv:2502.16982) | Newton–Schulz orthogonalized momentum | — | [R] |
| E21 | JAX `eigh` docs | QDWH is the TPU default; Jacobi available | TPU | [R] |
| E22 | Flury 1984; Flury & Gautschi 1986; Pham 2001; Pham & Cardoso 2001; Cardoso & Laheld 1996 | Common principal components / joint diagonalization MLE; Jacobi-type pairwise rotations; EASI relative-gradient serial updates; identifiability of nonstationary sources from variance profiles | statistics / signal processing | [R] |

## 4. SOAP weakness taxonomy (what an improvement must fix)

| W | Weakness | Evidence | Who attacked it |
|---|---|---|---|
| W1 | **Stale eigenbasis.** Basis refreshed every f steps from an EMA that lags; instability at large batch | E1, E3, E4 | NVIDIA (per-step QR), Purifying (adaptive frequency) |
| W2 | **Basis/statistics inconsistency.** The basis comes from pooled Kronecker factors (a Kronecker-*product* model) while the eigenvalues come from Adam's *free* diagonal; when the basis changes, V is only re-ordered, not transported | E1, E2 (Claim 5) | KL-SOAP fixes the factor estimator for the product model only |
| W3 | **Non-matmul linear algebra.** QR/eigh are sequential and slow on accelerators, worst on TPU (QDWH/Jacobi) | E1 §7.3, E21, E17 | TurboSoap (blog), KL-Shampoo QR variant |
| W4 | **Memory.** 2m²+2n²+3mn per layer (L,R,Q_L,Q_R,M,V,grad) | E1 §7.2 | one-sided SOAP, COSMOS, Clean, 4-bit Shampoo |
| W5 | **Statistical inefficiency of the basis estimator.** Pooled GGᵀ weights columns by raw magnitude; with heavy-tailed spectra the effective sample size collapses | [I] from E2 (KL weighting), proven in `theory/` (Thm 3) | KL-Shampoo (efficient only under separability) |
| W6 | **Kronecker-separable identifiability.** Pooled factors cannot identify a rotation between two directions whose *pooled* variances tie, even if their variance *profiles* differ | [I], proven in `theory/` (Thm 2) | none found |
| W7 | ε/precision sensitivity; small-eigenvalue noise floor | E3, E5 | ε tuning studies |
| W8 | No width transfer of LR without care | E13, E6 | μP scaling, SPlus shape scaling |
| W9 | Relative advantage depends on horizon (weaker at 1× Chinchilla vs Muon) | E10, E12 | — |

## 5. Closest-work matrix (basis estimation is the axis that matters for this project)

| Work | Basis source | Statistical model behind basis | Per-step fresh | QR/eigh needed | Stores L,R | Eigenvalues |
|---|---|---|---|---|---|---|
| SOAP | eig(EMA GGᵀ), eig(EMA GᵀG), power+QR every f | Kronecker product (pooled) | no (f≈10) | yes | yes | Adam (free diagonal) |
| SOAP real-time (NVIDIA) | same, per step incl. current G | Kronecker product (pooled) | yes | yes, every step | yes | Adam |
| KL-SOAP | eig of KL factors G S_b⁻¹Gᵀ | Kronecker product (MLE) | optional | yes | yes | Adam |
| KL-Shampoo | eig of KL factors | Kronecker product (MLE) | optional | yes | yes | λ_a ⊗ λ_b (separable) |
| Purifying Shampoo | eig, adaptive frequency | Kronecker product | adaptive | yes | yes | corrected |
| SPlus | eigh every ~100 steps | Kronecker product | no | yes | yes | sign (none) |
| COSMOS | top-r power iteration | low-rank pooled | no | yes (r-dim QR) | low-rank | Adam on top-r |
| Clean | Nyström sketch | pooled, low rank + residual | — | small | sketches | Adam |
| AdaDiag | SVD of current gradient, every 200–500 steps | instantaneous | no | yes (SVD) | no | Adam |
| ARO | QR of M·f(RᵀM)ᵀ | loss-decrease (Procrustes), not a covariance model | yes | yes (Cholesky-QR) | no | base optimizer |
| TurboSoap | Stiefel tangent / Brockett flow on L,R + Newton–Schulz | Kronecker product (pooled) | yes | no | yes | Adam |
| **Gimbal (this work)** | **natural-gradient flow on O(m)×O(n) of the free-diagonal Gaussian likelihood** | **Kronecker-rotated free diagonal (SOAP's own preconditioner model) = joint diagonalization / CPC** | **yes** | **no (matmul only after init)** | **no** | Adam (free diagonal), consistent with the basis model |

## 6. Contradictions and anomalies

| Observation | Sources in tension | Could differ because | Status |
|---|---|---|---|
| SOAP ≫ AdamW (E1) vs SOAP ≈ AdamW (E14) | E1, E14 | scale (10M vs 360M), batch size, tuning | Treat small-scale null results as regime-limited |
| Per-step refresh needed (E3) vs f=10 fine (E1) | E1, E3 | batch size 2M vs 24–100M tokens, model scale | Freshness matters more at large batch |
| Muon ≥ SOAP at 1× (E10) vs KL-SOAP ≥ Muon (E3) vs KL-Shampoo ≥ Muon (E5) | E3, E5, E10 | estimator variant (KL), tuning, horizon | Better factor estimation closes the Muon gap |

## 7. Saturation map

* **Well covered:** factor *estimators* for the Kronecker-product model (Frobenius, KL/LogDet, von Neumann); refresh frequency; low-rank and sketched factors; eigenvalue correction; sign or orthogonalized updates in the eigenbasis.
* **Repeatedly failed or weak:** naive one-sided bases (E1 §7.1), GaLore-style instantaneous SVD bases (E1 App. B).
* **Unresolved (no work found):** estimating the eigenbasis under the *same* free-diagonal model that SOAP's Adam step assumes; pair-dependent (gap-aware) weighting; matmul-only basis flows that are statistically efficient rather than eigenvector trackers of pooled factors.

## 8. Opportunity tickets (ranked)

### T1 (selected): Self-consistent eigenbasis by online joint diagonalization (Gimbal)
* **Unresolved claim:** SOAP's preconditioner model is P = (Q_L⊗Q_R) diag(D)^{-1/2} (Q_L⊗Q_R)ᵀ with free D, but the basis is estimated under a different (separable) model. The maximum-likelihood basis for SOAP's own model has never been used.
* **Why it matters:** fixes W1–W6 with one mechanism: efficient (Thm 3), identifiable in strictly more cases (Thm 2), per-step fresh, matmul-only, and removes the L, R buffers.
* **Closest work and delta:** KL-SOAP (MLE for the product model, needs factors + QR); TurboSoap (matmul tracking of pooled factors' eigenvectors); ARO (rotation by loss-decrease Procrustes). Delta: the estimating equation (pair-dependent, gap-aware weights from the free diagonal) and the natural-gradient flow on O(m)×O(n) that solves it online.
* **Smallest decisive test:** Monte Carlo of basis KL vs number of samples under controlled non-separability; noisy-quadratic loss with tuned LR; small-LM loss curves.
* **Expected observation if right:** advantage over SOAP/KL-SOAP grows with non-separability, vanishes (vs KL) for exactly separable D.
* **Strongest alternative explanation:** gains come from freshness alone (control: SOAP real-time), or from an implicit LR change (control: LR sweeps).
* **Kill criterion:** on real LM gradients, D is nearly separable *and* SOAP-real-time matches Gimbal's loss at lower cost.
* **Novelty falsifier:** a prior optimizer whose basis update is the CPC/joint-diagonalization estimating equation (pair-dependent inverse-variance weights). None found (Section 9).

### T2: Variance transport under basis change
Doubly-stochastic transport V' = (P_L∘P_L) V (P_R∘P_R)ᵀ. Sub-mechanism of T1; tested as an ablation.

### T3: Wiener-shrunk secant curvature in the eigenbasis
Decision-theoretic step size using regression curvature estimates. Kept as a reserve branch (higher risk: excitation, nonconvexity).

### T4: Residual-stream-shared bases across layers
Pooled statistics and amortized cost. Reserve branch (risk: bias across layers).

## 9. Adversarial novelty pass for T1 (Gimbal)

Searched (2026-10-06): "joint approximate diagonalization optimizer preconditioner", "common principal components optimizer Kronecker", "EASI ... optimizer preconditioning eigenvectors", "Riemannian ... eigenbasis tracking SOAP Shampoo QR-free Cayley natural gradient", "online eigenvector tracking Lie algebra ... optimizer", "SOAP second moment basis change rotate", plus the closest-work traversal above.

| Facet | Gimbal | Closest result | Overlap label |
|---|---|---|---|
| Model for the basis | free-diagonal rotated Gaussian (KRD) | KL-Shampoo: Kronecker product | adjacent |
| Estimating equation | E_ik = Σ_j z_ij z_kj (1/D_kj − 1/D_ij) = 0 | CPC / Pham (statistics, BSS) | component precedent (different field) |
| Solver | natural-gradient flow on O(m)×O(n), Fisher F_ik = Σ_j (D_ij−D_kj)²/(D_ij D_kj) | EASI (relative gradient, ICA); TurboSoap (tangent flow on pooled L) | component precedent / analogous |
| Use as optimizer basis with Adam | yes | SOAP, KL-SOAP | adjacent |
| Removes Kronecker factor buffers | yes | ARO, AdaDiag (no factors, different principle) | adjacent |

**Outcome:** no close overlap found within the searched scope (web search over arXiv, OpenReview, NeurIPS/ICLR/ICML pages, blogs) through 2026-10-06. Patents, non-English venues and closed workshops were not searched. This is "apparently distinct", not a proof of novelty.
