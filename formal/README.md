# Lean 4 formalization of the Gimbal theory

Machine-checked parts of `theory/gimbal_theory.md` (evidence label **L5**). Lean `v4.35.0-rc3`,
Mathlib `v4.35.0-rc3` (pinned in `lean-toolchain` and `lake-manifest.json`).

## Build

```bash
cd formal
lake exe cache get   # download prebuilt Mathlib (~5 GB, once)
lake build           # builds Formal/*.lean
lake env lean audit/Axioms.lean   # prints the axioms of every headline theorem
```

Every theorem below depends only on `propext`, `Classical.choice` and `Quot.sound`
(`audit/axioms_output.txt`). There is no `sorry`, `admit` or custom axiom.

## Theorem map

| Theory item | Lean name | File | Statement (informal) |
|---|---|---|---|
| Thm 3.2 | `weighted_cs` | `Efficiency.lean` | $(\sum w(a-b))^2 \le (\sum w^2ab)\,F$ |
| Thm 3.2 | `weighted_ge_mle` | `Efficiency.lean` | every weighted-factor estimator: $V_w \ge 1/F$ |
| Thm 3.2 (SOAP) | `pooled_ge_mle` | `Efficiency.lean` | pooled factors ($w\equiv1$): $V_1\ge 1/F$ |
| Thm 3.4 | `mle_weights_attain` | `Efficiency.lean` | pair-dependent weights $w=(a-b)/(ab)$ give exactly $1/F$ |
| Thm 3.3 (KL) | `kl_weights_efficient_separable` | `Efficiency.lean` | under $D=\lambda\mu^\top$, $w=1/\mu$ gives $1/F$ |
| Thm 3.3 (SOAP) | `pooled_loss_separable`, `pooled_loss_ge_one` | `Efficiency.lean` | SOAP's loss factor is $n\sum\mu^2/(\sum\mu)^2\ge1$ |
| Thm 3.3 (suff.) | `efficient_if_additive_inverse`, `efficient_if_additive_inverse'`, `variance_scale_invariant` | `Efficiency.lean` | $D^{\circ-1}=\mathbf1\beta^\top+\alpha w^\top$ ⇒ $w$ efficient |
| Thm 3.4 (strict) | `factor_estimators_strictly_inefficient` | `Efficiency.lean` | a non-separable $D$ on which **every** positive single weighting (SOAP, KL-SOAP, …) is strictly above Cramér–Rao for two pairs |
| Thm 2 | `fisher_pos_iff`, `fisher_nonneg` | `Identifiability.lean` | $F>0$ iff the profiles differ |
| Thm 2 | `pooled_implies_profile` | `Identifiability.lean` | distinct row sums ⇒ distinct profiles |
| Thm 2 (strict) | `tie_but_identifiable` | `Identifiability.lean` | tied row sums, $F>0$ |
| Example 1 | `tie_cost_pos`, `tie_cost_unbounded` | `Identifiability.lean` | the cost of a tie is positive and unbounded |
| Thm 5.1 | `cayley_orthogonal`, `one_sub_skew_det_ne_zero`, `dotProduct_skew_self` | `Retraction.lean` | Cayley transform of a real skew matrix is orthogonal |
| Thm 5.2 | `expm2_defect` | `Retraction.lean` | $X^\top X = I + \Omega^4/4$ for $X=I+\Omega+\Omega^2/2$ |
| Thm 5.3 | `ns_error` | `Retraction.lean` | $Y^\top Y - I = -\tfrac34E^2+\tfrac14E^3$ |
| Thm 7 | `hadamard_sq_kronecker` | `Transport.lean` | $(A\otimes B)^{\circ2}=A^{\circ2}\otimes B^{\circ2}$ |
| Thm 7 | `rowsum_hadamard_sq`, `colsum_hadamard_sq` | `Transport.lean` | $P^{\circ2}$ is doubly stochastic for orthogonal $P$ |
| Thm 7 | `transport_preserves_total` | `Transport.lean` | transport preserves total second moment |
| Thm 6 | `rotated_coords_invariant`, `update_covariant` | `Equivariance.lean` | one-step equivariance under $(P,R)\in O(m)\times O(n)$ |
| Thm 8.1 | `frobenius_adjoint`, `descent`, `descent_strict` | `Equivariance.lean` | $\langle G,U\rangle=\sum Z^2/c\ge0$, $>0$ unless $Z=0$ |
| Thm 8.2 | `generator_scale_invariant`, `fisher_scale_invariant` | `Equivariance.lean` | the frame flow ignores gradient scale |
| §5 | `fisher_term_identity` | `Flow.lean` | $x/y+y/x-2=(x-y)^2/(xy)$ (matmul formula for $F$) |
| Thm 2 | `score_variance_term` | `Flow.lean` | score variance summand equals Fisher summand |
| Thm 4.1 | `stationary_generator` | `Flow.lean` | zero cross moments ⇒ zero expected generator |
| Thm 4.3 | `variance_recursion_closed_form`, `variance_recursion_tendsto` | `Flow.lean` | angle-error variance → $\alpha\sigma^2/(2-\alpha)$ geometrically |
| Thm 4.4 | `bias_corrected_schedule` | `Flow.lean` | the schedule $\alpha_t=\alpha/(1-(1-\alpha)^t)$ yields exactly the bias-corrected EMA |
| Prop. 1 | `eigenvalue_cost_nonneg`, `eigenvalue_cost_eq_iff` | `Flow.lean` | $d/D+\log D\ge1+\log d$, equality iff $D=d$ |
| Lemma 5.4.1 | `weighted_profile_variance`, `weighted_profile_variance_eq_iff` | `Variance.lean` | with forgetting weights, the weighted likelihood of a variance is maximized exactly at the weighted average (the tied flow variance is the profile-likelihood variance) |
| Lemma 5.4.2 | `excess_variance_identity`, `excess_component_orthogonal` | `Variance.lean` | $\|w\|^2/\langle w,v\rangle^2 = 1/F + \|u\|^2/\langle w,v\rangle^2$, $u\perp v$: the exact cost of non-efficient (e.g. noisy plug-in) weights |
| Prop. 5.5 | `shrinkage_risk_ge`, `shrinkage_risk_eq_iff` | `Variance.lean` | $(c-1)^2S+c^2\nu\ge S\nu/(S+\nu)$, equality iff $c=S/(S+\nu)$ (optimal shrinkage factor) |
| Remark 5.6 | `restart_closed_form` | `Variance.lean` | after a restart at $T$, the bias-corrected schedule gives the restart value exactly the weight $1-\beta^T$ of the inputs it replaces |
| §3 (E2.10 a) | `pair_rotation_product`, `pair_rotation_cost_nonneg`, `pair_rotation_cost_le`, `pair_rotation_cost_sum_le` | `PairCost.lean` | a single-pair rotation changes column $j$'s variances to $c^2a+s^2b$, $s^2a+c^2b$, with product $ab+c^2s^2(a-b)^2$; its frame KL lies in $[0,\,c^2s^2F]$ |
| §3 (E2.10 a) | `sin_sq_mul_cos_sq_le`, `pair_rotation_cost_le_half_fisher_sq` | `PairCost.lean` | the frame KL of a rotation by $\theta$ in one pair is at most $\tfrac12F\theta^2$ (equality to second order) |
| Prop. 9 | `gimbal_state_le_soap` | `PairCost.lean` | $m^2+n^2+4mn\le2m^2+2n^2+2mn$: Gimbal's optimizer state never exceeds SOAP's |

## What is *not* formalized (and why)

* Asymptotic normality / delta method (Theorem 3.1) and the Fisher identity at a displaced
  parameter (Theorem 4.2): they need probability-theoretic infrastructure (CLT, differentiation
  under the integral) beyond the scope of this phase. They are proved in the theory document (L3)
  and checked by Monte Carlo in `experiments/phase1/check_identities.py`.
* Hadamard's determinant inequality (used in Proposition 1's non-negativity): not in Mathlib at
  this version; cited.
* Conjecture 4.1 (global convergence) is open; E2.11 gives numerical evidence only.
* Expansions in a small parameter (Lemma 5.4.2's second-order plug-in term, Proposition 4.5's
  Hessian as a second derivative) are proved on paper and checked numerically (E2.10 (b), (e)). Lean
  covers the exact identities they start from: `excess_variance_identity` and
  `pair_rotation_product`, with the bound `pair_rotation_cost_le_half_fisher_sq`.
* Statistical properties of the split-sample noise estimate (Proposition 5.5) are checked by Monte
  Carlo (E2.10 (c)); Lean covers the optimal shrinkage factor (`shrinkage_risk_eq_iff`).

## Fidelity audit

Each Lean statement was compared with its natural-language counterpart: the finite-sum lemmas take
the per-pair variance profiles `a = D_i·`, `b = D_k·` as arbitrary positive functions on a finite
index set, which is exactly the setting of Theorems 2–3; matrix lemmas are stated for arbitrary
finite index types and real entries. `factor_estimators_strictly_inefficient` instantiates a
concrete 3 × 2 array (rows `(1,1)`, `(2,1)`, `(1,2)`), stated through its two row pairs.
