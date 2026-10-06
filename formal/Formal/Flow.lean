import Mathlib

/-!
The frame flow's statistics (theory/gimbal_theory.md, Lemma A, Theorems 2 and 4, Prop. 1).

* `fisher_term_identity`: `x/y + y/x - 2 = (x - y)²/(x y)`, which is why the matrix-product
  formula `D Aᵀ + A Dᵀ - 2n` computes the Fisher information of Theorem 2.
* `score_variance_term`: with `E[z_i² z_k²] = d_i d_k`, one summand of the score variance is
  `d_i d_k (1/d_k - 1/d_i)² = (d_i - d_k)²/(d_i d_k)`.
* `stationary_generator`: if the cross moments vanish, the expected generator is zero.
* `variance_recursion_closed_form`, `variance_recursion_tendsto`: the linearized angle error
  `θ ↦ (1-α) θ + α ξ` has second moment converging geometrically to `α σ² / (2 - α)`.
* `eigenvalue_cost_nonneg`: for `d, D > 0`, `d/D + log D ≥ 1 + log d`, with equality iff `D = d`;
  this is the per-coordinate KL that makes Adam's EMA the right eigenvalue estimator for a given
  frame (Prop. 1).
-/

namespace Gimbal

theorem fisher_term_identity {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) :
    x / y + y / x - 2 = (x - y) ^ 2 / (x * y) := by
  field_simp
  ring

theorem score_variance_term {di dk : ℝ} (hi : di ≠ 0) (hk : dk ≠ 0) :
    di * dk * (1 / dk - 1 / di) ^ 2 = (di - dk) ^ 2 / (di * dk) := by
  field_simp

theorem stationary_generator {ι : Type*} (s : Finset ι) (cross weight : ι → ℝ)
    (h : ∀ j ∈ s, cross j = 0) : ∑ j ∈ s, cross j * weight j = 0 :=
  Finset.sum_eq_zero fun j hj => by rw [h j hj, zero_mul]

/-- Second-moment recursion of the linearized angle error. -/
def varRec (α σsq : ℝ) (v0 : ℝ) : ℕ → ℝ
  | 0 => v0
  | t + 1 => (1 - α) ^ 2 * varRec α σsq v0 t + α ^ 2 * σsq

theorem variance_recursion_closed_form (α σsq v0 : ℝ) (hα : α ≠ 2) (t : ℕ) :
    varRec α σsq v0 t - α * σsq / (2 - α) =
      ((1 - α) ^ 2) ^ t * (v0 - α * σsq / (2 - α)) := by
  have h2 : (2 - α) ≠ 0 := sub_ne_zero.mpr (Ne.symm hα)
  induction t with
  | zero => simp [varRec]
  | succ t ih =>
    have hstep : varRec α σsq v0 (t + 1) - α * σsq / (2 - α) =
        (1 - α) ^ 2 * (varRec α σsq v0 t - α * σsq / (2 - α)) := by
      change (1 - α) ^ 2 * varRec α σsq v0 t + α ^ 2 * σsq - α * σsq / (2 - α) = _
      field_simp
      ring
    rw [hstep, ih, pow_succ]
    ring

theorem variance_recursion_tendsto (α σsq v0 : ℝ) (h0 : 0 < α) (h1 : α < 2) :
    Filter.Tendsto (varRec α σsq v0) Filter.atTop (nhds (α * σsq / (2 - α))) := by
  have hρ0 : 0 ≤ (1 - α) ^ 2 := sq_nonneg _
  have hρ1 : (1 - α) ^ 2 < 1 := by nlinarith
  have hgeom : Filter.Tendsto (fun t : ℕ => ((1 - α) ^ 2) ^ t * (v0 - α * σsq / (2 - α)))
      Filter.atTop (nhds (0 * (v0 - α * σsq / (2 - α)))) :=
    (tendsto_pow_atTop_nhds_zero_of_lt_one hρ0 hρ1).mul_const _
  rw [zero_mul] at hgeom
  have hclosed : ∀ t, varRec α σsq v0 t =
      ((1 - α) ^ 2) ^ t * (v0 - α * σsq / (2 - α)) + α * σsq / (2 - α) := by
    intro t
    have := variance_recursion_closed_form α σsq v0 (by linarith) t
    linarith
  have hfun : varRec α σsq v0 =
      fun t => ((1 - α) ^ 2) ^ t * (v0 - α * σsq / (2 - α)) + α * σsq / (2 - α) :=
    funext hclosed
  rw [hfun]
  simpa using hgeom.add_const (α * σsq / (2 - α))

/-- Per-coordinate eigenvalue cost (Prop. 1): `d / D + log D ≥ 1 + log d`. -/
theorem eigenvalue_cost_nonneg {d D : ℝ} (hd : 0 < d) (hD : 0 < D) :
    1 + Real.log d ≤ d / D + Real.log D := by
  have h := Real.log_le_sub_one_of_pos (div_pos hd hD)
  rw [Real.log_div hd.ne' hD.ne'] at h
  linarith

theorem eigenvalue_cost_eq_iff {d D : ℝ} (hd : 0 < d) (hD : 0 < D) :
    d / D + Real.log D = 1 + Real.log d ↔ D = d := by
  constructor
  · intro h
    by_contra hne
    have hx : 0 < d / D := div_pos hd hD
    have hx1 : d / D ≠ 1 := by
      intro h1
      rw [div_eq_one_iff_eq hD.ne'] at h1
      exact hne h1.symm
    have hlt := Real.log_lt_sub_one_of_pos hx hx1
    rw [Real.log_div hd.ne' hD.ne'] at hlt
    linarith
  · rintro rfl
    rw [div_self hD.ne']

end Gimbal
