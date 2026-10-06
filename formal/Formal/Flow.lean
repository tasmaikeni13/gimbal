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

/-- Bias-corrected exponential moving average `(∑_{s<t} β^{t-1-s} (1-β) x_s) / (1 - β^t)`. -/
noncomputable def bcEma (β : ℝ) (x : ℕ → ℝ) (t : ℕ) : ℝ :=
  (∑ s ∈ Finset.range t, β ^ (t - 1 - s) * (1 - β) * x s) / (1 - β ^ t)

/-- **Theorem 4.4 (bias-corrected rotation schedule).** The recursion
`θ_{t+1} = θ_t + a_{t+1} (x_t - θ_t)` with `a_t = (1-β)/(1-β^t)` (Gimbal's schedule with
`β = 1 - α`) produces exactly the bias-corrected exponential average of the inputs. In the
linearized frame dynamics the inputs are the per-sample natural-gradient estimates, so the frame
error is the bias-corrected EMA of i.i.d. estimates (a batch average early, a tracker later). -/
theorem bias_corrected_schedule (β : ℝ) (hβ0 : 0 ≤ β) (hβ1 : β < 1) (x : ℕ → ℝ) (t : ℕ) :
    bcEma β x (t + 1) = bcEma β x t + (1 - β) / (1 - β ^ (t + 1)) * (x t - bcEma β x t) := by
  have hc : ∀ k : ℕ, 0 < 1 - β ^ k ∨ k = 0 := by
    intro k
    rcases Nat.eq_zero_or_pos k with h | h
    · exact Or.inr h
    · left
      have : β ^ k < 1 := pow_lt_one₀ hβ0 hβ1 (Nat.pos_iff_ne_zero.mp h)
      linarith
  have hct1 : 0 < 1 - β ^ (t + 1) := by
    rcases hc (t + 1) with h | h
    · exact h
    · exact absurd h (Nat.succ_ne_zero t)
  -- numerator recursion: N_{t+1} = β N_t + (1-β) x_t
  have hnum : (∑ s ∈ Finset.range (t + 1), β ^ (t + 1 - 1 - s) * (1 - β) * x s) =
      β * (∑ s ∈ Finset.range t, β ^ (t - 1 - s) * (1 - β) * x s) + (1 - β) * x t := by
    rw [Finset.sum_range_succ, Finset.mul_sum]
    congr 1
    · refine Finset.sum_congr rfl fun s hs => ?_
      have hs' : s < t := Finset.mem_range.mp hs
      have : t + 1 - 1 - s = (t - 1 - s) + 1 := by omega
      rw [this, pow_succ]
      ring
    · simp
  unfold bcEma
  rw [hnum]
  rcases Nat.eq_zero_or_pos t with h0 | hpos
  · subst h0
    have hb : (1 - β) ≠ 0 := by linarith
    simp
    field_simp
  · have hct : 0 < 1 - β ^ t := by
      have : β ^ t < 1 := pow_lt_one₀ hβ0 hβ1 (Nat.pos_iff_ne_zero.mp hpos)
      linarith
    have h1 := hct1.ne'
    have h2 := hct.ne'
    set N := ∑ s ∈ Finset.range t, β ^ (t - 1 - s) * (1 - β) * x s
    rw [div_eq_iff h1]
    field_simp
    ring

end Gimbal
