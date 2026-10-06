import Formal.Efficiency

/-!
Identifiability (theory/gimbal_theory.md, Theorem 2 and Example 1).

The likelihood identifies the rotation of a pair of rows iff their variance *profiles* differ;
pooled Kronecker factors identify it iff their *row sums* differ. The first condition is implied
by, and strictly weaker than, the second.
-/

open Finset

namespace Gimbal

variable {ι : Type*}

theorem fisher_term_nonneg {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : 0 ≤ (x - y) ^ 2 / (x * y) := by
  have := mul_pos hx hy
  positivity

theorem fisher_nonneg (s : Finset ι) (a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j) : 0 ≤ fisher s a b :=
  Finset.sum_nonneg fun j hj => fisher_term_nonneg (ha j hj) (hb j hj)

/-- **Identifiability.** The Fisher information of a pair rotation is positive iff the two
variance profiles differ somewhere. -/
theorem fisher_pos_iff (s : Finset ι) (a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j) :
    0 < fisher s a b ↔ ∃ j ∈ s, a j ≠ b j := by
  constructor
  · intro hpos
    by_contra hcon
    push Not at hcon
    have : fisher s a b = 0 := by
      unfold fisher
      refine Finset.sum_eq_zero fun j hj => ?_
      simp [hcon j hj]
    linarith
  · rintro ⟨j, hj, hne⟩
    unfold fisher
    have hterm : 0 < (a j - b j) ^ 2 / (a j * b j) := by
      have hab := mul_pos (ha j hj) (hb j hj)
      have : 0 < (a j - b j) ^ 2 := by
        have : a j - b j ≠ 0 := sub_ne_zero.mpr hne
        positivity
      positivity
    exact lt_of_lt_of_le hterm
      (Finset.single_le_sum (f := fun j => (a j - b j) ^ 2 / (a j * b j))
        (fun i hi => fisher_term_nonneg (ha i hi) (hb i hi)) hj)

/-- Distinct pooled variances (row sums) imply distinct profiles. -/
theorem pooled_implies_profile (s : Finset ι) (a b : ι → ℝ)
    (h : (∑ j ∈ s, a j) ≠ ∑ j ∈ s, b j) : ∃ j ∈ s, a j ≠ b j := by
  by_contra hcon
  push Not at hcon
  exact h (Finset.sum_congr rfl hcon)

/-- **Strictness.** A tie in pooled variances with different profiles: pooled factors cannot
identify the rotation, the likelihood can. -/
theorem tie_but_identifiable :
    ∃ a b : Fin 2 → ℝ, (∀ j, 0 < a j) ∧ (∀ j, 0 < b j) ∧
      (∑ j, a j) = (∑ j, b j) ∧ 0 < fisher Finset.univ a b := by
  refine ⟨![2, 1], ![1, 2], ?_, ?_, ?_, ?_⟩
  · intro j; fin_cases j <;> norm_num
  · intro j; fin_cases j <;> norm_num
  · simp [Fin.sum_univ_two]; norm_num
  · unfold fisher
    simp [Fin.sum_univ_two]
    norm_num

/-- **Example 1, cost of a tie.** Rotating a tied pair by 45° replaces the variances `a, b` of a
column by their mean in both rows. The per-column frame cost `log(((a+b)/2)² / (a b))` is
positive whenever `a ≠ b` ... -/
theorem tie_cost_pos {a b : ℝ} (hne : a ≠ b) :
    a * b < ((a + b) / 2) ^ 2 := by
  have : 0 < (a - b) ^ 2 := by
    have : a - b ≠ 0 := sub_ne_zero.mpr hne
    positivity
  nlinarith

/-- ... and unbounded: the ratio `((a+b)/2)² / (a b)` exceeds any bound as `a / b → ∞`. -/
theorem tie_cost_unbounded (M : ℝ) : ∃ a : ℝ, 0 < a ∧ M < ((a + 1) / 2) ^ 2 / (a * 1) := by
  refine ⟨4 * |M| + 4, by positivity, ?_⟩
  have hpos : (0 : ℝ) < 4 * |M| + 4 := by positivity
  rw [mul_one, lt_div_iff₀ hpos]
  nlinarith [abs_nonneg M, le_abs_self M, neg_abs_le M]

end Gimbal
