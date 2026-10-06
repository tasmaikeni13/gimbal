import Mathlib

/-!
Efficiency of frame estimators (theory/gimbal_theory.md, Theorem 3).

For one pair of rows `(i, k)` of the rotated gradient, write `a j = D i j` and `b j = D k j`
for the variances of the two entries in column `j`, and let `w` be column weights.

* The weighted-factor estimator with weights `w` has asymptotic angle variance
  `V_w = (∑ w² a b) / (∑ w (a - b))²`.
* The maximum-likelihood estimator has asymptotic variance `1 / F` with Fisher information
  `F = ∑ (a - b)² / (a b)`.

This file proves `V_w ≥ 1 / F` for every weight vector (a weighted Cauchy–Schwarz inequality),
that the pair-dependent weights `w = (a - b) / (a b)` attain the bound, that KL-Shampoo's weights
`w = 1 / μ` attain it under separability, and that SOAP's uniform weights lose the factor
`n ∑ μ² / (∑ μ)² ≥ 1` under separability.
-/

open Finset

namespace Gimbal

variable {ι : Type*}

/-- Fisher information of the rotation of one pair (Theorem 2). -/
noncomputable def fisher (s : Finset ι) (a b : ι → ℝ) : ℝ :=
  ∑ j ∈ s, (a j - b j) ^ 2 / (a j * b j)

/-- Asymptotic variance of the weighted-factor estimator (numerator and denominator). -/
noncomputable def wNum (s : Finset ι) (w a b : ι → ℝ) : ℝ := ∑ j ∈ s, w j ^ 2 * (a j * b j)

noncomputable def wGap (s : Finset ι) (w a b : ι → ℝ) : ℝ := ∑ j ∈ s, w j * (a j - b j)

/-- **Weighted Cauchy–Schwarz.** `(∑ w (a - b))² ≤ (∑ w² a b) · F`. -/
theorem weighted_cs (s : Finset ι) (w a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j) :
    wGap s w a b ^ 2 ≤ wNum s w a b * fisher s a b := by
  unfold wGap wNum fisher
  apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul
  · intro j hj
    have := mul_pos (ha j hj) (hb j hj)
    positivity
  · intro j hj
    have := mul_pos (ha j hj) (hb j hj)
    positivity
  · intro j hj
    have ha' := (ha j hj).ne'
    have hb' := (hb j hj).ne'
    apply le_of_eq
    field_simp

/-- **Cramér–Rao for factor estimators.** Every weighted-factor estimator has variance at least
`1 / F`, whenever its eigen-gap is non-zero and the pair is identifiable. -/
theorem weighted_ge_mle (s : Finset ι) (w a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j)
    (hgap : wGap s w a b ≠ 0) (hF : 0 < fisher s a b) :
    1 / fisher s a b ≤ wNum s w a b / wGap s w a b ^ 2 := by
  have h := weighted_cs s w a b ha hb
  have hg2 : 0 < wGap s w a b ^ 2 := by positivity
  rw [div_le_div_iff₀ hF hg2]
  linarith

/-- SOAP / Shampoo use uniform weights (pooled factors). -/
theorem pooled_ge_mle (s : Finset ι) (a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j)
    (hgap : (∑ j ∈ s, (a j - b j)) ≠ 0) (hF : 0 < fisher s a b) :
    1 / fisher s a b ≤ (∑ j ∈ s, a j * b j) / (∑ j ∈ s, (a j - b j)) ^ 2 := by
  have h := weighted_ge_mle s (fun _ => (1 : ℝ)) a b ha hb (by simpa [wGap] using hgap) hF
  simpa [wGap, wNum] using h

/-- The pair-dependent weights `w = (a - b) / (a b)` attain the bound: numerator and gap both
equal `F`, so the variance is exactly `1 / F`. -/
theorem mle_weights_attain (s : Finset ι) (a b : ι → ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j) :
    wGap s (fun j => (a j - b j) / (a j * b j)) a b = fisher s a b ∧
      wNum s (fun j => (a j - b j) / (a j * b j)) a b = fisher s a b := by
  constructor
  · unfold wGap fisher
    refine Finset.sum_congr rfl fun j hj => ?_
    have hab : a j * b j ≠ 0 := (mul_pos (ha j hj) (hb j hj)).ne'
    field_simp
  · unfold wNum fisher
    refine Finset.sum_congr rfl fun j hj => ?_
    have hab : a j * b j ≠ 0 := (mul_pos (ha j hj) (hb j hj)).ne'
    field_simp

/-- Under a separable model `a j = λi μ j`, `b j = λk μ j`, KL-Shampoo's weights `w = 1 / μ`
are efficient: the variance equals `1 / F`. -/
theorem kl_weights_efficient_separable (s : Finset ι) (μ : ι → ℝ) (li lk : ℝ)
    (hμ : ∀ j ∈ s, 0 < μ j) (hi : 0 < li) (hk : 0 < lk) (hne : li ≠ lk) (hs : s.Nonempty) :
    wNum s (fun j => 1 / μ j) (fun j => li * μ j) (fun j => lk * μ j) /
        wGap s (fun j => 1 / μ j) (fun j => li * μ j) (fun j => lk * μ j) ^ 2 =
      1 / fisher s (fun j => li * μ j) (fun j => lk * μ j) := by
  have hcard : (0 : ℝ) < s.card := by exact_mod_cast hs.card_pos
  have hnum : wNum s (fun j => 1 / μ j) (fun j => li * μ j) (fun j => lk * μ j)
      = s.card * (li * lk) := by
    unfold wNum
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j hj => ?_
    have := (hμ j hj).ne'
    field_simp
    push_cast
    ring
  have hgap : wGap s (fun j => 1 / μ j) (fun j => li * μ j) (fun j => lk * μ j)
      = s.card * (li - lk) := by
    unfold wGap
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j hj => ?_
    have := (hμ j hj).ne'
    field_simp
    push_cast
    ring
  have hfis : fisher s (fun j => li * μ j) (fun j => lk * μ j)
      = s.card * ((li - lk) ^ 2 / (li * lk)) := by
    unfold fisher
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j hj => ?_
    have := (hμ j hj).ne'
    have := hi.ne'
    have := hk.ne'
    field_simp
    push_cast
    ring
  rw [hnum, hgap, hfis]
  have hd : li - lk ≠ 0 := sub_ne_zero.mpr hne
  field_simp

/-- Under separability, SOAP's uniform weights lose exactly the factor `n ∑ μ² / (∑ μ)²`
relative to the Cramér–Rao bound. -/
theorem pooled_loss_separable (s : Finset ι) (μ : ι → ℝ) (li lk : ℝ)
    (hμ : ∀ j ∈ s, 0 < μ j) (hi : 0 < li) (hk : 0 < lk) (hne : li ≠ lk) (hs : s.Nonempty) :
    (wNum s (fun _ => 1) (fun j => li * μ j) (fun j => lk * μ j) /
        wGap s (fun _ => 1) (fun j => li * μ j) (fun j => lk * μ j) ^ 2) *
      fisher s (fun j => li * μ j) (fun j => lk * μ j) =
      s.card * (∑ j ∈ s, μ j ^ 2) / (∑ j ∈ s, μ j) ^ 2 := by
  have hsum : 0 < ∑ j ∈ s, μ j := Finset.sum_pos hμ hs
  have hnum : wNum s (fun _ => 1) (fun j => li * μ j) (fun j => lk * μ j)
      = li * lk * ∑ j ∈ s, μ j ^ 2 := by
    unfold wNum
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  have hgap : wGap s (fun _ => 1) (fun j => li * μ j) (fun j => lk * μ j)
      = (li - lk) * ∑ j ∈ s, μ j := by
    unfold wGap
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  have hfis : fisher s (fun j => li * μ j) (fun j => lk * μ j)
      = s.card * ((li - lk) ^ 2 / (li * lk)) := by
    unfold fisher
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun j hj => ?_
    have := (hμ j hj).ne'
    have := hi.ne'
    have := hk.ne'
    field_simp
    push_cast
    ring
  rw [hnum, hgap, hfis]
  have hd : li - lk ≠ 0 := sub_ne_zero.mpr hne
  have := hi.ne'
  have := hk.ne'
  have := hsum.ne'
  field_simp

/-- The separable efficiency loss is at least one (Cauchy–Schwarz), so SOAP's pooled frame is
never more efficient than KL-Shampoo's under separability. -/
theorem pooled_loss_ge_one (s : Finset ι) (μ : ι → ℝ) (hμ : ∀ j ∈ s, 0 < μ j)
    (hs : s.Nonempty) :
    1 ≤ s.card * (∑ j ∈ s, μ j ^ 2) / (∑ j ∈ s, μ j) ^ 2 := by
  have hsum : 0 < ∑ j ∈ s, μ j := Finset.sum_pos hμ hs
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq s (fun _ => (1 : ℝ)) μ
  simp only [one_mul, one_pow, Finset.sum_const, nsmul_eq_mul, mul_one] at hcs
  rw [le_div_iff₀ (by positivity)]
  linarith

/-- The variance of a weighted-factor estimator does not depend on the scale of the weights. -/
theorem variance_scale_invariant (s : Finset ι) (w a b : ι → ℝ) {c : ℝ} (hc : c ≠ 0) :
    wNum s (fun j => c * w j) a b / wGap s (fun j => c * w j) a b ^ 2 =
      wNum s w a b / wGap s w a b ^ 2 := by
  have hnum : wNum s (fun j => c * w j) a b = c ^ 2 * wNum s w a b := by
    unfold wNum; rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_; ring
  have hgap : wGap s (fun j => c * w j) a b = c * wGap s w a b := by
    unfold wGap; rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_; ring
  rw [hnum, hgap]
  by_cases hg : wGap s w a b = 0
  · simp [hg]
  · field_simp

/-- **Theorem 3.3 (sufficiency).** If the inverse variances of the two rows are
`1/a j = β j + αi w j` and `1/b j = β j + αk w j` with `αi ≠ αk` (the additive form
`D⁻¹ = 1 βᵀ + α wᵀ`), then the single weight vector `w` is efficient for this pair. Separable
`D = λ μᵀ` is the case `β = 0`, `w = 1/μ` (KL-Shampoo). -/
theorem efficient_if_additive_inverse (s : Finset ι) (w a b β : ι → ℝ) (αi αk : ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j)
    (hai : ∀ j ∈ s, 1 / a j = β j + αi * w j) (hbk : ∀ j ∈ s, 1 / b j = β j + αk * w j)
    (hα : αi ≠ αk) :
    wNum s w a b / wGap s w a b ^ 2 =
      wNum s (fun j => (a j - b j) / (a j * b j)) a b /
        wGap s (fun j => (a j - b j) / (a j * b j)) a b ^ 2 := by
  have hmle : ∀ j ∈ s, (a j - b j) / (a j * b j) = (αk - αi) * w j := by
    intro j hj
    have ha' := (ha j hj).ne'
    have hb' := (hb j hj).ne'
    have h1 := hai j hj
    have h2 := hbk j hj
    have : (a j - b j) / (a j * b j) = 1 / b j - 1 / a j := by field_simp
    rw [this, h1, h2]; ring
  have hcongr_num : wNum s (fun j => (a j - b j) / (a j * b j)) a b =
      wNum s (fun j => (αk - αi) * w j) a b := by
    unfold wNum; exact Finset.sum_congr rfl fun j hj => by simp only [hmle j hj]
  have hcongr_gap : wGap s (fun j => (a j - b j) / (a j * b j)) a b =
      wGap s (fun j => (αk - αi) * w j) a b := by
    unfold wGap; exact Finset.sum_congr rfl fun j hj => by simp only [hmle j hj]
  rw [hcongr_num, hcongr_gap, variance_scale_invariant s w a b (sub_ne_zero.mpr hα.symm)]

/-- Combining: under the additive-inverse form, `w` attains the Cramér–Rao value `1 / F`. -/
theorem efficient_if_additive_inverse' (s : Finset ι) (w a b β : ι → ℝ) (αi αk : ℝ)
    (ha : ∀ j ∈ s, 0 < a j) (hb : ∀ j ∈ s, 0 < b j)
    (hai : ∀ j ∈ s, 1 / a j = β j + αi * w j) (hbk : ∀ j ∈ s, 1 / b j = β j + αk * w j)
    (hα : αi ≠ αk) (hF : 0 < fisher s a b) :
    wNum s w a b / wGap s w a b ^ 2 = 1 / fisher s a b := by
  rw [efficient_if_additive_inverse s w a b β αi αk ha hb hai hbk hα]
  obtain ⟨h1, h2⟩ := mle_weights_attain s a b ha hb
  rw [h1, h2]
  have := hF.ne'
  field_simp

/-- **Strict separation from every single-factor estimator (SOAP, KL-SOAP, ...).**
Take the non-separable variance array with rows `(1, 1)`, `(2, 1)`, `(1, 2)` (two columns).
For the pair of rows 1–2 every weight vector with `w 1 > 0` (e.g. SOAP's `w = 1`, or any
positive KL weighting) has variance strictly above the Cramér–Rao value `1 / F = 2`, and for the
pair 1–3 every weight vector with `w 0 > 0` does. The likelihood flow attains `1 / F` for both
pairs (`mle_weights_attain`). -/
theorem factor_estimators_strictly_inefficient (w : Fin 2 → ℝ) (h0 : 0 < w 0) (h1 : 0 < w 1) :
    1 / fisher Finset.univ ![1, 1] ![2, 1] <
        wNum Finset.univ w ![1, 1] ![2, 1] / wGap Finset.univ w ![1, 1] ![2, 1] ^ 2 ∧
      1 / fisher Finset.univ ![1, 1] ![1, 2] <
        wNum Finset.univ w ![1, 1] ![1, 2] / wGap Finset.univ w ![1, 1] ![1, 2] ^ 2 := by
  simp only [fisher, wNum, wGap, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
  constructor
  · rw [lt_div_iff₀ (by nlinarith)]
    nlinarith [sq_nonneg (w 1), mul_pos h1 h1]
  · rw [lt_div_iff₀ (by nlinarith)]
    nlinarith [sq_nonneg (w 0), mul_pos h0 h0]

end Gimbal
