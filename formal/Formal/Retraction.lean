import Mathlib

/-!
Retractions on the orthogonal group (theory/gimbal_theory.md, Theorem 5).

* `expm2_defect`: for skew `Ω`, `X = 1 + Ω + Ω²/2` satisfies `Xᵀ X = 1 + Ω⁴/4`.
* `ns_error`: one Newton–Schulz step `Y = X (3/2 - XᵀX/2)` maps the defect `E = XᵀX - 1` to
  `YᵀY - 1 = -(3/4) E² + (1/4) E³`.
* `cayley_orthogonal`: for real skew `Ω`, `1 - Ω` is invertible and `(1 - Ω)⁻¹ (1 + Ω)` is
  orthogonal.
-/

open Matrix

namespace Gimbal

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The second-order exponential retraction loses orthogonality only at fourth order. -/
theorem expm2_defect (Ω : Matrix n n ℝ) (hΩ : Ωᵀ = -Ω) :
    (1 + Ω + (1 / 2 : ℝ) • (Ω * Ω))ᵀ * (1 + Ω + (1 / 2 : ℝ) • (Ω * Ω)) =
      1 + (1 / 4 : ℝ) • (Ω * Ω * Ω * Ω) := by
  have hsq : (Ω * Ω)ᵀ = Ω * Ω := by
    rw [transpose_mul, hΩ, neg_mul_neg]
  rw [transpose_add, transpose_add, transpose_one, transpose_smul, hsq, hΩ]
  simp only [add_mul, mul_add, neg_mul, one_mul, mul_one, smul_mul_assoc, mul_smul_comm,
    mul_assoc]
  module

/-- Quadratic convergence of the Newton–Schulz polish. -/
theorem ns_error (X : Matrix n n ℝ) :
    let G := Xᵀ * X
    let Y := X * ((3 / 2 : ℝ) • (1 : Matrix n n ℝ) - (1 / 2 : ℝ) • G)
    Yᵀ * Y - 1 =
      -(3 / 4 : ℝ) • ((G - 1) * (G - 1)) + (1 / 4 : ℝ) • ((G - 1) * (G - 1) * (G - 1)) := by
  intro G Y
  have hG : Gᵀ = G := by simp [G, transpose_mul]
  have hY : Yᵀ * Y = ((3 / 2 : ℝ) • (1 : Matrix n n ℝ) - (1 / 2 : ℝ) • G) * G *
      ((3 / 2 : ℝ) • (1 : Matrix n n ℝ) - (1 / 2 : ℝ) • G) := by
    simp only [Y, transpose_mul, transpose_sub, transpose_smul, transpose_one, hG, G]
    simp only [Matrix.mul_assoc]
  rw [hY]
  simp only [sub_mul, mul_sub, smul_mul_assoc, mul_smul_comm, smul_smul, one_mul, mul_one,
    Matrix.mul_assoc, smul_sub]
  module

omit [DecidableEq n] in
/-- `vᵀ Ω v = 0` for a real skew-symmetric `Ω`. -/
theorem dotProduct_skew_self (Ω : Matrix n n ℝ) (hΩ : Ωᵀ = -Ω) (v : n → ℝ) :
    v ⬝ᵥ (Ω *ᵥ v) = 0 := by
  have h : v ⬝ᵥ (Ω *ᵥ v) = -(v ⬝ᵥ (Ω *ᵥ v)) := by
    conv_lhs => rw [dotProduct_mulVec, ← mulVec_transpose, hΩ, neg_mulVec, neg_dotProduct,
      dotProduct_comm]
  linarith

/-- `1 - Ω` is invertible for real skew `Ω`. -/
theorem one_sub_skew_det_ne_zero (Ω : Matrix n n ℝ) (hΩ : Ωᵀ = -Ω) : (1 - Ω).det ≠ 0 := by
  intro hdet
  obtain ⟨v, hv, hker⟩ := (exists_mulVec_eq_zero_iff).mpr hdet
  have h1 : v ⬝ᵥ ((1 - Ω) *ᵥ v) = 0 := by rw [hker, dotProduct_zero]
  rw [sub_mulVec, one_mulVec, dotProduct_sub, dotProduct_skew_self Ω hΩ v, sub_zero] at h1
  exact hv (dotProduct_self_eq_zero.mp h1)

/-- **Cayley transform.** For real skew `Ω`, `C = (1 - Ω)⁻¹ (1 + Ω)` is orthogonal. -/
theorem cayley_orthogonal (Ω : Matrix n n ℝ) (hΩ : Ωᵀ = -Ω) :
    ((1 - Ω)⁻¹ * (1 + Ω))ᵀ * ((1 - Ω)⁻¹ * (1 + Ω)) = 1 := by
  have hA : IsUnit (1 - Ω).det := (one_sub_skew_det_ne_zero Ω hΩ).isUnit
  have hnegΩ : (-Ω)ᵀ = -(-Ω) := by rw [transpose_neg, hΩ]
  have hB : IsUnit (1 + Ω).det := by
    have := one_sub_skew_det_ne_zero (-Ω) hnegΩ
    rw [sub_neg_eq_add] at this
    exact this.isUnit
  have hAt : (1 - Ω)ᵀ = 1 + Ω := by rw [transpose_sub, transpose_one, hΩ, sub_neg_eq_add]
  have hBt : (1 + Ω)ᵀ = 1 - Ω := by rw [transpose_add, transpose_one, hΩ, ← sub_eq_add_neg]
  have hcomm : (1 - Ω) * (1 + Ω) = (1 + Ω) * (1 - Ω) := by noncomm_ring
  -- (1+Ω)⁻¹ (1-Ω)⁻¹ = (1-Ω)⁻¹ (1+Ω)⁻¹ because the two factors commute.
  have hinv : (1 + Ω)⁻¹ * (1 - Ω)⁻¹ = (1 - Ω)⁻¹ * (1 + Ω)⁻¹ := by
    rw [← Matrix.mul_inv_rev, ← Matrix.mul_inv_rev, hcomm]
  rw [transpose_mul, hBt, transpose_nonsing_inv, hAt]
  calc (1 - Ω) * (1 + Ω)⁻¹ * ((1 - Ω)⁻¹ * (1 + Ω))
      = (1 - Ω) * ((1 + Ω)⁻¹ * (1 - Ω)⁻¹) * (1 + Ω) := by simp only [Matrix.mul_assoc]
    _ = (1 - Ω) * ((1 - Ω)⁻¹ * (1 + Ω)⁻¹) * (1 + Ω) := by rw [hinv]
    _ = ((1 - Ω) * (1 - Ω)⁻¹) * ((1 + Ω)⁻¹ * (1 + Ω)) := by simp only [Matrix.mul_assoc]
    _ = 1 := by rw [mul_nonsing_inv _ hA, nonsing_inv_mul _ hB, one_mul]

end Gimbal
