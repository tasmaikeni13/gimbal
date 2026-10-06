import Mathlib

/-!
Second-moment transport under a frame change (theory/gimbal_theory.md, Theorem 7).

* `hadamard_sq_kronecker`: `(A ⊗ B) ∘ (A ⊗ B) = (A ∘ A) ⊗ (B ∘ B)`, so the transport of the
  matrix `V` factorizes into two small matrix products.
* `rowsum_hadamard_sq`, `colsum_hadamard_sq`: for orthogonal `P`, the rows and columns of
  `P ∘ P` sum to one (it is doubly stochastic).
* `transport_preserves_total`: the transported variances have the same total.
-/

open Matrix
open scoped Kronecker

namespace Gimbal

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

omit [Fintype m] [Fintype n] [DecidableEq n] in
theorem hadamard_sq_kronecker (A : Matrix m m ℝ) (B : Matrix n n ℝ) :
    (A ⊗ₖ B) ⊙ (A ⊗ₖ B) = (A ⊙ A) ⊗ₖ (B ⊙ B) := by
  ext ⟨i₁, i₂⟩ ⟨j₁, j₂⟩
  simp only [hadamard_apply, kronecker_apply]
  ring

theorem rowsum_hadamard_sq (P : Matrix n n ℝ) (h : P * Pᵀ = 1) (i : n) :
    ∑ k, (P ⊙ P) i k = 1 := by
  have := congrFun (congrFun h i) i
  simp only [mul_apply, transpose_apply, one_apply_eq] at this
  simpa [hadamard_apply] using this

theorem colsum_hadamard_sq (P : Matrix n n ℝ) (h : Pᵀ * P = 1) (k : n) :
    ∑ i, (P ⊙ P) i k = 1 := by
  have := congrFun (congrFun h k) k
  simp only [mul_apply, transpose_apply, one_apply_eq] at this
  simpa [hadamard_apply] using this

/-- The new-frame variances `v'_i = ∑_k P_{ki}² v_k` (Theorem 7) have the same total. -/
theorem transport_preserves_total (P : Matrix n n ℝ) (h : P * Pᵀ = 1)
    (v : n → ℝ) : ∑ i, ∑ k, (P k i) ^ 2 * v k = ∑ k, v k := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  have hk := rowsum_hadamard_sq P h k
  simp only [hadamard_apply] at hk
  rw [← Finset.sum_mul]
  simp only [sq]
  rw [hk, one_mul]

end Gimbal
