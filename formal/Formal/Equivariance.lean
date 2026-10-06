import Mathlib

/-!
Equivariance and descent (theory/gimbal_theory.md, Theorems 6 and 8).

* `rotated_coords_invariant`: rotating the parameter space by orthogonal `(P, R)` and the frame
  with it leaves the rotated gradient `Q_Lᵀ G Q_R` unchanged; every quantity Gimbal computes from
  it (second moment, generator, Fisher normalizer) is therefore unchanged.
* `update_covariant`: the update transforms as `ΔW ↦ P ΔW Rᵀ`.
* `frobenius_adjoint`: `⟨G, Q_L N Q_Rᵀ⟩ = ⟨Q_Lᵀ G Q_R, N⟩` (no orthogonality needed).
* `descent`: with `N = Z / c` for a positive array `c`, the update is a descent direction, strictly
  unless `Z = 0`.
* `generator_scale_invariant`, `fisher_scale_invariant`: the frame flow ignores gradient scale.
-/

open Matrix

namespace Gimbal

variable {m n : Type*} [Fintype m] [Fintype n]

theorem rotated_coords_invariant (P : Matrix m m ℝ) (R : Matrix n n ℝ)
    (QL : Matrix m m ℝ) (QR : Matrix n n ℝ) (G : Matrix m n ℝ)
    [DecidableEq m] [DecidableEq n] (hP : Pᵀ * P = 1) (hR : Rᵀ * R = 1) :
    (P * QL)ᵀ * (P * G * Rᵀ) * (R * QR) = QLᵀ * G * QR := by
  rw [transpose_mul]
  calc QLᵀ * Pᵀ * (P * G * Rᵀ) * (R * QR)
      = QLᵀ * (Pᵀ * P) * G * (Rᵀ * R) * QR := by simp only [Matrix.mul_assoc]
    _ = QLᵀ * G * QR := by rw [hP, hR, Matrix.mul_one, Matrix.mul_one]

theorem update_covariant (P : Matrix m m ℝ) (R : Matrix n n ℝ)
    (QL : Matrix m m ℝ) (QR : Matrix n n ℝ) (N : Matrix m n ℝ) :
    (P * QL) * N * (R * QR)ᵀ = P * (QL * N * QRᵀ) * Rᵀ := by
  rw [transpose_mul]
  simp only [Matrix.mul_assoc]

/-- Frobenius inner product of two `m × n` arrays. -/
def frob (A B : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, A i j * B i j

theorem frob_eq_trace (A B : Matrix m n ℝ) : frob A B = trace (Aᵀ * B) := by
  simp only [frob, trace, diag, mul_apply, transpose_apply]
  rw [Finset.sum_comm]

theorem frobenius_adjoint (G : Matrix m n ℝ) (QL : Matrix m m ℝ) (QR : Matrix n n ℝ)
    (N : Matrix m n ℝ) : frob G (QL * N * QRᵀ) = frob (QLᵀ * G * QR) N := by
  rw [frob_eq_trace, frob_eq_trace, transpose_mul, transpose_mul, transpose_transpose]
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, trace_mul_comm]
  simp only [Matrix.mul_assoc]

/-- **Descent.** `⟨G, Q_L (Z ⊘ c) Q_Rᵀ⟩ = ∑ Z² / c ≥ 0` for the rotated gradient `Z`. -/
theorem descent (G : Matrix m n ℝ) (QL : Matrix m m ℝ) (QR : Matrix n n ℝ)
    (c : Matrix m n ℝ) (hc : ∀ i j, 0 < c i j) :
    0 ≤ frob G (QL * Matrix.of (fun i j => (QLᵀ * G * QR) i j / c i j) * QRᵀ) := by
  rw [frobenius_adjoint]
  unfold frob
  refine Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_
  simp only [of_apply]
  rw [← mul_div_assoc]
  exact div_nonneg (mul_self_nonneg _) (hc i j).le

theorem descent_strict (G : Matrix m n ℝ) (QL : Matrix m m ℝ) (QR : Matrix n n ℝ)
    (c : Matrix m n ℝ) (hc : ∀ i j, 0 < c i j) (i : m) (j : n)
    (hz : (QLᵀ * G * QR) i j ≠ 0) :
    0 < frob G (QL * Matrix.of (fun i j => (QLᵀ * G * QR) i j / c i j) * QRᵀ) := by
  rw [frobenius_adjoint]
  unfold frob
  simp only [of_apply]
  have hterm : ∀ i' j', 0 ≤ (QLᵀ * G * QR) i' j' * ((QLᵀ * G * QR) i' j' / c i' j') := by
    intro i' j'
    rw [← mul_div_assoc]
    exact div_nonneg (mul_self_nonneg _) (hc i' j').le
  have hij : 0 < (QLᵀ * G * QR) i j * ((QLᵀ * G * QR) i j / c i j) := by
    rw [← mul_div_assoc]
    exact div_pos (mul_self_pos.mpr hz) (hc i j)
  have hrow : 0 < ∑ j', (QLᵀ * G * QR) i j' * ((QLᵀ * G * QR) i j' / c i j') :=
    lt_of_lt_of_le hij (Finset.single_le_sum (fun j' _ => hterm i j') (Finset.mem_univ j))
  exact lt_of_lt_of_le hrow (Finset.single_le_sum
    (f := fun i' => ∑ j', (QLᵀ * G * QR) i' j' * ((QLᵀ * G * QR) i' j' / c i' j'))
    (fun i' _ => Finset.sum_nonneg fun j' _ => hterm i' j') (Finset.mem_univ i))

omit [Fintype n] in
/-- One entry of the left generator, `E_ik = ∑_j z_ij z_kj (1/d_kj - 1/d_ij)`, is unchanged when
gradients scale by `c ≠ 0` and variances by `c²`. -/
theorem generator_scale_invariant (s : Finset n) (zi zk di dk : n → ℝ) {c : ℝ} (hc : c ≠ 0)
    (hdi : ∀ j ∈ s, di j ≠ 0) (hdk : ∀ j ∈ s, dk j ≠ 0) :
    ∑ j ∈ s, (c * zi j) * (c * zk j) * (1 / (c ^ 2 * dk j) - 1 / (c ^ 2 * di j)) =
      ∑ j ∈ s, zi j * zk j * (1 / dk j - 1 / di j) := by
  refine Finset.sum_congr rfl fun j hj => ?_
  have := hdi j hj
  have := hdk j hj
  field_simp

omit [Fintype n] in
/-- The Fisher normalizer `∑_j (d_ij/d_kj + d_kj/d_ij - 2)` is scale-free as well. -/
theorem fisher_scale_invariant (s : Finset n) (di dk : n → ℝ) {c : ℝ} (hc : c ≠ 0)
    (hdi : ∀ j ∈ s, di j ≠ 0) (hdk : ∀ j ∈ s, dk j ≠ 0) :
    ∑ j ∈ s, ((c ^ 2 * di j) / (c ^ 2 * dk j) + (c ^ 2 * dk j) / (c ^ 2 * di j) - 2) =
      ∑ j ∈ s, (di j / dk j + dk j / di j - 2) := by
  refine Finset.sum_congr rfl fun j hj => ?_
  have := hdi j hj
  have := hdk j hj
  field_simp

end Gimbal
