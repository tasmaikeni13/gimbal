import Mathlib

/-!
The frame KL of a single-pair rotation, and the memory claim of the cost table
(theory/gimbal_theory.md §2–3 and Proposition 9; used by E2.10 (a) and E2.8).

Rotating rows `i, k` of the true frame by an angle with cosine `c` and sine `s` (`c² + s² = 1`)
turns the variances `a = D_ij`, `b = D_kj` of column `j` into `c² a + s² b` and `s² a + c² b`; the
column's contribution to the frame KL `J` (Proposition 1) is half the log of their product over
`a b`.

* `pair_rotation_product`: `(c² a + s² b)(s² a + c² b) = a b + c² s² (a - b)²`.
* `pair_rotation_cost_nonneg`, `pair_rotation_cost_le`: each column's term lies in
  `[0, c² s² (a - b)² / (a b)]`.
* `pair_rotation_cost_sum_le`: summed over columns, the pair's term is at most `c² s² F` with
  `F = Σ (a - b)² / (a b)`, the Fisher information of the pair (Theorem 2).
* `sin_sq_mul_cos_sq_le`: `sin² θ cos² θ ≤ θ²`; together, `J ≤ ½ F θ²` for a rotation by `θ`,
  with equality to second order (the expansion E2.10 (a) uses).
* `gimbal_state_le_soap`: `m² + n² + 4 m n ≤ 2 m² + 2 n² + 2 m n`: Gimbal's optimizer state
  (two frames and four `m × n` buffers) never exceeds SOAP's (two factors, two frames, two buffers).
-/

open Finset

namespace Gimbal

theorem pair_rotation_product (a b c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (c ^ 2 * a + s ^ 2 * b) * (s ^ 2 * a + c ^ 2 * b) = a * b + c ^ 2 * s ^ 2 * (a - b) ^ 2 := by
  linear_combination (a * b * (c ^ 2 + s ^ 2 + 1)) * h

theorem pair_rotation_cost_nonneg {a b : ℝ} (c s : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : c ^ 2 + s ^ 2 = 1) :
    0 ≤ Real.log ((c ^ 2 * a + s ^ 2 * b) * (s ^ 2 * a + c ^ 2 * b) / (a * b)) := by
  rw [pair_rotation_product a b c s h]
  apply Real.log_nonneg
  rw [le_div_iff₀ (mul_pos ha hb)]
  nlinarith [sq_nonneg (c * s * (a - b))]

theorem pair_rotation_cost_le {a b : ℝ} (c s : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : c ^ 2 + s ^ 2 = 1) :
    Real.log ((c ^ 2 * a + s ^ 2 * b) * (s ^ 2 * a + c ^ 2 * b) / (a * b)) ≤
      c ^ 2 * s ^ 2 * (a - b) ^ 2 / (a * b) := by
  rw [pair_rotation_product a b c s h]
  have hab : 0 < a * b := mul_pos ha hb
  have hx : 0 < (a * b + c ^ 2 * s ^ 2 * (a - b) ^ 2) / (a * b) := by
    apply div_pos _ hab
    nlinarith [sq_nonneg (c * s * (a - b))]
  calc Real.log ((a * b + c ^ 2 * s ^ 2 * (a - b) ^ 2) / (a * b))
      ≤ (a * b + c ^ 2 * s ^ 2 * (a - b) ^ 2) / (a * b) - 1 := Real.log_le_sub_one_of_pos hx
    _ = c ^ 2 * s ^ 2 * (a - b) ^ 2 / (a * b) := by
      field_simp
      ring

theorem pair_rotation_cost_sum_le {ι : Type*} (S : Finset ι) (a b : ι → ℝ) (c s : ℝ)
    (ha : ∀ j ∈ S, 0 < a j) (hb : ∀ j ∈ S, 0 < b j) (h : c ^ 2 + s ^ 2 = 1) :
    ∑ j ∈ S, Real.log ((c ^ 2 * a j + s ^ 2 * b j) * (s ^ 2 * a j + c ^ 2 * b j) / (a j * b j)) ≤
      c ^ 2 * s ^ 2 * ∑ j ∈ S, (a j - b j) ^ 2 / (a j * b j) := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro j hj
  exact (pair_rotation_cost_le c s (ha j hj) (hb j hj) h).trans (le_of_eq (by ring))

theorem sin_sq_mul_cos_sq_le (θ : ℝ) : Real.sin θ ^ 2 * Real.cos θ ^ 2 ≤ θ ^ 2 := by
  have h := Real.sin_sq_le_sq (x := 2 * θ)
  rw [Real.sin_two_mul] at h
  have e1 : (2 * Real.sin θ * Real.cos θ) ^ 2 = 4 * (Real.sin θ ^ 2 * Real.cos θ ^ 2) := by ring
  have e2 : (2 * θ) ^ 2 = 4 * θ ^ 2 := by ring
  linarith

/-- The frame KL of a rotation by `θ` in one pair is at most `½ F θ²`. -/
theorem pair_rotation_cost_le_half_fisher_sq {ι : Type*} (S : Finset ι) (a b : ι → ℝ) (θ : ℝ)
    (ha : ∀ j ∈ S, 0 < a j) (hb : ∀ j ∈ S, 0 < b j) :
    (1 / 2) * ∑ j ∈ S, Real.log ((Real.cos θ ^ 2 * a j + Real.sin θ ^ 2 * b j) *
        (Real.sin θ ^ 2 * a j + Real.cos θ ^ 2 * b j) / (a j * b j)) ≤
      (1 / 2) * (∑ j ∈ S, (a j - b j) ^ 2 / (a j * b j)) * θ ^ 2 := by
  have hF : 0 ≤ ∑ j ∈ S, (a j - b j) ^ 2 / (a j * b j) := by
    apply Finset.sum_nonneg
    intro j hj
    exact div_nonneg (sq_nonneg _) (mul_pos (ha j hj) (hb j hj)).le
  have h1 := pair_rotation_cost_sum_le S a b (Real.cos θ) (Real.sin θ) ha hb
    (by rw [add_comm]; exact Real.sin_sq_add_cos_sq θ)
  have h2 := sin_sq_mul_cos_sq_le θ
  have h3 : Real.cos θ ^ 2 * Real.sin θ ^ 2 * ∑ j ∈ S, (a j - b j) ^ 2 / (a j * b j) ≤
      (∑ j ∈ S, (a j - b j) ^ 2 / (a j * b j)) * θ ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left h2 hF]
  linarith

theorem gimbal_state_le_soap (m n : ℕ) :
    m ^ 2 + n ^ 2 + 4 * m * n ≤ 2 * m ^ 2 + 2 * n ^ 2 + 2 * m * n := by
  zify
  nlinarith [sq_nonneg ((m : ℤ) - n)]

end Gimbal
