import Mathlib
import Formal.Flow

/-!
The variance estimates seen by the frame flow (theory/gimbal_theory.md, Lemma 5.4,
Proposition 5.5, Remark 5.6).

* `weighted_profile_variance`, `weighted_profile_variance_eq_iff`: with forgetting weights `w`,
  the weighted Gaussian log-likelihood of one variance is maximized exactly at the weighted average
  of the squared observations. So the flow's "tied" variance average is the profile-likelihood
  variance for the frame's own forgetting factor (Lemma 5.4.1).
* `excess_variance_identity`: for any weights `w`, the estimator variance `‖w‖² / ⟨w, v⟩²` equals
  the Cramér–Rao value `1 / ‖v‖²` plus `‖u‖² / ⟨w, v⟩²`, where `v` are the efficient weights and `u`
  is the component of `w` orthogonal to `v` (Lemma 5.4.2: the cost of noisy plug-in weights).
* `shrinkage_risk_ge`, `shrinkage_risk_eq_iff`: the risk `(c - 1)² S + c² ν` of shrinking a noisy
  residual (signal energy `S`, noise energy `ν`) by the factor `c` is minimized exactly at
  `c = S / (S + ν)`, with value `S ν / (S + ν)` (Proposition 5.5).
* `restart_closed_form`: after a restart at time `T` from the value `θT`, the bias-corrected
  schedule gives `θT` exactly the weight `1 - β^T` that the first `T` inputs would have had
  (Remark 5.6, the pooled warm start).
* `ema_weight_sq_sum`: the normalized weights of a bias-corrected exponential average over `T`
  inputs have squared sum `η = (1 - β)(1 + β^T) / ((1 + β)(1 - β^T))`, the factor by which the
  momentum averages the gradient noise (Proposition 5.7, the empirical-Bayes centering).
-/

open Finset

namespace Gimbal

variable {ι : Type*}

/-- The weighted log-likelihood sum collapses to the single-variance cost at the weighted mean. -/
theorem weighted_profile_sum (s : Finset ι) (w x : ι → ℝ) {D : ℝ}
    (hW : (∑ j ∈ s, w j) ≠ 0) :
    ∑ j ∈ s, w j * (x j / D + Real.log D) =
      (∑ j ∈ s, w j) * ((∑ j ∈ s, w j * x j) / (∑ j ∈ s, w j) / D + Real.log D) := by
  have h1 : ∑ j ∈ s, w j * (x j / D + Real.log D) =
      (∑ j ∈ s, w j * x j) / D + (∑ j ∈ s, w j) * Real.log D := by
    rw [Finset.sum_div, Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [h1, mul_add]
  congr 1
  field_simp

/-- **Lemma 5.4.1.** For positive total weight and a positive weighted mean `d`, every variance
`D > 0` has weighted negative log-likelihood at least that of `d`. -/
theorem weighted_profile_variance (s : Finset ι) (w x : ι → ℝ) {D : ℝ}
    (hW : 0 < ∑ j ∈ s, w j) (hd : 0 < (∑ j ∈ s, w j * x j) / ∑ j ∈ s, w j) (hD : 0 < D) :
    (∑ j ∈ s, w j) * (1 + Real.log ((∑ j ∈ s, w j * x j) / ∑ j ∈ s, w j)) ≤
      ∑ j ∈ s, w j * (x j / D + Real.log D) := by
  rw [weighted_profile_sum s w x hW.ne']
  exact mul_le_mul_of_nonneg_left (eigenvalue_cost_nonneg hd hD) hW.le

/-- **Lemma 5.4.1 (uniqueness).** The weighted mean is the only maximizer. -/
theorem weighted_profile_variance_eq_iff (s : Finset ι) (w x : ι → ℝ) {D : ℝ}
    (hW : 0 < ∑ j ∈ s, w j) (hd : 0 < (∑ j ∈ s, w j * x j) / ∑ j ∈ s, w j) (hD : 0 < D) :
    ∑ j ∈ s, w j * (x j / D + Real.log D) =
        (∑ j ∈ s, w j) * (1 + Real.log ((∑ j ∈ s, w j * x j) / ∑ j ∈ s, w j)) ↔
      D = (∑ j ∈ s, w j * x j) / ∑ j ∈ s, w j := by
  rw [weighted_profile_sum s w x hW.ne', mul_right_inj' hW.ne']
  exact eigenvalue_cost_eq_iff hd hD

/-- **Lemma 5.4.2 (excess variance of arbitrary weights).** With the inner product
`⟨u, u'⟩ = ∑ g u u'`, efficient weights `v` (`‖v‖² = F`, the Fisher information) and any weights
`w` with `⟨w, v⟩ ≠ 0`, the variance `‖w‖² / ⟨w, v⟩²` exceeds `1 / F` by exactly
`‖u‖² / ⟨w, v⟩²`, where `u = w - (⟨w, v⟩ / F) v` is orthogonal to `v`. -/
theorem excess_variance_identity (s : Finset ι) (g w v : ι → ℝ)
    (hF : 0 < ∑ j ∈ s, g j * v j ^ 2) (hP : ∑ j ∈ s, g j * w j * v j ≠ 0) :
    (∑ j ∈ s, g j * w j ^ 2) / (∑ j ∈ s, g j * w j * v j) ^ 2 =
      1 / (∑ j ∈ s, g j * v j ^ 2) +
        (∑ j ∈ s,
            g j * (w j - (∑ i ∈ s, g i * w i * v i) / (∑ i ∈ s, g i * v i ^ 2) * v j) ^ 2) /
          (∑ j ∈ s, g j * w j * v j) ^ 2 := by
  set F := ∑ j ∈ s, g j * v j ^ 2 with hFdef
  set P := ∑ j ∈ s, g j * w j * v j with hPdef
  set N := ∑ j ∈ s, g j * w j ^ 2 with hNdef
  have hexp :
      ∑ j ∈ s, g j * (w j - P / F * v j) ^ 2 = N - 2 * (P / F) * P + (P / F) ^ 2 * F := by
    have h : ∀ j ∈ s, g j * (w j - P / F * v j) ^ 2 =
        g j * w j ^ 2 - 2 * (P / F) * (g j * w j * v j) + (P / F) ^ 2 * (g j * v j ^ 2) := by
      intro j _
      ring
    rw [Finset.sum_congr rfl h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum]
  rw [hexp]
  have hF' : F ≠ 0 := hF.ne'
  field_simp
  ring

/-- The weights `u` of `excess_variance_identity` are orthogonal to `v`. -/
theorem excess_component_orthogonal (s : Finset ι) (g w v : ι → ℝ)
    (hF : 0 < ∑ j ∈ s, g j * v j ^ 2) :
    ∑ j ∈ s,
      g j * (w j - (∑ i ∈ s, g i * w i * v i) / (∑ i ∈ s, g i * v i ^ 2) * v j) * v j = 0 := by
  set F := ∑ j ∈ s, g j * v j ^ 2 with hFdef
  set P := ∑ j ∈ s, g j * w j * v j with hPdef
  have h : ∀ j ∈ s,
      g j * (w j - P / F * v j) * v j = g j * w j * v j - P / F * (g j * v j ^ 2) := by
    intro j _
    ring
  rw [Finset.sum_congr rfl h, Finset.sum_sub_distrib, ← Finset.mul_sum]
  field_simp
  ring

/-- **Proposition 5.5 (shrinkage risk).** Shrinking a residual with signal energy `S ≥ 0` observed
with independent noise of energy `ν ≥ 0` by `c` has risk `(c - 1)² S + c² ν ≥ S ν / (S + ν)`. -/
theorem shrinkage_risk_ge {S ν : ℝ} (hpos : 0 < S + ν) (c : ℝ) :
    S * ν / (S + ν) ≤ (c - 1) ^ 2 * S + c ^ 2 * ν := by
  rw [div_le_iff₀ hpos]
  have key : ((c - 1) ^ 2 * S + c ^ 2 * ν) * (S + ν) - S * ν = (c * (S + ν) - S) ^ 2 := by ring
  nlinarith [sq_nonneg (c * (S + ν) - S)]

/-- **Proposition 5.5 (optimal factor).** The risk bound is attained exactly at
`c = S / (S + ν)`. -/
theorem shrinkage_risk_eq_iff {S ν : ℝ} (hpos : 0 < S + ν) (c : ℝ) :
    (c - 1) ^ 2 * S + c ^ 2 * ν = S * ν / (S + ν) ↔ c = S / (S + ν) := by
  have key : ((c - 1) ^ 2 * S + c ^ 2 * ν) * (S + ν) - S * ν = (c * (S + ν) - S) ^ 2 := by ring
  rw [eq_div_iff hpos.ne', eq_div_iff hpos.ne']
  constructor
  · intro h
    have h2 : (c * (S + ν) - S) ^ 2 = 0 := by rw [← key, h]; ring
    have h3 : c * (S + ν) - S = 0 := pow_eq_zero_iff (two_ne_zero) |>.mp h2
    linarith
  · intro h
    have h3 : c * (S + ν) - S = 0 := by linarith
    have h2 : ((c - 1) ^ 2 * S + c ^ 2 * ν) * (S + ν) - S * ν = 0 := by rw [key, h3]; ring
    linarith

/-- The frame recursion with Gimbal's schedule restarted at time `T` from `θT`:
`r 0 = θT`, `r (k + 1) = r k + (1 - β) / (1 - β^(T + k + 1)) * (x (T + k) - r k)`. -/
noncomputable def restartSeq (β θT : ℝ) (T : ℕ) (x : ℕ → ℝ) : ℕ → ℝ
  | 0 => θT
  | k + 1 => restartSeq β θT T x k +
      (1 - β) / (1 - β ^ (T + k + 1)) * (x (T + k) - restartSeq β θT T x k)

/-- **Remark 5.6 (warm start).** After a restart at time `T`, the bias-corrected schedule yields
`(1 - β^(T+k)) r_k = β^k (1 - β^T) θT + ∑_{s<k} β^(k-1-s) (1 - β) x_(T+s)`: the restart value
carries exactly the total weight `1 - β^T` of the first `T` inputs it replaces. -/
theorem restart_closed_form (β θT : ℝ) (hβ0 : 0 ≤ β) (hβ1 : β < 1) (T : ℕ) (x : ℕ → ℝ)
    (k : ℕ) :
    (1 - β ^ (T + k)) * restartSeq β θT T x k =
      β ^ k * ((1 - β ^ T) * θT) +
        ∑ s ∈ Finset.range k, β ^ (k - 1 - s) * (1 - β) * x (T + s) := by
  induction k with
  | zero => simp [restartSeq]
  | succ k ih =>
    have hpos : 0 < 1 - β ^ (T + k + 1) := by
      have : β ^ (T + k + 1) < 1 := pow_lt_one₀ hβ0 hβ1 (by omega)
      linarith
    have step : (1 - β ^ (T + (k + 1))) * restartSeq β θT T x (k + 1) =
        β * ((1 - β ^ (T + k)) * restartSeq β θT T x k) + (1 - β) * x (T + k) := by
      rw [show T + (k + 1) = T + k + 1 by ring]
      simp only [restartSeq]
      have h1 := hpos.ne'
      field_simp
      ring
    have hnum : (∑ s ∈ Finset.range (k + 1), β ^ (k + 1 - 1 - s) * (1 - β) * x (T + s)) =
        β * (∑ s ∈ Finset.range k, β ^ (k - 1 - s) * (1 - β) * x (T + s)) +
          (1 - β) * x (T + k) := by
      rw [Finset.sum_range_succ, Finset.mul_sum]
      congr 1
      · refine Finset.sum_congr rfl fun s hs => ?_
        have hs' : s < k := Finset.mem_range.mp hs
        have : k + 1 - 1 - s = (k - 1 - s) + 1 := by omega
        rw [this, pow_succ]
        ring
      · simp
    rw [step, ih, hnum]
    ring

/-- **Proposition 5.7 (momentum noise).** The bias-corrected average of `T` inputs with forgetting
factor `β` has normalized weights `(1 - β) β^s / (1 - β^T)`, `s < T` (they sum to one). The sum of
their squares, the factor `η` by which the average scales the variance of independent inputs, is
`(1 - β)(1 + β^T) / ((1 + β)(1 - β^T))`: `1` for one input and `(1 - β) / (1 + β)` as `T → ∞`. -/
theorem ema_weight_sq_sum (β : ℝ) (hβ0 : 0 ≤ β) (hβ1 : β < 1) (T : ℕ) (hT : 0 < T) :
    ∑ s ∈ range T, ((1 - β) * β ^ s / (1 - β ^ T)) ^ 2 =
      (1 - β) * (1 + β ^ T) / ((1 + β) * (1 - β ^ T)) := by
  have hT1 : β ^ T < 1 := pow_lt_one₀ hβ0 hβ1 hT.ne'
  have hden : 1 - β ^ T ≠ 0 := by linarith
  have h1b : 1 + β ≠ 0 := by linarith
  have h1mb : 1 - β ≠ 0 := by linarith
  have hb2 : β ^ 2 ≠ 1 := by
    have : β ^ 2 < 1 := by nlinarith
    exact this.ne
  have hb2' : β ^ 2 - 1 ≠ 0 := sub_ne_zero.mpr hb2
  have hgeom : ∑ s ∈ range T, (β ^ 2) ^ s = ((β ^ 2) ^ T - 1) / (β ^ 2 - 1) :=
    geom_sum_eq hb2 T
  have hterm : ∀ s ∈ range T, ((1 - β) * β ^ s / (1 - β ^ T)) ^ 2 =
      (1 - β) ^ 2 / (1 - β ^ T) ^ 2 * (β ^ 2) ^ s := by
    intro s _
    rw [← pow_mul, mul_comm 2 s, pow_mul]
    field_simp
  rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum, hgeom]
  have hpow : (β ^ 2) ^ T = (β ^ T) ^ 2 := by rw [← pow_mul, ← pow_mul, mul_comm]
  rw [hpow]
  field_simp
  ring

end Gimbal
