# Lean 4 Proofs: T_thresh Window Non-Emptiness + Arrhenius Grid Approximation Bound

> **[Run in Lean Playground](https://live.lean-lang.org/#codez=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)**

[![Open in Lean](https://img.shields.io/badge/Lean_4-Verify_Proof-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgMTloMjBMMTIgMnoiLz48L3N2Zz4=)](https://live.lean-lang.org/#codez=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)

---

## Summary

All five theorems/lemmas verified by Lean 4 compiler (verified=true, has_sorries=false):

**Theorem A — thresh_window_nonempty**: For Ea, N, K : ℝ with Ea>0, K>1, K<N: proves Ea/log(N) < Ea/log(K). This formalizes that the [T_thresh, T_TURN] operating window is non-empty whenever the sample budget N exceeds the number of answer choices K. Proof: Real.log_pos (×2) + Real.log_lt_log + div_lt_div_iff₀ + mul_lt_mul_of_pos_left. Corollary: concrete MCQA instance (K=4, N=16, Ea=2) verified by direct application.

**Lemma — arrhenius_error_eq**: Proves log(1+exp(Δ/T)) − Δ/T = log(1+exp(−Δ/T)). Algebraic identity via intermediate h1 lemma (isolates only the subtracted Δ/T term to avoid pattern-match conflicts), Real.log_div, Real.exp_neg, field_simp+ring.

**Theorem B — arrhenius_error_bound**: For Δ,T>0: log(1+exp(−Δ/T)) ≤ exp(−Δ/T). One-step calc proof: Real.log_le_sub_one_of_pos (log x ≤ x−1) applied to x=1+exp(−Δ/T), then ring.

**Theorem — arrhenius_grid_max_error**: For all T in {0.05,0.1,0.2,0.3,0.5,0.7,1.0}, log(1+exp(−Δ/T)) ≤ exp(−Δ). Uses private grid_step helper (T≤1 → exp(−Δ/T)≤exp(−Δ) via neg_div rewrite + linarith from le_div_iff₀), dispatched by rcases on Set membership.

**Theorem — arrhenius_tmin_bound**: For Δ≥2: log(1+exp(−Δ/0.05)) ≤ exp(−40) < 4.5×10⁻¹⁸. Chains arrhenius_error_bound → ring (−Δ/0.05=−20Δ) → Real.exp_le_exp.mpr → linarith (−20Δ≤−40 from Δ≥2).

Key Mathlib lemmas used: Real.log_pos, Real.log_lt_log, div_lt_div_iff₀, mul_lt_mul_of_pos_left, Real.log_le_sub_one_of_pos, Real.exp_pos, Real.exp_le_exp, Real.log_div, Real.exp_neg, neg_div, le_div_iff₀. Compilation: 0 errors, 0 sorries, 4 benign unused-variable warnings.

## Lean Code

```lean
import Mathlib

-- ============================================================
-- Theorem A: Window Non-Emptiness
-- The operating window [T_thresh, T_TURN] is non-empty when N > K
-- T_thresh = Ea/ln(N), T_TURN = Ea/ln(K)
-- ============================================================

theorem thresh_window_nonempty (Ea N K : ℝ)
    (hEa : 0 < Ea)
    (hK  : 1 < K)
    (hN  : K < N)
    : Ea / Real.log N < Ea / Real.log K := by
  have hlogK : 0 < Real.log K := Real.log_pos hK
  have hN1 : 1 < N := by linarith
  have hlogN : 0 < Real.log N := Real.log_pos hN1
  have hloglt : Real.log K < Real.log N :=
    Real.log_lt_log (by linarith) hN
  rw [div_lt_div_iff₀ hlogN hlogK]
  exact mul_lt_mul_of_pos_left hloglt hEa

-- Corollary: concrete numeric example for MCQA (K=4, N=16, Ea=2)
example : (2 : ℝ) / Real.log 16 < (2 : ℝ) / Real.log 4 :=
  thresh_window_nonempty 2 16 4 (by norm_num) (by norm_num) (by norm_num)

-- ============================================================
-- Helper Lemma: Arrhenius error identity
-- log(1 + exp(Δ/T)) - Δ/T = log(1 + exp(-Δ/T))
-- ============================================================

lemma arrhenius_error_eq (Δ T : ℝ) (hΔ : 0 < Δ) (hT : 0 < T) :
    Real.log (1 + Real.exp (Δ / T)) - Δ / T =
    Real.log (1 + Real.exp (-Δ / T)) := by
  have hexpD : 0 < Real.exp (Δ / T) := Real.exp_pos _
  have h1expD : 0 < 1 + Real.exp (Δ / T) := by linarith
  -- Step 1: replace Δ/T with log(exp(Δ/T)) only in the subtracted term
  have h1 : Real.log (1 + Real.exp (Δ / T)) - Δ / T =
            Real.log (1 + Real.exp (Δ / T)) - Real.log (Real.exp (Δ / T)) := by
    rw [Real.log_exp]
  -- Step 2: combine the two logs
  rw [h1, ← Real.log_div h1expD.ne' hexpD.ne']
  -- Step 3: simplify (1 + exp(x)) / exp(x) = 1 + exp(-x)
  congr 1
  rw [show -Δ / T = -(Δ / T) by ring, Real.exp_neg]
  field_simp
  ring

-- ============================================================
-- Theorem B: Arrhenius Grid Approximation Bound
-- error(T) = log(1 + exp(-Δ/T)) ≤ exp(-Δ/T)
-- ============================================================

theorem arrhenius_error_bound (Δ T : ℝ) (hΔ : 0 < Δ) (hT : 0 < T) :
    Real.log (1 + Real.exp (-Δ / T)) ≤ Real.exp (-Δ / T) := by
  have hexp_pos : 0 < Real.exp (-Δ / T) := Real.exp_pos _
  have h1z_pos : 0 < 1 + Real.exp (-Δ / T) := by linarith
  calc Real.log (1 + Real.exp (-Δ / T))
      ≤ (1 + Real.exp (-Δ / T)) - 1 := Real.log_le_sub_one_of_pos h1z_pos
    _ = Real.exp (-Δ / T)           := by ring

-- Private helper: single-point grid bound (T ≤ 1 → error ≤ exp(-Δ))
private lemma grid_step (Δ T : ℝ) (hΔ : 0 < Δ) (hTpos : 0 < T) (hTle1 : T ≤ 1) :
    Real.log (1 + Real.exp (-Δ / T)) ≤ Real.exp (-Δ) := by
  apply le_trans (arrhenius_error_bound Δ T hΔ hTpos)
  apply Real.exp_le_exp.mpr
  -- Need: -Δ/T ≤ -Δ, equivalently Δ ≤ Δ/T (since Δ > 0 and T ≤ 1)
  have hDivGe : Δ ≤ Δ / T := by
    rw [le_div_iff₀ hTpos]
    nlinarith
  rw [neg_div]
  linarith

-- Grid theorem: maximum error over {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} ≤ exp(-Δ)
theorem arrhenius_grid_max_error (Δ : ℝ) (hΔ : 0 < Δ) :
    ∀ T ∈ ({0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} : Set ℝ),
    Real.log (1 + Real.exp (-Δ / T)) ≤ Real.exp (-Δ) := by
  intro T hT
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hT
  rcases hT with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact grid_step Δ _ hΔ (by norm_num) (by norm_num)

-- Near-machine-epsilon bound at T_min = 0.05 for Δ ≥ 2
-- Error ≤ exp(-40) < 4.5×10^{-18} (machine-epsilon accurate)
theorem arrhenius_tmin_bound (Δ : ℝ) (hΔ : 2 ≤ Δ) :
    Real.log (1 + Real.exp (-Δ / 0.05)) ≤ Real.exp (-40 : ℝ) := by
  have hΔpos : 0 < Δ := by linarith
  have step1 : Real.log (1 + Real.exp (-Δ / 0.05)) ≤ Real.exp (-Δ / 0.05) :=
    arrhenius_error_bound Δ 0.05 hΔpos (by norm_num)
  have step2 : Real.exp (-Δ / 0.05) = Real.exp (-20 * Δ) := by
    congr 1; ring
  have step3 : Real.exp (-20 * Δ) ≤ Real.exp (-40 : ℝ) := by
    apply Real.exp_le_exp.mpr
    linarith
  linarith [step2 ▸ step3]

```

---
*Generated by AI Inventor Pipeline*
