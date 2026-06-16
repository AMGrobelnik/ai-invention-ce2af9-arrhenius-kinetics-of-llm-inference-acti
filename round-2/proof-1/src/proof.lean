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
