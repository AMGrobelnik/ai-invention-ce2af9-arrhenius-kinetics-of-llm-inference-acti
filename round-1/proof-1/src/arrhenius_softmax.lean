import Mathlib

set_option maxHeartbeats 800000

open Real Finset

namespace ArrheniusLLM

/-!
  Formal verification that the softmax probability of the correct token follows
  an Arrhenius-type law, with tight additive error bounds on log P ≈ -Ea/T.

  Five theorems are proved:
  1. two_token_prob_eq: exact two-token softmax formula P = exp(-Δ/T)/(1+exp(-Δ/T))
  2. two_token_log_prob: log form log P = -Δ/T - log(1+exp(-Δ/T))
  3. arrhenius_approx_bound: error bound -exp(-Δ/T) ≤ err ≤ 0
  4. k_token_factorization: K-token P = A(T)·exp(-Ea/T)
  5. log_A_nonpositive: log(A(T)) ≤ 0
-/

-- ============================================================
-- KEY LEMMA: log(1 + x) ≤ x for x ≥ 0
-- Proof: apply log_le_sub_one_of_pos at (1+x), gives log(1+x) ≤ x
-- ============================================================

lemma log_one_add_le (x : ℝ) (hx : 0 ≤ x) : Real.log (1 + x) ≤ x := by
  have h1 : 0 < 1 + x := by linarith
  linarith [Real.log_le_sub_one_of_pos h1]

-- ============================================================
-- THEOREM 1: Two-token exact softmax formula
-- P(correct|T) = exp(-Δ/T) / (1 + exp(-Δ/T))  where Δ = lw - lc
-- ============================================================

theorem two_token_prob_eq (lc lw T : ℝ) (hT : 0 < T) :
    let Δ := lw - lc
    exp (lc / T) / (exp (lc / T) + exp (lw / T)) =
    exp (-(Δ / T)) / (1 + exp (-(Δ / T))) := by
  simp only
  have hD : 0 < exp (-((lw - lc) / T)) := exp_pos _
  have hd : 0 < exp (lc / T) + exp (lw / T) := by
    linarith [exp_pos (lc / T), exp_pos (lw / T)]
  have hone : 0 < 1 + exp (-((lw - lc) / T)) := by linarith
  -- Key factorization: exp(lc/T) = exp(lw/T) · exp(-Δ/T)
  have key : exp (lc / T) = exp (lw / T) * exp (-((lw - lc) / T)) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [div_eq_div_iff hd.ne' hone.ne', key]; ring

-- ============================================================
-- THEOREM 2: Log-form of two-token probability
-- log P = -Δ/T - log(1 + exp(-Δ/T))
-- ============================================================

theorem two_token_log_prob (lc lw T : ℝ) (hT : 0 < T) :
    let Δ := lw - lc
    Real.log (exp (lc / T) / (exp (lc / T) + exp (lw / T))) =
    -(Δ / T) - Real.log (1 + exp (-(Δ / T))) := by
  simp only
  have hD_pos : 0 < exp (-((lw - lc) / T)) := exp_pos _
  have hone_pos : 0 < 1 + exp (-((lw - lc) / T)) := by linarith
  rw [two_token_prob_eq lc lw T hT, Real.log_div hD_pos.ne' hone_pos.ne', Real.log_exp]

-- ============================================================
-- THEOREM 3: Arrhenius approximation error bound
-- -exp(-Δ/T) ≤ log P - (-Δ/T) ≤ 0
-- Note: conclusion wrapped in parens to prevent '-exp' being
--       parsed as continuation of the 'let err :=' expression
-- ============================================================

theorem arrhenius_approx_bound (lc lw T : ℝ) (hT : 0 < T) (hlcw : lc < lw) :
    let Δ := lw - lc
    let p := exp (lc / T) / (exp (lc / T) + exp (lw / T))
    let err := Real.log p - (-(Δ / T))
    ((-exp (-(Δ / T)) ≤ err) ∧ (err ≤ 0)) := by
  simp only
  have hD_pos : 0 < exp (-((lw - lc) / T)) := exp_pos _
  rw [two_token_log_prob lc lw T hT]
  constructor
  · -- Lower bound: log(1+exp(-Δ/T)) ≤ exp(-Δ/T)  [Key Lemma]
    linarith [log_one_add_le (exp (-((lw - lc) / T))) hD_pos.le]
  · -- Upper bound: log(1+exp(-Δ/T)) ≥ 0  [since 1+exp ≥ 1]
    linarith [Real.log_nonneg (by linarith : (1 : ℝ) ≤ 1 + exp (-((lw - lc) / T)))]

-- ============================================================
-- HELPER LEMMA: Extract two elements from a finite sum
-- ∑ f k = f w + f c + ∑ (if k≠w∧k≠c then f k else 0)
-- ============================================================

lemma sum_split_two_elems {K : ℕ} (f : Fin K → ℝ) (w c : Fin K) (hwc : w ≠ c) :
    ∑ k : Fin K, f k =
    f w + f c + ∑ k : Fin K, if k ≠ w ∧ k ≠ c then f k else 0 := by
  have hw_mem : w ∈ (Finset.univ : Finset (Fin K)) := Finset.mem_univ w
  have hc_erased : c ∈ (Finset.univ : Finset (Fin K)).erase w :=
    Finset.mem_erase.mpr ⟨hwc.symm, Finset.mem_univ c⟩
  -- Extract w from the full sum
  have step1 : ∑ k : Fin K, f k =
      f w + ∑ k ∈ (Finset.univ : Finset (Fin K)).erase w, f k := by
    rw [add_comm]; exact (Finset.sum_erase_add _ f hw_mem).symm
  -- Extract c from the remaining sum
  have step2 : ∑ k ∈ (Finset.univ : Finset (Fin K)).erase w, f k =
      f c + ∑ k ∈ ((Finset.univ : Finset (Fin K)).erase w).erase c, f k := by
    rw [add_comm]; exact (Finset.sum_erase_add _ f hc_erased).symm
  -- Remaining sum = indicator sum over all Fin K
  have step3 : ∑ k ∈ ((Finset.univ : Finset (Fin K)).erase w).erase c, f k =
      ∑ k : Fin K, if k ≠ w ∧ k ≠ c then f k else 0 := by
    have set_eq : ((Finset.univ : Finset (Fin K)).erase w).erase c =
        (Finset.univ : Finset (Fin K)).filter (fun k => k ≠ w ∧ k ≠ c) := by
      ext k; simp [Finset.mem_erase, Finset.mem_filter, Finset.mem_univ, and_comm]
    rw [set_eq, Finset.sum_filter]
  rw [step1, step2, step3]; ring

-- ============================================================
-- THEOREM 4: K-token exact factorization P = A(T) · exp(-Ea/T)
-- where A(T) = 1 / (1 + exp(-Ea/T) + Σ_{k≠w,c} exp((lk-lw)/T))
-- ============================================================

theorem k_token_factorization {K : ℕ} (hK : 2 ≤ K)
    (logits : Fin K → ℝ) (c w : Fin K) (hcw : c ≠ w)
    (T : ℝ) (hT : 0 < T) (hlcw : logits c < logits w) :
    let Ea := logits w - logits c
    let other := ∑ k : Fin K,
      if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0
    let A_T := 1 / (1 + exp (-(Ea / T)) + other)
    let denom := ∑ k : Fin K, exp (logits k / T)
    exp (logits c / T) / denom = A_T * exp (-(Ea / T)) := by
  simp only
  have hw_pos : 0 < exp (logits w / T) := exp_pos _
  have hw_ne : exp (logits w / T) ≠ 0 := hw_pos.ne'
  have hEa_pos : 0 < exp (-((logits w - logits c) / T)) := exp_pos _
  have hother_nn : 0 ≤ ∑ k : Fin K,
      if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0 := by
    apply Finset.sum_nonneg; intros k _
    split_ifs with h
    · exact (exp_pos _).le
    · exact le_refl 0
  -- Factor exp(logits w / T) from denominator
  have denom_eq : ∑ k : Fin K, exp (logits k / T) =
      exp (logits w / T) * ∑ k : Fin K, exp ((logits k - logits w) / T) := by
    rw [Finset.mul_sum]; congr 1; ext k
    rw [← Real.exp_add]; congr 1; ring
  -- Numerator factorization
  have num_eq : exp (logits c / T) =
      exp (logits w / T) * exp (-((logits w - logits c) / T)) := by
    rw [← Real.exp_add]; congr 1; ring
  -- Split normalized sum: w-term=1, c-term=exp(-Ea/T), rest=other
  have split_eq : ∑ k : Fin K, exp ((logits k - logits w) / T) =
      1 + exp (-((logits w - logits c) / T)) +
      ∑ k : Fin K, if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0 := by
    rw [sum_split_two_elems (fun k => exp ((logits k - logits w) / T)) w c hcw.symm]
    have hw_term : exp ((logits w - logits w) / T) = 1 := by simp
    have hc_term : exp ((logits c - logits w) / T) = exp (-((logits w - logits c) / T)) := by
      congr 1; ring
    rw [hw_term, hc_term]
  have hdenom_pos : 0 < 1 + exp (-((logits w - logits c) / T)) +
      ∑ k : Fin K, if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0 := by
    linarith
  rw [denom_eq, split_eq, num_eq]
  have h_denom_ne : exp (logits w / T) *
      (1 + exp (-((logits w - logits c) / T)) +
       ∑ k : Fin K, if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0) ≠ 0 :=
    mul_ne_zero hw_ne hdenom_pos.ne'
  field_simp [h_denom_ne]; ring

-- ============================================================
-- THEOREM 5: log(A(T)) ≤ 0
-- Since D = 1/A(T) ≥ 1, we have A(T) ≤ 1, so log(A(T)) ≤ 0
-- ============================================================

lemma log_A_nonpositive {K : ℕ}
    (logits : Fin K → ℝ) (c w : Fin K) (T : ℝ) (hT : 0 < T) :
    let Ea := logits w - logits c
    let other := ∑ k : Fin K,
      if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0
    let A_T := 1 / (1 + exp (-(Ea / T)) + other)
    Real.log A_T ≤ 0 := by
  simp only
  have hexp : 0 < exp (-((logits w - logits c) / T)) := exp_pos _
  have hother : 0 ≤ ∑ k : Fin K,
      if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0 := by
    apply Finset.sum_nonneg; intros k _
    split_ifs with h
    · exact (exp_pos _).le
    · exact le_refl 0
  have hdenom_pos : 0 < 1 + exp (-((logits w - logits c) / T)) +
      ∑ k : Fin K, if k ≠ w ∧ k ≠ c then exp ((logits k - logits w) / T) else 0 := by
    linarith
  apply Real.log_nonpos
  · exact div_nonneg zero_le_one (by linarith)
  · rw [div_le_one hdenom_pos]; linarith

end ArrheniusLLM
