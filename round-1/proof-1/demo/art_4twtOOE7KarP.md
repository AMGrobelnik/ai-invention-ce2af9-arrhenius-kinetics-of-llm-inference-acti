# Lean 4 Formal Proof: Softmax Arrhenius Law and Approximation Error Bounds

> **[Run in Lean Playground](https://live.lean-lang.org/#codez=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)**

[![Open in Lean](https://img.shields.io/badge/Lean_4-Verify_Proof-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgMTloMjBMMTIgMnoiLz48L3N2Zz4=)](https://live.lean-lang.org/#codez=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)

---

## Summary

All five theorems from the artifact plan are formally verified in Lean 4 + Mathlib with verified=true and has_sorries=false.

**Theorems proved (all without sorry):**
1. `two_token_prob_eq`: The exact two-token softmax P = exp(lc/T)/(exp(lc/T)+exp(lw/T)) equals exp(-Δ/T)/(1+exp(-Δ/T)). Proved by cross-multiplying via div_eq_div_iff after the key factorization exp(lc/T) = exp(lw/T)·exp(-Δ/T), closing with ring.
2. `two_token_log_prob`: log P = -Δ/T - log(1+exp(-Δ/T)). Proved by rewriting with Theorem 1, applying Real.log_div and Real.log_exp.
3. `arrhenius_approx_bound`: -exp(-Δ/T) ≤ log P - (-Δ/T) ≤ 0. Lower bound uses the key lemma log(1+x) ≤ x; upper bound uses Real.log_nonneg (since 1+exp ≥ 1). This is the mathematical heart of the Arrhenius approximation.
4. `k_token_factorization`: For K tokens, P(correct|T) = A(T)·exp(-Ea/T) exactly, where A(T) = 1/(1+exp(-Ea/T)+other). Proved via the helper lemma sum_split_two_elems (using Finset.sum_erase_add twice to extract the w and c terms), then field_simp + ring.
5. `log_A_nonpositive`: log(A(T)) ≤ 0 since A(T) = 1/D with D ≥ 1. Proved by Real.log_nonpos with div_nonneg and div_le_one.

**Key helper lemmas proved:**
- `log_one_add_le`: log(1+x) ≤ x for x ≥ 0 (from Real.log_le_sub_one_of_pos)
- `sum_split_two_elems`: Extracts two named elements from a finite sum over Fin K using Finset.sum_erase_add + Finset.sum_filter

**Critical Lean 4 gotcha fixed:** In `arrhenius_approx_bound`, the let-chain conclusion starting with `-exp(...)` on the line after `let err :=` was parsed as binary subtraction continuing the let-expression. Fixed by wrapping conclusion in explicit parentheses.

**Proof files:** `proof.lean` (complete Lean 4 source), `proof_out.json` (schema-validated output with all lemma details).

## Lean Code

```lean
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

```

---
*Generated by AI Inventor Pipeline*
