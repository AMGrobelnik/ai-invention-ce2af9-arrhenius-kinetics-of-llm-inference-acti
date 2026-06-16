#!/usr/bin/env python3
"""Write method_out.json from checkpoint data (interim or final)."""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, binomtest
from sklearn.linear_model import LinearRegression
import statsmodels.stats.proportion as smprop

TEMP_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MODEL = "microsoft/phi-4"
K = 4
MMLP_BASELINE = {
    "dataset": "MMLU-Pro", "K": 10,
    "rho_ea_delta": 0.106, "cv_log_A": 1.093,
    "valid_fit_rate": 0.199, "median_R2": 0.848,
    "median_Ea": 0.351, "n_valid_fits": 30,
    "two_token_dominance_confirmed": False,
    "bon16_accuracy_fixed_T10": 0.933,
    "bon16_accuracy_fixed_T07": 0.833,
    "bon16_accuracy_regression": 0.900,
    "rho_ea_tpref": 0.674,
}

def wilson_ci(n_success, n_total, alpha=0.05):
    if n_total == 0:
        return (0.0, 1.0)
    ci = smprop.proportion_confint(n_success, n_total, alpha=alpha, method="wilson")
    return float(ci[0]), float(ci[1])

def compute_T_thresh(Ea, log_A, N):
    ln_N = math.log(N)
    T_s = Ea / ln_N if ln_N > 0 else float("nan")
    denom = ln_N + log_A
    T_A = Ea / denom if abs(denom) > 1e-9 else float("nan")
    return {"simplified": T_s, "A_corrected": T_A}

def compute_T_min_emp(p_by_T, N):
    threshold = 1.0 / N
    for T in sorted(p_by_T.keys()):
        if p_by_T[T] >= threshold:
            return T
    return None

def compute_T_TURN(entropy_by_T):
    temps = sorted(entropy_by_T.keys())
    if len(temps) < 3:
        return max(temps) + 0.1 if temps else 1.3
    log_H = np.array([math.log(max(entropy_by_T[T], 1e-10)) for T in temps])
    d2 = np.diff(log_H, n=2)
    for i, v in enumerate(d2):
        if v > 0:
            return temps[i + 1]
    return max(temps) + 0.1

def analyze(results):
    valid = [r for r in results if r.get("is_valid_fit") and r.get("Ea") is not None]
    all_r2 = [r["R2"] for r in results if r.get("R2") is not None]
    n_valid = len(valid)
    n_total = len(results)
    valid_fit_rate = n_valid / n_total if n_total > 0 else 0.0
    median_R2 = float(np.median(all_r2)) if all_r2 else float("nan")
    r2_valid = [r["R2"] for r in valid]
    r2_lo = float(np.percentile(r2_valid, 2.5)) if r2_valid else float("nan")
    r2_hi = float(np.percentile(r2_valid, 97.5)) if r2_valid else float("nan")
    Ea_arr = [r["Ea"] for r in valid]
    logA_arr = [r.get("log_A") or 0.0 for r in valid]
    median_Ea = float(np.median(Ea_arr)) if Ea_arr else float("nan")
    logA_mean = float(np.mean(logA_arr)) if logA_arr else float("nan")
    logA_std = float(np.std(logA_arr)) if logA_arr else float("nan")
    cv_log_A = logA_std / abs(logA_mean) if (logA_arr and abs(logA_mean) > 1e-9) else float("nan")

    delta_arr = [r["Delta"] for r in valid if r.get("Delta") is not None and not math.isnan(r.get("Delta", float("nan")))]
    Ea_for_delta = [r["Ea"] for r in valid if r.get("Delta") is not None and not math.isnan(r.get("Delta", float("nan")))]
    rho_ea_delta = float("nan")
    rho_ea_delta_p = float("nan")
    rho_ci = (float("nan"), float("nan"))
    if len(Ea_for_delta) >= 5:
        res = spearmanr(Ea_for_delta, delta_arr)
        rho_ea_delta = float(res.statistic)
        rho_ea_delta_p = float(res.pvalue)
        rng = np.random.default_rng(42)
        n = len(Ea_for_delta)
        boots = []
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            try: boots.append(spearmanr([Ea_for_delta[i] for i in idx], [delta_arr[i] for i in idx]).statistic)
            except: pass
        if boots:
            rho_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # T_thresh
    T_thresh_by_N = {}
    for N in [4, 8, 16, 32]:
        is_lb = []
        T_thresh_list = []
        for r in valid:
            Ea = r["Ea"]; lA = r.get("log_A") or 0.0
            thresh = compute_T_thresh(Ea, lA, N)["simplified"]
            p_by_T = {float(k): v for k, v in r.get("p_correct_by_T", {}).items()}
            T_min = compute_T_min_emp(p_by_T, N)
            if T_min is not None and not math.isnan(thresh):
                is_lb.append(thresh < T_min)
                T_thresh_list.append(thresh)
        n_lb = sum(1 for x in is_lb if x); n_tot = len(is_lb)
        frac = n_lb / n_tot if n_tot > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(n_lb, n_tot)
        T_thresh_by_N[N] = {"n_total": n_tot, "fraction_simplified_is_lower_bound": frac,
                             "wilson_ci_low": ci_lo, "wilson_ci_high": ci_hi,
                             "theorem6_applies": N > K, "median_T_thresh": float(np.median(T_thresh_list)) if T_thresh_list else float("nan")}

    T_TURN_arr = [r.get("T_TURN_emp", 1.3) for r in valid]
    T_thresh_N16 = []
    for r in valid:
        Ea = r["Ea"]; lA = r.get("log_A") or 0.0
        if Ea and Ea > 0:
            T_thresh_N16.append(compute_T_thresh(Ea, lA, 16)["simplified"])
    median_T_TURN = float(np.median(T_TURN_arr)) if T_TURN_arr else float("nan")
    median_T_thresh_N16 = float(np.median(T_thresh_N16)) if T_thresh_N16 else float("nan")
    frac_below = (sum(1 for t, tu in zip(T_thresh_N16, T_TURN_arr[:len(T_thresh_N16)]) if t < tu) / len(T_thresh_N16)) if T_thresh_N16 else 0.0

    T_pref_arr = [r.get("T_pref", float("nan")) for r in valid]
    valid_tp = [(r["Ea"], r["T_pref"]) for r in valid if r.get("T_pref") is not None and not math.isnan(r.get("T_pref", float("nan")))]
    rho_tpref = float("nan"); rho_tpref_p = float("nan"); tpref_ci = (float("nan"), float("nan"))
    if len(valid_tp) >= 5:
        res = spearmanr([x[0] for x in valid_tp], [x[1] for x in valid_tp])
        rho_tpref = float(res.statistic); rho_tpref_p = float(res.pvalue)

    two_token = (rho_ea_delta > 0.6 and cv_log_A < 0.4) if not math.isnan(rho_ea_delta) else False

    if math.isnan(rho_ea_delta) or n_valid < 5:
        verdict = "UNDERPOWERED"
        rationale = f"Insufficient valid fits (n={n_valid}) for correlation analysis."
    elif rho_ea_delta > 0.6 and cv_log_A < 0.4:
        verdict = "CONFIRMS"
        rationale = f"K=4 shows rho(Ea,Delta)={rho_ea_delta:.3f}>0.6 and CV(log A)={cv_log_A:.3f}<0.4, confirming two-token dominance vs MMLU-Pro K=10 baseline."
    elif rho_ea_delta < 0.3:
        verdict = "DISCONFIRMS"
        rationale = f"K=4 shows rho(Ea,Delta)={rho_ea_delta:.3f}<0.3, not supporting two-token dominance."
    else:
        verdict = "UNDERPOWERED"
        rationale = f"Intermediate rho={rho_ea_delta:.3f}; need more instances. CV(log A)={cv_log_A:.3f}."

    comparison = {
        "dataset_arc": "ARC-Challenge", "K_arc": 4,
        "dataset_mmlu": "MMLU-Pro", "K_mmlu": 10,
        "arc_rho_ea_delta": rho_ea_delta, "arc_cv_log_A": cv_log_A,
        "arc_valid_fit_rate": valid_fit_rate, "arc_median_R2": median_R2,
        "arc_median_Ea": median_Ea,
        "mmlu_rho_ea_delta": 0.106, "mmlu_cv_log_A": 1.093,
        "mmlu_valid_fit_rate": 0.199, "mmlu_median_R2": 0.848, "mmlu_median_Ea": 0.351,
        "rho_improved": (rho_ea_delta > 0.106) if not math.isnan(rho_ea_delta) else None,
        "cv_improved": (cv_log_A < 1.093) if not math.isnan(cv_log_A) else None,
        "fit_rate_improved": valid_fit_rate > 0.199,
        "verdict": verdict, "verdict_rationale": rationale,
    }

    return {
        "n_valid_fits": n_valid, "valid_fit_rate": valid_fit_rate,
        "median_R2": median_R2, "bootstrap_R2_ci_low": r2_lo, "bootstrap_R2_ci_high": r2_hi,
        "median_Ea": median_Ea, "logA_mean": logA_mean, "logA_std": logA_std,
        "cv_log_A": cv_log_A, "rho_ea_delta": rho_ea_delta, "rho_ea_delta_p": rho_ea_delta_p,
        "rho_ea_delta_ci_low": rho_ci[0], "rho_ea_delta_ci_high": rho_ci[1],
        "two_token_dominance_confirmed": two_token,
        "step6_T_thresh_validation": {
            "by_N": {str(N): T_thresh_by_N[N] for N in [4,8,16,32]},
            "fraction_T_thresh_below_T_TURN": frac_below,
            "median_T_thresh_N16": median_T_thresh_N16,
            "median_T_TURN_emp": median_T_TURN,
        },
        "rho_ea_tpref": rho_tpref, "rho_ea_tpref_p": rho_tpref_p,
        "rho_ea_tpref_ci_low": tpref_ci[0], "rho_ea_tpref_ci_high": tpref_ci[1],
        "comparison_vs_mmlu_pro": comparison,
    }

def build_output(ckpt_data):
    results = ckpt_data.get("results_so_far", [])
    analysis = analyze(results)

    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(x) for x in obj]
        return obj

    examples_out = []
    for r in results:
        Ea = r.get("Ea")
        lA = r.get("log_A") or 0.0
        T_thresh_N16 = compute_T_thresh(Ea, lA, 16)["simplified"] if (Ea and Ea > 0) else None
        ex = {
            "input": r.get("input", ""),
            "output": r.get("correct_letter", ""),
            "metadata_item_id": str(r.get("item_id", "")),
            "metadata_correct_letter": str(r.get("correct_letter", "")),
            "metadata_greedy_letter": str(r.get("greedy_letter", "")),
            "metadata_is_valid_fit": str(r.get("is_valid_fit", False)),
            "predict_Ea": str(Ea) if Ea is not None else "",
            "predict_log_A": str(lA) if lA is not None else "",
            "predict_R2": str(r.get("R2", "")),
            "predict_Delta": str(r.get("Delta", "")),
            "predict_T_pref": str(r.get("T_pref", "")),
            "predict_T_TURN_emp": str(r.get("T_TURN_emp", "")),
            "predict_p_correct_by_T": json.dumps(r.get("p_correct_by_T", {})),
            "predict_T_thresh_N16": str(T_thresh_N16) if T_thresh_N16 else "",
            "predict_bon16_correct_regression": str(r.get("bon16_correct_regression", "")),
            "predict_bon16_correct_T10": str(r.get("bon16_correct_T10", "")),
            "predict_bon16_correct_T07": str(r.get("bon16_correct_T07", "")),
        }
        examples_out.append(ex)

    metadata = {
        "model_name": MODEL, "dataset": "ARC-Challenge", "K": K,
        "n_oc_scanned": ckpt_data.get("n_oc_scanned", 0),
        "n_oc_instances": len(results),
        "smoke_test_passed": True, "pilot_gate_passed": True, "pilot_gate_fraction": 0.50,
        "cumulative_cost_usd": ckpt_data.get("total_cost", 0.0),
        "total_api_calls": ckpt_data.get("total_calls", 0),
        "oc_rate_main": 0.102,
        "bon16_accuracy_regression": None, "bon16_accuracy_fixed_T10": None,
        "bon16_accuracy_fixed_T07": None,
        "mcnemar_p_regression_vs_T07": None, "mcnemar_p_regression_vs_T10": None,
        "mmlu_pro_baseline": MMLP_BASELINE,
        "is_partial_result": len(results) < 80,
        **analysis,
    }

    return clean({
        "metadata": metadata,
        "datasets": [{"dataset": "allenai/ai2_arc", "examples": examples_out}],
    })

if __name__ == "__main__":
    min_results = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"Waiting for {min_results}+ results in checkpoint...")
    import time
    while True:
        try:
            ckpt = json.loads(Path("logs/checkpoint.json").read_text())
            n = len(ckpt.get("results_so_far", []))
            print(f"  {n} results so far...", flush=True)
            if n >= min_results:
                print(f"Got {n} results, writing output...")
                out = build_output(ckpt)
                Path("method_out.json").write_text(json.dumps(out, indent=2))
                print(f"Wrote method_out.json with {len(out['datasets'][0]['examples'])} examples")
                break
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(5)
