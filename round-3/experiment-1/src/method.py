#!/usr/bin/env python3
"""
Arrhenius Kinetics of LLM Inference — Iter-3 Experiment.
Protocol: 9-step Arrhenius temperature-selection on Qwen-2.5-7B-instruct.
Primary new result: two-token dominance test (rho_ea_delta > 0.6, cv_log_A < 0.4).
"""

import argparse
import asyncio
import gc
import json
import math
import os
import resource
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ── Logging setup ─────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_PATH = (WORKSPACE / "../../../iter_1/gen_art/gen_art_dataset_1/full_data_out.json").resolve()
OUTPUT_PATH = WORKSPACE / "method_out.json"
CHECKPOINT_PATH = WORKSPACE / "checkpoint.json"

TEMP_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SAMPLES_PER_TEMP = 50
N_BON = 16
MAX_BUDGET_USD = 8.0
MAX_CONCURRENT = 10
MAX_RETRIES = 5

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

CANDIDATES = [
    # Primary: GPT-4o-mini confirmed logprob support via Azure/OpenAI on OpenRouter
    "openai/gpt-4o-mini",
    # Fallbacks (may not support logprobs depending on routing)
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.03),
    "qwen/qwen-2.5-7b-instruct:fireworks": (0.04, 0.10),
    "meta-llama/llama-3.1-8b-instruct:fireworks": (0.02, 0.03),
}
DEFAULT_PRICING = (0.15, 0.60)

# Global mutable state (single async event loop, no lock needed)
_total_cost_usd: float = 0.0
_selected_model: str = ""


# ── Logprob helpers ───────────────────────────────────────────────────────────

def get_letter_score(top_logprobs: list, letter: str) -> float:
    """Max logprob for letter, handling 'A' and ' A' token forms."""
    target = letter.strip().upper()
    best = -999.0
    for item in top_logprobs:
        tok = item.get("token", "").strip().upper()
        if tok == target:
            lp = float(item.get("logprob", -999.0))
            if lp > best:
                best = lp
    return best


def argmax_letter(top_logprobs: list, num_choices: int) -> str:
    """Return most likely letter from top_logprobs."""
    letters = "ABCDEFGHIJ"[:num_choices]
    return max(letters, key=lambda l: get_letter_score(top_logprobs, l))


def letter_entropy_from_samples(samples: list, num_choices: int) -> float:
    """Shannon entropy over letter distribution from a list of sampled letters."""
    letters = "ABCDEFGHIJ"[:num_choices]
    total = len(samples)
    if total == 0:
        return 0.0
    h = 0.0
    for l in letters:
        p = samples.count(l) / total
        if p > 1e-10:
            h -= p * math.log(p)
    return h


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 95% CI for proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ── Async API client ─────────────────────────────────────────────────────────

async def call_api_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 1,
    top_logprobs: int = 0,
) -> Optional[dict]:
    """Async API call to OpenRouter with exponential backoff retry."""
    global _total_cost_usd

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://arrhenius-llm-kinetics.example.com",
        "X-Title": "Arrhenius-Kinetics-Iter3",
    }
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": max(float(temperature), 0.0),
        "max_tokens": max_tokens,
    }
    if top_logprobs > 0:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status == 429:
                        wait = min(2 ** (attempt + 1), 60)
                        logger.warning(f"Rate limit, waiting {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 500:
                        wait = 2 ** attempt
                        logger.warning(f"Server error {resp.status}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(f"API error {resp.status}: {body[:300]}")
                        return None

                    data = await resp.json()

                    # Track cost: prefer actual cost from response, fallback to pricing estimate
                    api_cost = data.get("usage", {}).get("cost")
                    if api_cost is not None:
                        _total_cost_usd += float(api_cost)
                    else:
                        usage = data.get("usage", {})
                        inp_tok = usage.get("prompt_tokens", 150)
                        out_tok = usage.get("completion_tokens", 1)
                        price_in, price_out = PRICING.get(model, DEFAULT_PRICING)
                        _total_cost_usd += (inp_tok * price_in + out_tok * price_out) / 1_000_000

                    return data

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = 2 ** attempt
            logger.warning(f"Connection error (attempt {attempt+1}): {type(e).__name__}: {e}. Retrying in {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return None

    logger.error(f"All {MAX_RETRIES} retries exhausted for model={model}")
    return None


def extract_letter(data: Optional[dict]) -> Optional[str]:
    """Extract first A-J letter from response content."""
    if data is None:
        return None
    try:
        content = data["choices"][0]["message"]["content"] or ""
        for ch in content.strip():
            if ch.upper() in "ABCDEFGHIJ":
                return ch.upper()
    except (KeyError, IndexError, TypeError):
        pass
    return None


def extract_logprobs(data: Optional[dict]) -> Optional[list]:
    """Extract top_logprobs list [{token, logprob}, ...] from response."""
    if data is None:
        return None
    try:
        lp = data["choices"][0].get("logprobs")
        if lp is None:
            return None
        content = lp.get("content") or []
        if not content:
            return None
        return content[0].get("top_logprobs") or []
    except (KeyError, IndexError, TypeError):
        return None


# ── Step 1: Model selection ───────────────────────────────────────────────────

async def smoke_test_model(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    examples: list,
) -> Optional[dict]:
    """Smoke test: logprob access + OC rate on 30 examples."""
    logger.info(f"Smoke test: {model}")
    tasks = [
        call_api_async(session, semaphore, model, ex["input"], 0.0, 1, 20)
        for ex in examples[:30]
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    oc_count = 0
    logprob_ok = False
    valid_count = 0

    for ex, resp in zip(examples[:30], responses):
        if isinstance(resp, Exception) or resp is None:
            continue
        lps = extract_logprobs(resp)
        if lps is None:
            continue
        logprob_ok = True
        # Use actual model output (not argmax of logprobs) for correct OC detection
        pred = extract_letter(resp)
        if pred is None:
            continue
        correct = ex["output"]
        correct_lp = get_letter_score(lps, correct)
        if pred != correct and correct_lp > -15:
            oc_count += 1
        valid_count += 1

    if not logprob_ok:
        logger.warning(f"{model}: logprobs NOT available")
        return None

    oc_rate = oc_count / max(valid_count, 1)
    logger.info(f"{model}: logprobs_ok=True, oc_rate={oc_rate:.3f} ({oc_count}/{valid_count})")
    return {"model": model, "logprobs_ok": True, "oc_count": oc_count, "oc_rate": oc_rate, "valid_count": valid_count}


async def select_model(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    pilot_examples: list,
) -> tuple[str, list]:
    """Select model via smoke test; return (selected_model, model_selection_log)."""
    model_selection_log = []

    for candidate in CANDIDATES:
        result = await smoke_test_model(session, semaphore, candidate, pilot_examples)
        if result is None:
            model_selection_log.append({"model": candidate, "logprobs_ok": False, "oc_rate": 0.0})
        else:
            model_selection_log.append(result)
            if result["logprobs_ok"] and result["oc_rate"] >= 0.15:
                logger.info(f"Selected model: {candidate} (oc_rate={result['oc_rate']:.3f})")
                return candidate, model_selection_log

    # Fallback: first model with logprobs, ignoring OC threshold
    for res in model_selection_log:
        if res.get("logprobs_ok"):
            logger.warning(f"No candidate met OC threshold 0.15; falling back to {res['model']}")
            return res["model"], model_selection_log

    # Last resort: try with :fireworks provider suffix
    for candidate in CANDIDATES[:2]:
        fw_candidate = candidate + ":fireworks"
        result = await smoke_test_model(session, semaphore, fw_candidate, pilot_examples)
        if result and result["logprobs_ok"]:
            model_selection_log.append(result)
            logger.info(f"Selected fireworks model: {fw_candidate}")
            return fw_candidate, model_selection_log

    raise RuntimeError("No model with logprob support found")


# ── Step 2: Pilot gate ────────────────────────────────────────────────────────

async def run_pilot_gate(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    pilot_examples: list,
) -> dict:
    """Pilot gate: find OC instances, check rising limb frac >= 0.30."""
    logger.info(f"=== STEP 2: Pilot Gate on {len(pilot_examples)} examples ===")

    # Greedy logprob calls on all pilot examples
    tasks = [call_api_async(session, semaphore, model, ex["input"], 0.0, 1, 20) for ex in pilot_examples]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    oc_instances = []
    for ex, resp in zip(pilot_examples, responses):
        if isinstance(resp, Exception) or resp is None:
            continue
        lps = extract_logprobs(resp)
        if lps is None:
            continue
        num_choices = ex.get("metadata_num_choices", 10)
        pred = extract_letter(resp)  # use actual output, not argmax of T=1 logprobs
        if pred is None:
            continue
        correct = ex["output"]
        correct_lp = get_letter_score(lps, correct)
        wrong_letter = pred  # actual prediction = wrong letter

        if pred != correct and correct_lp > -15:
            oc_instances.append({"ex": ex, "pred": pred, "correct": correct, "correct_lp": correct_lp,
                                  "wrong_letter": wrong_letter})

    pilot_oc_rate = len(oc_instances) / max(len(pilot_examples), 1)
    logger.info(f"Pilot OC: {len(oc_instances)}/{len(pilot_examples)} = {pilot_oc_rate:.3f}")

    # Gate: use 10 samples per temp to save budget, on first 50 OC instances
    gate_oc = oc_instances[:50]
    n_pilot_samples = 10
    rising_count = 0

    for i, oc in enumerate(gate_oc):
        if _total_cost_usd >= MAX_BUDGET_USD * 0.3:
            logger.warning("Budget at 30% limit, stopping pilot gate early")
            break
        ex = oc["ex"]
        tasks_t01 = [call_api_async(session, semaphore, model, ex["input"], 0.1, 1) for _ in range(n_pilot_samples)]
        tasks_t05 = [call_api_async(session, semaphore, model, ex["input"], 0.5, 1) for _ in range(n_pilot_samples)]
        all_resps = await asyncio.gather(*tasks_t01, *tasks_t05, return_exceptions=True)
        resps_01 = all_resps[:n_pilot_samples]
        resps_05 = all_resps[n_pilot_samples:]
        correct = oc["correct"]
        p_01 = sum(1 for r in resps_01 if not isinstance(r, Exception) and extract_letter(r) == correct) / n_pilot_samples
        p_05 = sum(1 for r in resps_05 if not isinstance(r, Exception) and extract_letter(r) == correct) / n_pilot_samples
        if p_05 > p_01 + 0.05:
            rising_count += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Pilot gate: {i+1}/{len(gate_oc)} done, cost=${_total_cost_usd:.3f}")

    n_gated = min(len(gate_oc), i + 1 if gate_oc else 0)
    pilot_rising_frac = rising_count / max(n_gated, 1)
    pilot_gate_passed = pilot_rising_frac >= 0.30

    logger.info(f"Pilot gate: rising_frac={pilot_rising_frac:.3f}, passed={pilot_gate_passed}")
    return {
        "pilot_gate_passed": pilot_gate_passed,
        "pilot_oc_rate": pilot_oc_rate,
        "pilot_rising_frac": pilot_rising_frac,
        "n_pilot_oc": len(oc_instances),
        "n_gated": n_gated,
    }


# ── Step 3: Build OC set ──────────────────────────────────────────────────────

async def build_oc_set(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    examples: list,
    n_target: int = 200,
) -> list:
    """Greedy pass on examples; collect up to n_target OC instances."""
    logger.info(f"Building OC set from {len(examples)} examples (target={n_target})")

    # Run greedy in batches to stay memory-efficient
    batch_size = 50
    oc_instances = []

    for batch_start in range(0, len(examples), batch_size):
        if len(oc_instances) >= n_target:
            break
        if _total_cost_usd >= MAX_BUDGET_USD * 0.5:
            logger.warning("Budget at 50%, stopping OC collection")
            break
        batch = examples[batch_start: batch_start + batch_size]
        tasks = [call_api_async(session, semaphore, model, ex["input"], 0.0, 1, 20) for ex in batch]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for ex, resp in zip(batch, responses):
            if len(oc_instances) >= n_target:
                break
            if isinstance(resp, Exception) or resp is None:
                continue
            lps = extract_logprobs(resp)
            if lps is None:
                continue
            num_choices = ex.get("metadata_num_choices", 10)
            pred = extract_letter(resp)  # actual model output at T=0, not argmax of T=1 logprobs
            if pred is None:
                continue
            correct = ex["output"]
            correct_lp = get_letter_score(lps, correct)
            # wrong_letter = actual model prediction (guaranteed in logprobs since model output it)
            wrong_letter = pred
            wrong_lp = get_letter_score(lps, wrong_letter)

            # OC: model predicted wrong, AND correct letter has non-negligible logprob,
            # AND wrong letter is actually found in logprobs with higher score
            if pred != correct and correct_lp > -15 and wrong_lp > -100 and wrong_lp > correct_lp:
                # Count competing tokens (logprob within 1 nit of correct)
                n_competing = sum(
                    1 for item in lps
                    if item.get("token", "").strip().upper() in "ABCDEFGHIJ"
                    and float(item.get("logprob", -999)) > correct_lp - 1.0
                )
                oc_instances.append({
                    "question_id": ex.get("metadata_question_id"),
                    "subject": ex.get("metadata_subject", "unknown"),
                    "split": ex.get("metadata_split", "main_set"),
                    "input": ex["input"],
                    "output": ex["output"],
                    "correct_letter": correct,
                    "wrong_letter": wrong_letter,
                    "correct_lp": float(correct_lp),
                    "wrong_lp": float(wrong_lp),
                    "Delta": float(wrong_lp - correct_lp),
                    "n_competing": n_competing,
                    "metadata_num_choices": num_choices,
                    "metadata_src": ex.get("metadata_src", ""),
                })

        logger.info(f"  OC set: {len(oc_instances)} found, scanned {batch_start + len(batch)}/{len(examples)}")
        gc.collect()

    logger.info(f"Final OC set size: {len(oc_instances)}")
    return oc_instances


# ── Step 4: Temperature sweep ─────────────────────────────────────────────────

async def sweep_instance(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    oc: dict,
    temperatures: list,
    n_samples: int,
) -> dict:
    """Sweep one OC instance across all temperatures. Returns {T_str: {p_hat, ci, k, n, samples}}."""
    prompt = oc["input"]
    correct = oc["correct_letter"]

    # Launch all (temperature, sample) pairs concurrently
    all_tasks = []
    all_meta = []
    for T in temperatures:
        for _ in range(n_samples):
            all_tasks.append(call_api_async(session, semaphore, model, prompt, T, 1))
            all_meta.append(T)

    responses = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Group by temperature
    temp_letters: dict[float, list] = {T: [] for T in temperatures}
    for resp, T in zip(responses, all_meta):
        if not isinstance(resp, Exception) and resp is not None:
            letter = extract_letter(resp)
            if letter:
                temp_letters[T].append(letter)

    result = {}
    for T in temperatures:
        letters = temp_letters[T]
        k = letters.count(correct)
        n = len(letters)
        p_hat = k / max(n, 1)
        ci = wilson_ci(k, n)
        result[str(T)] = {"p_hat": float(p_hat), "ci": list(ci), "k": k, "n": n, "samples": letters}

    return result


async def run_temperature_sweep(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    oc_instances: list,
    temperatures: list,
    n_samples: int,
    checkpoint: dict,
) -> dict:
    """Sweep all OC instances; checkpoint every 10."""
    logger.info(f"=== STEP 4: Temperature Sweep ({len(oc_instances)} instances × {len(temperatures)} temps × {n_samples} samples) ===")

    sweep_results: dict = checkpoint.get("sweep_results", {})

    for i, oc in enumerate(oc_instances):
        if _total_cost_usd >= MAX_BUDGET_USD:
            logger.warning(f"Budget limit ${MAX_BUDGET_USD} reached at instance {i}. Stopping sweep.")
            break

        qid = str(oc["question_id"])
        if qid in sweep_results:
            logger.debug(f"  Instance {i} qid={qid}: loaded from checkpoint")
            continue

        result = await sweep_instance(session, semaphore, model, oc, temperatures, n_samples)
        sweep_results[qid] = result

        if (i + 1) % 5 == 0 or i == len(oc_instances) - 1:
            logger.info(f"  Sweep {i+1}/{len(oc_instances)}, cost=${_total_cost_usd:.4f}")

        if (i + 1) % 10 == 0:
            checkpoint["sweep_results"] = sweep_results
            checkpoint["total_cost_usd"] = _total_cost_usd
            CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))
            logger.info(f"  Checkpoint saved at instance {i+1}")

        gc.collect()

    # Final save
    checkpoint["sweep_results"] = sweep_results
    checkpoint["total_cost_usd"] = _total_cost_usd
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))
    logger.info(f"Sweep complete: {len(sweep_results)} instances in checkpoint")
    return sweep_results


# ── Step 5: Arrhenius fitting ─────────────────────────────────────────────────

def fit_arrhenius(temps: list, p_hats: list) -> dict:
    """
    Fit Arrhenius model log P = -Ea/T + log_A and three alternatives.
    Returns fit stats dict.
    """
    temps_arr = np.array(temps, dtype=float)
    p_arr = np.array(p_hats, dtype=float)

    valid = (p_arr > 0) & (temps_arr > 0)
    if valid.sum() < 3:
        return {"valid_fit": False, "reason": f"only {valid.sum()} valid temps (need ≥3)"}

    t_v = temps_arr[valid]
    p_v = p_arr[valid]
    log_p = np.log(p_v)

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    def _lsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return c

    # Arrhenius: log P = [-1/T, 1] · [Ea, log_A]
    X_arr = np.column_stack([1.0 / t_v, np.ones(len(t_v))])
    c_arr = _lsq(X_arr, log_p)
    Ea = float(-c_arr[0])
    log_A = float(c_arr[1])
    R2_arr = _r2(log_p, X_arr @ c_arr)

    # Exponential: log P = a*T + b
    X_exp = np.column_stack([t_v, np.ones(len(t_v))])
    R2_exp = _r2(log_p, X_exp @ _lsq(X_exp, log_p))

    # Power: log P = a*log(T) + b
    log_t = np.log(t_v)
    X_pow = np.column_stack([log_t, np.ones(len(t_v))])
    R2_pow = _r2(log_p, X_pow @ _lsq(X_pow, log_p))

    # Linear: P = a*T + b (in original scale)
    X_lin = np.column_stack([t_v, np.ones(len(t_v))])
    R2_lin = _r2(p_v, X_lin @ _lsq(X_lin, p_v))

    R2_advantage = float(R2_arr - max(R2_exp, R2_pow, R2_lin))

    # Bootstrap CI on R2_arr
    n = len(t_v)
    rng = np.random.default_rng(42)
    r2_boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        Xb = X_arr[idx]
        yb = log_p[idx]
        cb = _lsq(Xb, yb)
        r2_boots.append(_r2(yb, Xb @ cb))
    ci_r2 = [float(np.percentile(r2_boots, 2.5)), float(np.percentile(r2_boots, 97.5))]

    return {
        "valid_fit": True,
        "n_valid_temps": int(valid.sum()),
        "Ea": Ea,
        "log_A": log_A,
        "A_hat": float(np.exp(log_A)),
        "R2_arr": float(R2_arr),
        "R2_exp": float(R2_exp),
        "R2_pow": float(R2_pow),
        "R2_lin": float(R2_lin),
        "R2_advantage": R2_advantage,
        "CI_R2": ci_r2,
    }


# ── Step 6: Two-token dominance ───────────────────────────────────────────────

def two_token_dominance_test(fit_results: list) -> dict:
    """Compute rho(Ea, Delta) and cv(log_A) for two-token dominance test."""
    valid = [r for r in fit_results if r.get("valid_fit")]
    if len(valid) < 5:
        return {
            "two_token_dominance_confirmed": False,
            "reason": f"only {len(valid)} valid fits",
            "n_valid": len(valid),
        }

    Ea_list = [r["Ea"] for r in valid]
    Delta_list = [r["Delta"] for r in valid]
    log_A_list = [r["log_A"] for r in valid]
    n_comp_list = [r.get("n_competing", 0) for r in valid]

    rho_ea_delta, p_ea_delta = stats.spearmanr(Ea_list, Delta_list)
    mean_log_A = float(np.mean(log_A_list))
    std_log_A = float(np.std(log_A_list))
    cv_log_A = std_log_A / abs(mean_log_A) if abs(mean_log_A) > 1e-6 else float("inf")

    # Competing-token regression: does n_competing explain Ea - Delta deviation?
    deviation = np.array(Ea_list) - np.array(Delta_list)
    competing_rho, competing_p = stats.spearmanr(deviation, n_comp_list)

    confirmed = bool(rho_ea_delta >= 0.6 and cv_log_A < 0.4)

    return {
        "n_valid": len(valid),
        "rho_ea_delta": float(rho_ea_delta),
        "p_ea_delta": float(p_ea_delta),
        "mean_log_A": mean_log_A,
        "std_log_A": std_log_A,
        "cv_log_A": float(cv_log_A),
        "competing_rho": float(competing_rho),
        "competing_p": float(competing_p),
        "two_token_dominance_confirmed": confirmed,
    }


# ── Step 7: T_thresh + TURN ───────────────────────────────────────────────────

def compute_thresh_triplet(Ea: float, Delta: float, A_hat: float, N: int) -> dict:
    """Compute T_thresh for simple, approx (Delta), and A-corrected strategies."""
    log_N = math.log(max(N, 2))
    simple = Ea / log_N if log_N > 1e-6 else float("inf")
    approx = Delta / log_N if log_N > 1e-6 else float("inf")
    inner = N * A_hat
    A_corr = Ea / math.log(inner) if inner > 1.0 else float("inf")
    return {"simple": float(simple), "approx": float(approx), "A_corrected": float(A_corr)}


def t_emp_min(sweep: dict, N: int) -> Optional[float]:
    """Min temperature where N * P_hat >= 1 (i.e. expected ≥1 correct in N draws)."""
    for T_str in sorted(sweep, key=lambda x: float(x)):
        p_hat = sweep[T_str]["p_hat"]
        if N * p_hat >= 1.0:
            return float(T_str)
    return None


def validate_thresh_bounds(fit_results: list, sweep_results: dict, N_values: list) -> dict:
    """Compute fraction lower-bound valid for each N and each threshold strategy."""
    out = {}
    for N in N_values:
        lb_simple, lb_approx, lb_A, track_A = [], [], [], []
        for r in fit_results:
            if not r.get("valid_fit"):
                continue
            qid = str(r["question_id"])
            sweep = sweep_results.get(qid, {})
            T_min = t_emp_min(sweep, N)
            if T_min is None:
                continue
            thresh = compute_thresh_triplet(r["Ea"], r["Delta"], r.get("A_hat", 1.0), N)
            lb_simple.append(thresh["simple"] < T_min)
            lb_approx.append(thresh["approx"] < T_min)
            lb_A.append(thresh["A_corrected"] < T_min)
            if T_min > 0:
                track_A.append(abs(thresh["A_corrected"] - T_min) / T_min < 0.20)
        out[str(N)] = {
            "simple": float(np.mean(lb_simple)) if lb_simple else None,
            "approx": float(np.mean(lb_approx)) if lb_approx else None,
            "A_corrected": float(np.mean(lb_A)) if lb_A else None,
            "tracking_A": float(np.mean(track_A)) if track_A else None,
            "n_instances": len(lb_simple),
        }
    return out


def compute_turn(fit_results: list, sweep_results: dict) -> dict:
    """
    Compute TURN inflection temperature from entropy-vs-T on sampled data.
    Uses TEMP_GRID sampling data to estimate per-instance entropy H(T),
    then finds first inflection in log H_mean(T).
    """
    grid = TEMP_GRID
    H_by_T = {}

    for T in grid:
        T_str = str(T)
        h_vals = []
        for r in fit_results:
            qid = str(r["question_id"])
            sweep = sweep_results.get(qid, {})
            sw_T = sweep.get(T_str, {})
            samples = sw_T.get("samples", [])
            if samples:
                num_choices = r.get("metadata_num_choices", 10)
                h_vals.append(letter_entropy_from_samples(samples, num_choices))
        H_by_T[T_str] = float(np.mean(h_vals)) if h_vals else 0.0

    # Find log-entropy inflection
    grid_H = [(T, H_by_T[str(T)]) for T in grid if H_by_T.get(str(T), 0) > 1e-6]
    T_TURN = 0.7  # default

    if len(grid_H) >= 3:
        log_H = [math.log(H) for _, H in grid_H]
        for j in range(1, len(log_H) - 1):
            d2 = log_H[j + 1] - 2 * log_H[j] + log_H[j - 1]
            if d2 > 0:
                T_TURN = float(grid_H[j][0]) + 0.1  # beta = 0.1 for best-of-N
                break

    return {
        "T_TURN": float(T_TURN),
        "H_by_T": {k: v for k, v in H_by_T.items()},
        "n_instances_used": len([r for r in fit_results if str(r["question_id"]) in sweep_results]),
        "method": "entropy_inflection_from_sampling",
    }


# ── Step 8: Best-of-N accuracy ────────────────────────────────────────────────

def bootstrap_bon_accuracy(
    sweep: dict,
    correct_letter: str,
    T_op: float,
    N: int = 16,
    n_boot: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Bootstrap Best-of-N success rate using nearest-grid samples."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Find nearest grid temperature
    available = [float(T_str) for T_str in sweep]
    if not available:
        return {"accuracy": None, "majority_vote": None}
    nearest = min(available, key=lambda t: abs(t - T_op))
    samples = sweep[str(nearest)].get("samples", [])

    if not samples:
        return {"accuracy": None, "majority_vote": None, "nearest_T": nearest}

    # Bootstrap: each trial draws N samples, success = at least 1 correct
    successes = sum(
        1 for _ in range(n_boot)
        if correct_letter in rng.choice(samples, size=min(N, len(samples)), replace=True)
    )
    acc = successes / n_boot
    ci = wilson_ci(successes, n_boot)

    # Majority vote letter
    letter_counts: dict[str, int] = {}
    for l in samples:
        letter_counts[l] = letter_counts.get(l, 0) + 1
    majority_vote = max(letter_counts, key=letter_counts.get) if letter_counts else correct_letter

    return {
        "accuracy": float(acc),
        "ci": list(ci),
        "nearest_T": float(nearest),
        "n_samples_available": len(samples),
        "majority_vote": str(majority_vote),
    }


def compute_accuracy_table(
    fit_results: list,
    sweep_results: dict,
    T_TURN: float,
    N: int = N_BON,
) -> dict:
    """Compute Best-of-N accuracy for 5 strategies via bootstrap."""
    strategy_accs: dict[str, list] = {
        "T_op_regression": [],
        "T_op_approx_delta": [],
        "fixed_T07": [],
        "fixed_T10": [],
        "TURN": [],
    }
    rng = np.random.default_rng(42)
    log_N = math.log(N)

    for r in fit_results:
        if not r.get("valid_fit"):
            continue
        qid = str(r["question_id"])
        sweep = sweep_results.get(qid, {})
        correct = r["correct_letter"]
        Ea = r["Ea"]
        Delta = r["Delta"]

        T_reg = Ea / log_N + 0.3
        T_approx = Delta / log_N + 0.3

        for strat, T_op in [
            ("T_op_regression", T_reg),
            ("T_op_approx_delta", T_approx),
            ("fixed_T07", 0.7),
            ("fixed_T10", 1.0),
            ("TURN", T_TURN + 0.1),
        ]:
            res = bootstrap_bon_accuracy(sweep, correct, T_op, N, 500, rng)
            if res.get("accuracy") is not None:
                strategy_accs[strat].append(res["accuracy"])

    table = {}
    api_calls = {
        "T_op_regression": 351,    # 1 logprob + 350 sweep
        "T_op_approx_delta": 1,    # 1 logprob only
        "fixed_T07": 16,
        "fixed_T10": 16,
        "TURN": 30,
    }
    for name, accs in strategy_accs.items():
        if accs:
            mean_acc = float(np.mean(accs))
            n = len(accs)
            k = round(mean_acc * n)
            ci = wilson_ci(k, n)
            table[name] = {
                "accuracy": mean_acc,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "n_instances": n,
                "api_calls_per_instance": api_calls[name],
            }
    return table


# ── Step 9: Ea predicts T_pref ────────────────────────────────────────────────

def ea_vs_tpref(fit_results: list) -> dict:
    """Spearman rho(Ea, T_pref) and partial rho controlling for subject."""
    valid = [r for r in fit_results if r.get("valid_fit") and r.get("T_pref") is not None]
    if len(valid) < 5:
        return {"rho_ea_tpref": None, "p_ea_tpref": None, "n_valid": len(valid)}

    Ea_list = [r["Ea"] for r in valid]
    T_pref_list = [r["T_pref"] for r in valid]
    subjects = [r.get("subject", "unknown") for r in valid]

    rho, p = stats.spearmanr(Ea_list, T_pref_list)

    # Partial Spearman controlling for subject via residualization
    partial_rho, p_partial = None, None
    try:
        from statsmodels.formula.api import ols  # type: ignore
        df = pd.DataFrame({"Ea": Ea_list, "T_pref": T_pref_list, "subject": subjects})
        Ea_resid = np.array(df["Ea"]) - ols("Ea ~ C(subject)", df).fit().fittedvalues.values
        Tp_resid = np.array(df["T_pref"]) - ols("T_pref ~ C(subject)", df).fit().fittedvalues.values
        pr, pp = stats.spearmanr(Ea_resid, Tp_resid)
        partial_rho, p_partial = float(pr), float(pp)
    except Exception as e:
        logger.warning(f"Partial Spearman failed (statsmodels): {e}")

    return {
        "rho_ea_tpref": float(rho),
        "p_ea_tpref": float(p),
        "partial_rho_ea_tpref": partial_rho,
        "p_partial": p_partial,
        "R2_ea_tpref": float(rho ** 2),
        "n_valid": len(valid),
    }


# ── Step 10: Catalysis test ───────────────────────────────────────────────────

def make_cot_prompt(prompt: str) -> str:
    return prompt + "\n\nThink step by step before answering."


def make_4shot_prompt(prompt: str, shot_examples: list) -> str:
    shots = ""
    for ex in shot_examples[:4]:
        shots += ex["input"] + f"\nThe answer is ({ex['output']}).\n\n"
    return shots + prompt


async def run_catalysis_test(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    cat_oc: list,
    examples_by_subject: dict,
) -> dict:
    """Catalysis test: zero-shot vs CoT vs 4-shot on catalysis OC instances."""
    logger.info(f"=== STEP 10: Catalysis Test ({len(cat_oc)} OC instances) ===")
    results = []

    for oc in cat_oc:
        qid = oc["question_id"]
        subject = oc["subject"]
        correct = oc["correct_letter"]
        wrong = oc["wrong_letter"]
        prompt_zs = oc["input"]
        prompt_cot = make_cot_prompt(prompt_zs)

        # 4-shot: find examples from same subject, excluding this instance
        shot_pool = [
            ex for ex in examples_by_subject.get(subject, [])
            if ex.get("metadata_question_id") != qid
        ]
        prompt_4shot = make_4shot_prompt(prompt_zs, shot_pool) if len(shot_pool) >= 4 else prompt_cot

        tasks = [
            call_api_async(session, semaphore, model, prompt_zs, 1.0, 1, 20),
            call_api_async(session, semaphore, model, prompt_cot, 1.0, 1, 20),
            call_api_async(session, semaphore, model, prompt_4shot, 1.0, 1, 20),
        ]
        resps = await asyncio.gather(*tasks, return_exceptions=True)

        cond_data = {}
        for cond, resp in zip(["zero_shot", "cot", "four_shot"], resps):
            if isinstance(resp, Exception) or resp is None:
                cond_data[cond] = None
                continue
            lps = extract_logprobs(resp)
            if lps is None:
                cond_data[cond] = None
                continue
            c_lp = get_letter_score(lps, correct)
            w_lp = get_letter_score(lps, wrong)
            cond_data[cond] = {
                "correct_lp": float(c_lp),
                "wrong_lp": float(w_lp),
                "Delta": float(w_lp - c_lp),
                "Ea_approx": float(w_lp - c_lp),
            }

        zs = cond_data.get("zero_shot")
        cot = cond_data.get("cot")
        if zs and cot:
            cot_reduces_ea = cot["Ea_approx"] < zs["Ea_approx"]
            delta_correct = cot["correct_lp"] - zs["correct_lp"]
            delta_wrong = cot["wrong_lp"] - zs["wrong_lp"]
            dominant = "catalysis" if delta_correct > abs(delta_wrong) else "other"
            results.append({
                "question_id": qid, "subject": subject,
                "Ea_zs": zs["Ea_approx"], "Ea_cot": cot["Ea_approx"],
                "cot_reduces_ea": bool(cot_reduces_ea),
                "delta_correct_logit": float(delta_correct),
                "delta_wrong_logit": float(delta_wrong),
                "dominant_mechanism": dominant,
                "conditions": cond_data,
            })

    if not results:
        return {"n_instances": 0, "catalysis_fraction": None, "paired_t_pvalue": None, "catalysis_ci": None}

    n_reduces = sum(1 for r in results if r["cot_reduces_ea"])
    n_catalysis = sum(1 for r in results if r["cot_reduces_ea"] and r["dominant_mechanism"] == "catalysis")
    cat_frac = n_catalysis / max(n_reduces, 1)
    cat_ci = list(wilson_ci(n_catalysis, max(n_reduces, 1)))

    Ea_zs = [r["Ea_zs"] for r in results]
    Ea_cot = [r["Ea_cot"] for r in results]
    t_pval = None
    if len(Ea_zs) > 1:
        _, t_pval = stats.ttest_rel(Ea_zs, Ea_cot)

    return {
        "n_instances": len(results),
        "n_cot_reduces_ea": n_reduces,
        "catalysis_fraction": float(cat_frac),
        "catalysis_ci": cat_ci,
        "paired_t_pvalue": float(t_pval) if t_pval is not None else None,
    }


# ── Main orchestration ────────────────────────────────────────────────────────

@logger.catch(reraise=True)
async def run_experiment(args: argparse.Namespace) -> dict:
    global _total_cost_usd, _selected_model

    logger.info("=" * 60)
    logger.info("Arrhenius Kinetics Iter-3: Weak Model Experiment")
    logger.info(f"  max_instances={args.max_instances}, n_samples={args.n_samples}, mini={args.mini}")
    logger.info("=" * 60)

    # RAM limit
    try:
        resource.setrlimit(resource.RLIMIT_AS, (20 * 1024 ** 3, 20 * 1024 ** 3))
    except Exception:
        pass

    # Load checkpoint
    checkpoint: dict = {}
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = json.loads(CHECKPOINT_PATH.read_text())
            _total_cost_usd = float(checkpoint.get("total_cost_usd", 0.0))
            logger.info(f"Checkpoint loaded; cost so far=${_total_cost_usd:.4f}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e}")
            checkpoint = {}

    # Load dataset
    logger.info(f"Loading dataset from {DATASET_PATH}")
    raw = json.loads(DATASET_PATH.read_text())
    all_examples: list = raw["datasets"][0]["examples"]
    logger.info(f"Dataset: {len(all_examples)} examples total")

    pilot_set = [e for e in all_examples if e.get("metadata_split") == "pilot_set"]
    main_set = [e for e in all_examples if e.get("metadata_split") == "main_set"]
    catalysis_set = [e for e in all_examples if e.get("metadata_split") == "catalysis_set"]
    logger.info(f"Splits: pilot={len(pilot_set)}, main={len(main_set)}, catalysis={len(catalysis_set)}")

    if args.mini:
        # Mini mode: enough examples to produce OC instances (GPT-4o-mini ~70% accurate on MMLU-Pro)
        pilot_set = pilot_set[:10]
        main_set = main_set[:50]
        catalysis_set = catalysis_set[:10]
        all_examples = pilot_set + main_set + catalysis_set
        logger.info(f"MINI MODE: pilot={len(pilot_set)}, main={len(main_set)}, catalysis={len(catalysis_set)}")

    # Index examples by subject for 4-shot sampling
    examples_by_subject: dict = {}
    for ex in all_examples:
        s = ex.get("metadata_subject", "unknown")
        examples_by_subject.setdefault(s, []).append(ex)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 3)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:

        # ── Step 1: Model selection ──────────────────────────────────────────
        if checkpoint.get("selected_model"):
            _selected_model = checkpoint["selected_model"]
            model_selection_log = checkpoint.get("model_selection_log", [])
            logger.info(f"Model from checkpoint: {_selected_model}")
        else:
            _selected_model, model_selection_log = await select_model(session, semaphore, pilot_set[:30])
            checkpoint["selected_model"] = _selected_model
            checkpoint["model_selection_log"] = model_selection_log
            checkpoint["total_cost_usd"] = _total_cost_usd
            CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

        model = _selected_model
        logger.info(f"Using model: {model}")

        # ── Step 2: Pilot gate ───────────────────────────────────────────────
        if checkpoint.get("pilot_gate_result"):
            pilot_gate_result = checkpoint["pilot_gate_result"]
            logger.info(f"Pilot gate from checkpoint: passed={pilot_gate_result.get('pilot_gate_passed')}")
        else:
            pilot_gate_result = await run_pilot_gate(session, semaphore, model, pilot_set)
            checkpoint["pilot_gate_result"] = pilot_gate_result
            checkpoint["total_cost_usd"] = _total_cost_usd
            CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

        if not pilot_gate_result.get("pilot_gate_passed"):
            logger.warning("Pilot gate FAILED (rising_frac < 0.30). Proceeding with experiment anyway.")

        # ── Step 3: Build main OC set ────────────────────────────────────────
        if checkpoint.get("oc_instances") is not None:
            oc_instances = checkpoint["oc_instances"]
            cat_oc = checkpoint.get("cat_oc", [])
            logger.info(f"OC instances from checkpoint: main={len(oc_instances)}, cat={len(cat_oc)}")
        else:
            oc_instances = await build_oc_set(session, semaphore, model, main_set, args.max_instances)
            cat_oc = await build_oc_set(session, semaphore, model, catalysis_set, 50)
            checkpoint["oc_instances"] = oc_instances
            checkpoint["cat_oc"] = cat_oc
            checkpoint["total_cost_usd"] = _total_cost_usd
            CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

        logger.info(f"OC set sizes: main={len(oc_instances)}, catalysis={len(cat_oc)}")
        if not oc_instances:
            raise RuntimeError("No OC instances found — cannot proceed with experiment")

        # ── Step 4: Temperature sweep ────────────────────────────────────────
        sweep_results = await run_temperature_sweep(
            session, semaphore, model, oc_instances, TEMP_GRID, args.n_samples, checkpoint
        )

        # ── Step 5: Arrhenius fitting ────────────────────────────────────────
        logger.info("=== STEP 5: Arrhenius Fitting ===")
        fit_results = []
        for oc in oc_instances:
            qid = str(oc["question_id"])
            sweep = sweep_results.get(qid, {})

            temps = sorted([float(T_str) for T_str in sweep])
            p_hats = [sweep[str(T)]["p_hat"] for T in temps]

            fit = fit_arrhenius(temps, p_hats)

            # T_pref = temperature with highest P_hat
            T_pref = float(temps[int(np.argmax(p_hats))]) if temps and p_hats else None

            fit_results.append({**oc, **fit, "T_pref": T_pref})

            if fit["valid_fit"]:
                logger.debug(
                    f"  qid={qid}: Ea={fit['Ea']:.3f}, log_A={fit['log_A']:.3f}, "
                    f"R2={fit['R2_arr']:.3f}, T_pref={T_pref}"
                )

        n_valid = sum(1 for r in fit_results if r.get("valid_fit"))
        logger.info(f"Valid Arrhenius fits: {n_valid}/{len(fit_results)}")

        valid_fits = [r for r in fit_results if r.get("valid_fit")]
        median_R2 = float(np.median([r["R2_arr"] for r in valid_fits])) if valid_fits else 0.0
        frac_R2_085 = sum(1 for r in valid_fits if r["R2_arr"] >= 0.85) / max(n_valid, 1)
        median_R2_adv = float(np.median([r["R2_advantage"] for r in valid_fits])) if valid_fits else 0.0
        logger.info(f"Median R2={median_R2:.3f}, frac R2>0.85={frac_R2_085:.3f}, advantage={median_R2_adv:.3f}")

        # ── Step 6: Two-token dominance ──────────────────────────────────────
        logger.info("=== STEP 6: Two-Token Dominance Test ===")
        dominance = two_token_dominance_test(fit_results)
        logger.info(
            f"  confirmed={dominance['two_token_dominance_confirmed']}, "
            f"rho_ea_delta={dominance.get('rho_ea_delta', 'N/A')}, "
            f"cv_log_A={dominance.get('cv_log_A', 'N/A')}"
        )

        # ── Step 7: T_thresh validation + TURN ──────────────────────────────
        logger.info("=== STEP 7: T_thresh Validation + TURN ===")
        thresh_by_N = validate_thresh_bounds(fit_results, sweep_results, [4, 8, 16, 32])
        turn_result = compute_turn(fit_results, sweep_results)
        T_TURN = turn_result["T_TURN"]
        logger.info(f"T_TURN={T_TURN}")

        log_N16 = math.log(N_BON)
        thresh_lt_TURN = [
            r["Ea"] / log_N16 < T_TURN
            for r in valid_fits if r.get("Ea") is not None
        ]
        frac_thresh_lt_TURN = float(np.mean(thresh_lt_TURN)) if thresh_lt_TURN else None

        window_nontrivial = [
            r for r in valid_fits
            if r.get("Ea") is not None and 0 < r["Ea"] / log_N16 < max(TEMP_GRID)
        ]
        window_nontrivial_frac = len(window_nontrivial) / max(n_valid, 1)

        # ── Step 8: Best-of-N accuracy ───────────────────────────────────────
        logger.info("=== STEP 8: Best-of-N Accuracy Comparison ===")
        accuracy_table = compute_accuracy_table(fit_results, sweep_results, T_TURN, N_BON)
        for name, res in accuracy_table.items():
            logger.info(f"  {name}: {res['accuracy']:.3f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")

        # ── Step 9: Ea predicts T_pref ───────────────────────────────────────
        logger.info("=== STEP 9: Ea → T_pref ===")
        tpref_result = ea_vs_tpref(fit_results)
        logger.info(
            f"  rho={tpref_result.get('rho_ea_tpref')}, p={tpref_result.get('p_ea_tpref')}, "
            f"partial_rho={tpref_result.get('partial_rho_ea_tpref')}"
        )

        # ── Step 10: Catalysis test ──────────────────────────────────────────
        if cat_oc and _total_cost_usd < MAX_BUDGET_USD - 1.0:
            catalysis_result = await run_catalysis_test(
                session, semaphore, model, cat_oc[:20], examples_by_subject
            )
        else:
            reason = "no catalysis OC" if not cat_oc else f"budget ${_total_cost_usd:.2f} near limit"
            logger.warning(f"Skipping catalysis test ({reason})")
            catalysis_result = {"n_instances": 0, "catalysis_fraction": None,
                                "paired_t_pvalue": None, "catalysis_ci": None, "n_cot_reduces_ea": 0}

        checkpoint["total_cost_usd"] = _total_cost_usd
        CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))

    # ── Step 11: Write method_out.json ───────────────────────────────────────
    logger.info("=== STEP 11: Writing method_out.json ===")
    rng_out = np.random.default_rng(0)

    output_examples = []
    for r in fit_results:
        qid = str(r["question_id"])
        sweep = sweep_results.get(qid, {})
        ex_meta: dict = {}

        # Per-temperature P_hat and CI
        for T in TEMP_GRID:
            T_str = str(T)
            key = "T" + T_str.replace(".", "")
            sw_T = sweep.get(T_str, {})
            ex_meta[f"metadata_P_hat_{key}"] = sw_T.get("p_hat")
            ci = sw_T.get("ci")
            if ci:
                ex_meta[f"metadata_CI_lo_{key}"] = ci[0]
                ex_meta[f"metadata_CI_hi_{key}"] = ci[1]

        # Arrhenius fit fields
        ex_meta.update({
            "metadata_question_id": r.get("question_id"),
            "metadata_subject": r.get("subject"),
            "metadata_split": r.get("split"),
            "metadata_Delta": r.get("Delta"),
            "metadata_n_competing": r.get("n_competing"),
            "metadata_valid_fit": r.get("valid_fit", False),
            "metadata_Ea": r.get("Ea") if r.get("valid_fit") else None,
            "metadata_log_A": r.get("log_A") if r.get("valid_fit") else None,
            "metadata_A_hat": r.get("A_hat") if r.get("valid_fit") else None,
            "metadata_R2_arr": r.get("R2_arr") if r.get("valid_fit") else None,
            "metadata_R2_linear": r.get("R2_lin") if r.get("valid_fit") else None,
            "metadata_R2_exp": r.get("R2_exp") if r.get("valid_fit") else None,
            "metadata_R2_pow": r.get("R2_pow") if r.get("valid_fit") else None,
            "metadata_R2_advantage": r.get("R2_advantage") if r.get("valid_fit") else None,
            "metadata_T_pref": r.get("T_pref"),
        })

        # T_thresh for N=16
        if r.get("valid_fit"):
            thresh16 = compute_thresh_triplet(r["Ea"], r["Delta"], r.get("A_hat", 1.0), N_BON)
            ex_meta["metadata_T_thresh_simple_N16"] = thresh16["simple"]
            ex_meta["metadata_T_thresh_approx_N16"] = thresh16["approx"]
            ex_meta["metadata_T_thresh_A_N16"] = thresh16["A_corrected"]
            ex_meta["metadata_T_emp_min_N16"] = t_emp_min(sweep, N_BON)

        # Predict fields (best-of-N majority vote per strategy)
        predict_fields: dict = {}
        if r.get("valid_fit"):
            Ea = r["Ea"]
            Delta = r["Delta"]
            T_reg = Ea / log_N16 + 0.3
            T_approx = Delta / log_N16 + 0.3
            for strat_name, T_op in [
                ("T_op_regression", T_reg),
                ("T_op_approx_delta", T_approx),
                ("fixed_T07", 0.7),
                ("fixed_T10", 1.0),
                ("TURN", T_TURN + 0.1),
            ]:
                bon = bootstrap_bon_accuracy(sweep, r["correct_letter"], T_op, N_BON, 100, rng_out)
                mv = bon.get("majority_vote")
                if mv:
                    predict_fields[f"predict_{strat_name}"] = mv

        example = {
            "input": r.get("input", ""),
            "output": r.get("correct_letter", ""),
            **ex_meta,
            **predict_fields,
        }
        output_examples.append(example)

    # Top-level metadata (full aggregate results)
    output = {
        "metadata": {
            "experiment_id": "iter3_exp1_arrhenius_weak_model",
            "model_name": _selected_model,
            "model_selection_log": model_selection_log,
            "pilot_gate_passed": pilot_gate_result.get("pilot_gate_passed"),
            "pilot_oc_rate": pilot_gate_result.get("pilot_oc_rate"),
            "pilot_rising_frac": pilot_gate_result.get("pilot_rising_frac"),
            "n_oc_instances_main": len(oc_instances),
            "n_valid_fit_instances": n_valid,
            "valid_fit_rate": n_valid / max(len(fit_results), 1),
            "two_token_dominance_confirmed": dominance.get("two_token_dominance_confirmed"),
            "rho_ea_delta": dominance.get("rho_ea_delta"),
            "p_ea_delta": dominance.get("p_ea_delta"),
            "cv_log_A": dominance.get("cv_log_A"),
            "mean_log_A": dominance.get("mean_log_A"),
            "median_R2_arrhenius": median_R2,
            "fraction_R2_gt_085": frac_R2_085,
            "median_R2_advantage_over_linear": median_R2_adv,
            "T_TURN": T_TURN,
            "fraction_thresh_lt_TURN": frac_thresh_lt_TURN,
            "window_nontrivial_frac": window_nontrivial_frac,
            "thresh_lower_bound_by_N": thresh_by_N,
            "accuracy_table": accuracy_table,
            "rho_ea_tpref": tpref_result.get("rho_ea_tpref"),
            "p_ea_tpref": tpref_result.get("p_ea_tpref"),
            "partial_rho_ea_tpref": tpref_result.get("partial_rho_ea_tpref"),
            "p_partial": tpref_result.get("p_partial"),
            "R2_ea_tpref": tpref_result.get("R2_ea_tpref"),
            "catalysis": {
                "n_instances": catalysis_result.get("n_instances", 0),
                "n_cot_reduces_ea": catalysis_result.get("n_cot_reduces_ea", 0),
                "catalysis_fraction": catalysis_result.get("catalysis_fraction"),
                "catalysis_ci": catalysis_result.get("catalysis_ci"),
                "paired_t_pvalue": catalysis_result.get("paired_t_pvalue"),
            },
            "total_cost_usd": _total_cost_usd,
            "temp_grid": TEMP_GRID,
            "n_samples_per_temp": args.n_samples,
            "N_BON": N_BON,
            "turn_diagnostics": turn_result,
        },
        "datasets": [
            {
                "dataset": "TIGER-Lab/MMLU-Pro",
                "examples": output_examples,
            }
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    logger.info(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB, {len(output_examples)} examples)")
    logger.info(f"Total cost: ${_total_cost_usd:.4f}")
    return output


@logger.catch(reraise=True)
def main() -> None:
    parser = argparse.ArgumentParser(description="Arrhenius Kinetics Iter-3 Experiment")
    parser.add_argument("--mini", action="store_true", help="Test mode: 9 examples only")
    parser.add_argument("--max-instances", type=int, default=200, help="Max OC instances for sweep")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_TEMP, help="Samples per temperature")
    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
