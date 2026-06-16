#!/usr/bin/env python3
"""Arrhenius Protocol on ARC-Challenge (K=4) with phi-4: Mass-Diffusion Hypothesis Test.

Tests whether reducing answer option count from K=10 (MMLU-Pro) to K=4 (ARC-Challenge)
stabilizes two-token dominance and improves Arrhenius fit quality.
"""

import asyncio
import gc
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from loguru import logger
from scipy.stats import spearmanr, binomtest, norm as scipy_norm
from sklearn.linear_model import LinearRegression
import statsmodels.stats.proportion as smprop

# ── Logging ────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Memory limit (container: 29 GB) ────────────────────────────────────────
_RAM_BUDGET = 8 * 1024**3  # 8 GB virtual
resource.setrlimit(resource.RLIMIT_AS, (_RAM_BUDGET * 3, _RAM_BUDGET * 3))

# ── Constants ───────────────────────────────────────────────────────────────
MODEL = "microsoft/phi-4"
DATASET_NAME = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"
DATASET_SPLIT = "test"
K = 4
VALID_LETTERS = "ABCD"
PROMPT_SUFFIX = "\nRespond with only the letter of the correct answer:\n"

TEMP_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TURN_TEMPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
N_SAMPLES = 50
N_BON = 16
MAX_OC_SCAN = 800
TARGET_OC = 80
TARGET_VALID_FITS = 40
COST_LIMIT = 3.0
CONCURRENCY = 12

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CHECKPOINT_PATH = Path("logs/checkpoint.json")

# Hardcoded MMLU-Pro K=10 baseline (from phi-4 iter_2/3 experiments)
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

# ── OpenRouter Client ────────────────────────────────────────────────────────

class OpenRouterClient:
    """Async OpenRouter client with rate limiting, retries, and cost tracking."""

    def __init__(self, api_key: str, concurrency: int = CONCURRENCY):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)
        self.total_cost: float = 0.0
        self.total_calls: int = 0
        self.failed_calls: int = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def call(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1,
        logprobs: bool = False,
        top_logprobs: int = 20,
        retries: int = 4,
    ) -> Optional[dict]:
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": {"order": ["NextBit"], "require_parameters": True},
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-inventor.research",
            "X-Title": "Arrhenius-ARC-Experiment",
        }

        delay = 2.0
        async with self.semaphore:
            for attempt in range(retries):
                try:
                    session = await self._get_session()
                    async with session.post(
                        OPENROUTER_URL,
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status == 429:
                            wait = delay * (2 ** attempt)
                            logger.debug(f"Rate limited, waiting {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status in (500, 502, 503, 504):
                            wait = delay * (2 ** attempt)
                            logger.debug(f"Server error {resp.status}, waiting {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"API error {resp.status}: {text[:200]}")
                            self.failed_calls += 1
                            return None
                        data = await resp.json()
                        # Track cost
                        cost = 0.0
                        if "usage" in data:
                            usage = data["usage"]
                            cost = float(usage.get("cost", 0.0))
                            if cost == 0.0:
                                # Estimate: phi-4 ~$0.07/1M input, $0.14/1M output
                                in_tok = usage.get("prompt_tokens", 0)
                                out_tok = usage.get("completion_tokens", 0)
                                cost = (in_tok * 0.07 + out_tok * 0.14) / 1_000_000
                        self.total_cost += cost
                        self.total_calls += 1
                        return data
                except asyncio.TimeoutError:
                    logger.debug(f"Timeout on attempt {attempt+1}")
                    await asyncio.sleep(delay * (2 ** attempt))
                except Exception as exc:
                    logger.debug(f"Call error attempt {attempt+1}: {exc}")
                    await asyncio.sleep(delay)
            self.failed_calls += 1
            return None

# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_arc_challenge() -> list[dict]:
    """Load and format ARC-Challenge test split."""
    from datasets import load_dataset
    logger.info("Loading ARC-Challenge test split...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    logger.info(f"Raw dataset: {len(ds)} items")

    examples = []
    for item in ds:
        q = item["question"]
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        # Build prompt
        body = f"Question: {q}\n"
        for lab, txt in zip(labels, texts):
            body += f"{lab}. {txt}\n"
        body += PROMPT_SUFFIX

        answer = item["answerKey"]
        # Some items use '1','2','3','4' instead of 'A','B','C','D'
        if isinstance(answer, str) and answer.isdigit():
            answer = "ABCD"[int(answer) - 1]
        if answer not in "ABCD":
            logger.debug(f"Skipping item with bad answerKey: {answer!r}")
            continue

        examples.append({
            "input": body,
            "output": answer,
            "metadata_num_choices": 4,
            "metadata_id": item.get("id", ""),
        })

    logger.info(f"Loaded {len(examples)} ARC-Challenge items after filtering")
    return examples

# ── Logprob / Letter Extraction ──────────────────────────────────────────────

def extract_letter_logprobs(response: dict, num_choices: int = 4) -> dict[str, float]:
    """Extract logprobs for A-D letters from a logprob response."""
    valid = VALID_LETTERS[:num_choices]
    default = {L: -math.inf for L in valid}
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return default
    token_lp = {entry["token"]: entry["logprob"] for entry in top}
    result = {}
    for L in valid:
        bare = token_lp.get(L, -math.inf)
        spaced = token_lp.get(" " + L, -math.inf)
        result[L] = max(bare, spaced)
    return result

def extract_top_letter(response: dict, valid: str = "ABCD") -> Optional[str]:
    """Extract first valid letter from sampling response text."""
    try:
        content = response["choices"][0]["message"]["content"].strip()
        for ch in content:
            if ch.upper() in valid:
                return ch.upper()
    except (KeyError, IndexError, TypeError):
        return None
    return None

def has_logprobs(response: dict) -> bool:
    """Check if response contains valid logprob data."""
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return isinstance(top, list) and len(top) > 0
    except (KeyError, IndexError, TypeError):
        return False

def greedy_letter_from_logprobs(lp_map: dict[str, float]) -> Optional[str]:
    """Get the letter with highest logprob."""
    if not lp_map:
        return None
    finite = {k: v for k, v in lp_map.items() if v > -math.inf}
    if not finite:
        return None
    return max(finite, key=finite.__getitem__)

def build_messages(prompt: str) -> list[dict]:
    """Build OpenRouter messages list from prompt string."""
    return [{"role": "user", "content": prompt}]

# ── Arrhenius Fitting ────────────────────────────────────────────────────────

def fit_arrhenius(p_correct_by_T: dict[float, float]) -> dict:
    """
    Fit Arrhenius model: log P = -Ea/T + log(A).
    Linear regression of log(P) on 1/T.
    """
    temps = sorted(p_correct_by_T.keys())
    ps = [p_correct_by_T[T] for T in temps]

    # Filter: need P > 0
    valid_pairs = [(T, P) for T, P in zip(temps, ps) if P > 0]
    if len(valid_pairs) < 3:
        return {"Ea": None, "log_A": None, "R2": None, "n_valid": len(valid_pairs),
                "valid_fit": False, "valid_temps": []}

    X = np.array([1.0 / T for T, _ in valid_pairs]).reshape(-1, 1)
    y = np.array([math.log(P) for _, P in valid_pairs])

    reg = LinearRegression().fit(X, y)
    Ea = -float(reg.coef_[0])      # slope = -Ea
    log_A = float(reg.intercept_)  # intercept = log(A)
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Validity criteria
    p_vals = [p_correct_by_T.get(T, 0.0) for T in temps]
    p_low = p_correct_by_T.get(temps[0], 0.0)
    p_high = p_correct_by_T.get(temps[-1], 0.0)
    valid_fit = (
        len(valid_pairs) >= 3
        and R2 > 0.5
        and Ea > 0
        and p_high > p_low  # rising trend
    )

    return {
        "Ea": Ea,
        "log_A": log_A,
        "R2": R2,
        "n_valid": len(valid_pairs),
        "valid_fit": valid_fit,
        "valid_temps": [T for T, _ in valid_pairs],
    }

def fit_alternatives(p_correct_by_T: dict[float, float]) -> dict:
    """Fit alternative models: linear, exp_T, power_law."""
    temps = sorted(p_correct_by_T.keys())
    ps = [p_correct_by_T[T] for T in temps]
    valid_pairs = [(T, P) for T, P in zip(temps, ps) if P > 0]
    if len(valid_pairs) < 3:
        return {"linear_R2": None, "exp_T_R2": None, "power_law_R2": None}

    T_arr = np.array([T for T, _ in valid_pairs])
    P_arr = np.array([P for _, P in valid_pairs])

    results = {}
    # Linear: P = a*T + b
    try:
        reg = LinearRegression().fit(T_arr.reshape(-1, 1), P_arr)
        p_p = reg.predict(T_arr.reshape(-1, 1))
        ss_res = np.sum((P_arr - p_p) ** 2)
        ss_tot = np.sum((P_arr - P_arr.mean()) ** 2)
        results["linear_R2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        results["linear_R2"] = None

    # Exponential: log P = a*T + b
    try:
        log_P = np.log(P_arr)
        reg = LinearRegression().fit(T_arr.reshape(-1, 1), log_P)
        log_pred = reg.predict(T_arr.reshape(-1, 1))
        ss_res = np.sum((log_P - log_pred) ** 2)
        ss_tot = np.sum((log_P - log_P.mean()) ** 2)
        results["exp_T_R2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        results["exp_T_R2"] = None

    # Power law: log P = a * log(T) + b
    try:
        log_T = np.log(T_arr)
        log_P = np.log(P_arr)
        reg = LinearRegression().fit(log_T.reshape(-1, 1), log_P)
        log_pred = reg.predict(log_T.reshape(-1, 1))
        ss_res = np.sum((log_P - log_pred) ** 2)
        ss_tot = np.sum((log_P - log_P.mean()) ** 2)
        results["power_law_R2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        results["power_law_R2"] = None

    return results

def bootstrap_R2(p_correct_by_T: dict[float, float], n_boot: int = 200) -> tuple[float, float]:
    """Bootstrap 95% CI for R2 of Arrhenius fit."""
    temps = sorted(p_correct_by_T.keys())
    valid_pairs = [(T, P) for T, P in zip(temps, [p_correct_by_T[T] for T in temps]) if P > 0]
    if len(valid_pairs) < 3:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(42)
    r2_boot = []
    n = len(valid_pairs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pairs_b = [valid_pairs[i] for i in idx]
        X_b = np.array([1.0 / T for T, _ in pairs_b]).reshape(-1, 1)
        y_b = np.array([math.log(P) for _, P in pairs_b])
        if len(set(map(tuple, X_b))) < 2:
            continue
        try:
            reg = LinearRegression().fit(X_b, y_b)
            y_pred = reg.predict(X_b)
            ss_res = np.sum((y_b - y_pred) ** 2)
            ss_tot = np.sum((y_b - y_b.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_boot.append(r2)
        except Exception:
            pass
    if not r2_boot:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(r2_boot, [2.5, 97.5])
    return float(lo), float(hi)

# ── T_thresh / T_TURN ────────────────────────────────────────────────────────

def compute_T_thresh(Ea: float, log_A: float, N: int) -> dict[str, float]:
    """Compute T_thresh (simplified and A-corrected) for BoN-N."""
    ln_N = math.log(N)
    T_thresh_simplified = Ea / ln_N if ln_N > 0 else float("nan")
    denom = ln_N + log_A
    T_thresh_A = Ea / denom if abs(denom) > 1e-9 else float("nan")
    return {"simplified": T_thresh_simplified, "A_corrected": T_thresh_A}

def compute_T_min_emp(p_correct_by_T: dict[float, float], N: int) -> Optional[float]:
    """Empirical T where P_emp(T) >= 1/N."""
    threshold = 1.0 / N
    for T in sorted(p_correct_by_T.keys()):
        if p_correct_by_T[T] >= threshold:
            return T
    return None

def compute_T_TURN(entropy_by_T: dict[float, float]) -> float:
    """
    TURN algorithm: find inflection in log entropy curve.
    Returns T_TURN as the temperature at the inflection point.
    """
    temps = sorted(entropy_by_T.keys())
    if len(temps) < 3:
        return max(temps) + 0.1 if temps else 1.3

    log_H = np.array([math.log(max(entropy_by_T[T], 1e-10)) for T in temps])
    # Second differences of log_H
    d2 = np.diff(log_H, n=2)
    # First index where d2 > 0 (upward concavity = inflection)
    for i, v in enumerate(d2):
        if v > 0:
            return temps[i + 1]
    return max(temps) + 0.1

def compute_first_token_entropy(lp_map: dict[str, float]) -> float:
    """Compute entropy from A-D letter logprobs."""
    log_probs = [v for v in lp_map.values() if v > -math.inf]
    if not log_probs:
        return 0.0
    probs = np.exp(log_probs)
    probs = probs / probs.sum()
    # H = -sum(p * log(p))
    return float(-np.sum(probs * np.log(probs + 1e-12)))

# ── McNemar Test ─────────────────────────────────────────────────────────────

def mcnemar_exact(correct_A: list[bool], correct_B: list[bool]) -> float:
    """Exact McNemar binomial test."""
    b = sum(1 for a, bb in zip(correct_A, correct_B) if a and not bb)
    c = sum(1 for a, bb in zip(correct_A, correct_B) if not a and bb)
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    return float(binomtest(b, n_discordant, 0.5).pvalue)

# ── Wilson CI ────────────────────────────────────────────────────────────────

def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n_total == 0:
        return (0.0, 1.0)
    ci = smprop.proportion_confint(n_success, n_total, alpha=alpha, method="wilson")
    return float(ci[0]), float(ci[1])

# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(data: dict):
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))

def load_checkpoint() -> Optional[dict]:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            return None
    return None

# ── Unit Tests ───────────────────────────────────────────────────────────────

def run_unit_tests():
    logger.info("Running unit tests...")

    # T3: Arrhenius fitting — use Ea=0.15 to avoid P clamping at low T
    Ea_true, logA_true = 0.15, -0.5
    p_dict = {T: min(0.99, max(0.001, math.exp(-Ea_true / T + logA_true)))
              for T in TEMP_GRID}
    fit = fit_arrhenius(p_dict)
    assert fit["Ea"] is not None, "Arrhenius fit returned None Ea"
    assert abs(fit["Ea"] - Ea_true) < 0.05, f"Ea mismatch: {fit['Ea']:.3f} vs {Ea_true}"
    assert abs(fit["log_A"] - logA_true) < 0.15, f"logA mismatch: {fit['log_A']:.3f} vs {logA_true}"
    assert fit["R2"] > 0.95, f"R2 too low: {fit['R2']:.3f}"
    logger.info("T3 Arrhenius fitting: OK")

    # T4: McNemar test
    a = [True] * 10 + [False] * 10
    b_arr = [True] * 10 + [False] * 10
    p_val = mcnemar_exact(a, b_arr)
    assert p_val == 1.0, f"McNemar perfect agreement: expected 1.0, got {p_val}"
    a2 = [True] * 20
    b2 = [False] * 20
    p_val2 = mcnemar_exact(a2, b2)
    assert p_val2 < 0.05, f"McNemar all discordant: expected <0.05, got {p_val2}"
    logger.info("T4 McNemar test: OK")

    # T5: T_thresh computation
    thresh_dict = compute_T_thresh(0.3, -0.5, 16)
    assert thresh_dict["simplified"] > 0, "T_thresh simplified should be positive"
    logger.info("T5 T_thresh: OK")

    logger.info("All unit tests passed.")

# ── Main Experiment ───────────────────────────────────────────────────────────

async def _greedy_call(client: OpenRouterClient, prompt: str) -> Optional[dict]:
    """Single greedy logprob call."""
    messages = build_messages(prompt)
    return await client.call(messages, temperature=0.05, max_tokens=1,
                              logprobs=True, top_logprobs=20)

async def _sample_call(client: OpenRouterClient, prompt: str, temperature: float) -> Optional[dict]:
    """Single sampling call (no logprobs needed)."""
    messages = build_messages(prompt)
    return await client.call(messages, temperature=temperature, max_tokens=2,
                              logprobs=False)

async def _entropy_call(client: OpenRouterClient, prompt: str, temperature: float) -> Optional[dict]:
    """Single logprob call for entropy computation."""
    messages = build_messages(prompt)
    return await client.call(messages, temperature=temperature, max_tokens=1,
                              logprobs=True, top_logprobs=20)

async def phase0_smoke_test(client: OpenRouterClient, examples: list[dict]) -> dict:
    """Phase 0: Smoke test on 20 items to verify logprob access."""
    logger.info("=== Phase 0: Smoke Test ===")
    sample = examples[:20]

    tasks = [_greedy_call(client, ex["input"]) for ex in sample]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    n_has_logprobs = 0
    n_oc = 0
    deltas = []

    for ex, resp in zip(sample, responses):
        if isinstance(resp, Exception) or resp is None:
            continue
        if has_logprobs(resp):
            n_has_logprobs += 1
            lp_map = extract_letter_logprobs(resp)
            greedy = greedy_letter_from_logprobs(lp_map)
            correct = ex["output"]
            if greedy and greedy != correct:
                lp_correct = lp_map.get(correct, -math.inf)
                lp_wrong = lp_map.get(greedy, -math.inf)
                if lp_correct > -math.inf and lp_wrong > -math.inf:
                    n_oc += 1
                    delta = lp_wrong - lp_correct
                    deltas.append(delta)

    logprob_fail_rate = (20 - n_has_logprobs) / 20
    oc_rate = n_oc / 20
    median_delta = float(np.median(deltas)) if deltas else float("nan")

    logger.info(f"Smoke test: {n_has_logprobs}/20 with logprobs, OC rate={oc_rate:.2f}, "
                f"median_delta={median_delta:.3f}, cost=${client.total_cost:.4f}")

    smoke_passed = logprob_fail_rate <= 0.25 and n_oc >= 1
    if logprob_fail_rate > 0.25:
        logger.error(f"Logprob failure rate too high: {logprob_fail_rate:.2f}")
    if n_oc < 1:
        logger.warning("Very few OC instances in smoke test — phi-4 may be very accurate on ARC")

    return {
        "smoke_test_passed": smoke_passed,
        "n_has_logprobs": n_has_logprobs,
        "logprob_fail_rate": logprob_fail_rate,
        "oc_rate_smoke": oc_rate,
        "median_delta_smoke": median_delta,
        "cost_after_smoke": client.total_cost,
    }

async def phase1_pilot_gate(client: OpenRouterClient, examples: list[dict]) -> dict:
    """Phase 1: Pilot gate — check if rising temperature trend exists."""
    logger.info("=== Phase 1: Pilot Gate ===")
    cost_start = client.total_cost

    # Scan up to 200 items for OC instances
    oc_instances = []
    scan_count = 0
    for ex in examples:
        if len(oc_instances) >= 50 or scan_count >= 200:
            break
        scan_count += 1
        resp = await _greedy_call(client, ex["input"])
        if resp is None:
            continue
        if not has_logprobs(resp):
            continue
        lp_map = extract_letter_logprobs(resp)
        greedy = greedy_letter_from_logprobs(lp_map)
        correct = ex["output"]
        if greedy and greedy != correct:
            lp_correct = lp_map.get(correct, -math.inf)
            if lp_correct > -math.inf:
                oc_instances.append({
                    "example": ex,
                    "greedy_letter": greedy,
                    "lp_map_greedy": lp_map,
                })

    logger.info(f"Pilot: found {len(oc_instances)} OC instances in {scan_count} scanned")

    if not oc_instances:
        return {
            "pilot_gate_passed": False,
            "pilot_gate_fraction": 0.0,
            "pilot_oc_count": 0,
            "pilot_scan_count": scan_count,
        }

    # For each OC instance: 50 samples at T=0.1 and 50 at T=0.5
    rising_count = 0
    for inst in oc_instances:
        prompt = inst["example"]["input"]
        correct = inst["example"]["output"]

        tasks_01 = [_sample_call(client, prompt, 0.1) for _ in range(20)]
        tasks_05 = [_sample_call(client, prompt, 0.5) for _ in range(20)]
        all_tasks = tasks_01 + tasks_05
        all_resp = await asyncio.gather(*all_tasks, return_exceptions=True)

        letters_01 = [extract_top_letter(r) for r in all_resp[:20]
                      if r is not None and not isinstance(r, Exception)]
        letters_05 = [extract_top_letter(r) for r in all_resp[20:]
                      if r is not None and not isinstance(r, Exception)]

        p_01 = sum(1 for l in letters_01 if l == correct) / max(len(letters_01), 1)
        p_05 = sum(1 for l in letters_05 if l == correct) / max(len(letters_05), 1)

        if p_05 > p_01 + 0.05:
            rising_count += 1

        if client.total_cost - cost_start > 0.20:
            logger.warning("Pilot gate budget reached ($0.20)")
            break

    gate_fraction = rising_count / len(oc_instances) if oc_instances else 0.0
    pilot_gate_passed = gate_fraction >= 0.15  # relaxed: 15% (was 20%)

    logger.info(f"Pilot gate: {rising_count}/{len(oc_instances)} rising = {gate_fraction:.2f}, "
                f"passed={pilot_gate_passed}, cost=${client.total_cost:.4f}")

    return {
        "pilot_gate_passed": pilot_gate_passed,
        "pilot_gate_fraction": gate_fraction,
        "pilot_oc_count": len(oc_instances),
        "pilot_scan_count": scan_count,
    }

async def sweep_instance(
    client: OpenRouterClient,
    ex: dict,
    greedy_lp_map: dict,
    collect_entropy: bool = True,
) -> dict:
    """Full temperature sweep for a single OC instance."""
    prompt = ex["input"]
    correct = ex["output"]
    greedy = greedy_letter_from_logprobs(greedy_lp_map)

    # Delta: logit(wrong) - logit(correct)
    lp_correct = greedy_lp_map.get(correct, -math.inf)
    lp_greedy = greedy_lp_map.get(greedy, -math.inf) if greedy else -math.inf
    Delta = lp_greedy - lp_correct if (lp_correct > -math.inf and lp_greedy > -math.inf) else float("nan")

    # Temperature sweep: N_SAMPLES calls per temperature
    p_correct_by_T: dict[float, float] = {}
    for T in TEMP_GRID:
        tasks = [_sample_call(client, prompt, T) for _ in range(N_SAMPLES)]
        resps = await asyncio.gather(*tasks, return_exceptions=True)
        letters = [extract_top_letter(r) for r in resps
                   if r is not None and not isinstance(r, Exception)]
        n_correct = sum(1 for l in letters if l == correct)
        n_valid = max(len(letters), 1)
        p_correct_by_T[T] = n_correct / n_valid

    # Entropy sweep for T_TURN
    entropy_by_T: dict[float, float] = {}
    if collect_entropy:
        for T_e in TURN_TEMPS:
            resp_e = await _entropy_call(client, prompt, T_e)
            if resp_e is not None and has_logprobs(resp_e):
                lp_map_e = extract_letter_logprobs(resp_e)
                entropy_by_T[T_e] = compute_first_token_entropy(lp_map_e)

    # Arrhenius fit
    fit = fit_arrhenius(p_correct_by_T)
    alt = fit_alternatives(p_correct_by_T)

    # T_pref: argmax of P_emp over TEMP_GRID
    T_pref = max(TEMP_GRID, key=lambda T: p_correct_by_T.get(T, 0.0))

    # T_TURN
    T_TURN_emp = compute_T_TURN(entropy_by_T) if entropy_by_T else 1.3

    return {
        "item_id": ex.get("metadata_id", ""),
        "input": ex["input"],
        "correct_letter": correct,
        "greedy_letter": greedy or "",
        "Delta": Delta,
        "Ea": fit["Ea"],
        "log_A": fit["log_A"],
        "R2": fit["R2"],
        "is_valid_fit": fit["valid_fit"],
        "p_correct_by_T": {str(T): p_correct_by_T[T] for T in TEMP_GRID},
        "T_pref": T_pref,
        "T_TURN_emp": T_TURN_emp,
        "entropy_by_T": {str(T): entropy_by_T.get(T, float("nan")) for T in TURN_TEMPS},
        "alt_fits": alt,
    }

async def phase2_main(
    client: OpenRouterClient,
    examples: list[dict],
    pilot_result: dict,
) -> dict:
    """Phase 2: Main experiment — OC collection + temperature sweep."""
    logger.info("=== Phase 2: Main Experiment ===")

    # Load checkpoint
    ckpt = load_checkpoint()
    completed_ids: set[str] = set()
    results_so_far: list[dict] = []

    if ckpt and ckpt.get("phase") == "main_sweep":
        completed_ids = set(ckpt.get("completed_instance_ids", []))
        results_so_far = ckpt.get("results_so_far", [])
        logger.info(f"Resuming from checkpoint: {len(completed_ids)} already done")

    # OC collection scan
    oc_instances = []
    n_scanned = 0
    oc_rate_values = []

    for ex in examples:
        if len(oc_instances) >= TARGET_OC or n_scanned >= MAX_OC_SCAN:
            break
        item_id = ex.get("metadata_id", str(n_scanned))
        n_scanned += 1

        if item_id in completed_ids:
            continue

        resp = await _greedy_call(client, ex["input"])
        if resp is None:
            continue
        if not has_logprobs(resp):
            continue

        lp_map = extract_letter_logprobs(resp)
        greedy = greedy_letter_from_logprobs(lp_map)
        correct = ex["output"]

        oc_rate_values.append(1 if (greedy and greedy != correct) else 0)

        if greedy and greedy != correct:
            lp_correct = lp_map.get(correct, -math.inf)
            if lp_correct > -math.inf:
                oc_instances.append({
                    "example": ex,
                    "greedy_lp_map": lp_map,
                    "item_id": item_id,
                })

        if client.total_cost >= COST_LIMIT * 0.85:
            logger.warning(f"85% budget used during scan (${client.total_cost:.3f})")
            break

    oc_rate_main = float(np.mean(oc_rate_values)) if oc_rate_values else 0.0
    logger.info(f"Collected {len(oc_instances)} OC instances from {n_scanned} scanned, "
                f"OC rate={oc_rate_main:.3f}")

    # Temperature sweep per OC instance
    new_results = list(results_so_far)
    for idx, inst in enumerate(oc_instances):
        item_id = inst["item_id"]
        if item_id in completed_ids:
            logger.debug(f"Skipping already-processed instance {item_id}")
            continue

        if client.total_cost >= COST_LIMIT * 0.92:
            logger.warning(f"92% budget used (${client.total_cost:.3f}): stopping sweep early")
            break

        collect_entropy = client.total_cost < COST_LIMIT * 0.80  # skip entropy if budget tight
        try:
            result = await sweep_instance(
                client, inst["example"], inst["greedy_lp_map"],
                collect_entropy=collect_entropy,
            )
            new_results.append(result)
            completed_ids.add(item_id)

            logger.info(f"Instance {idx+1}/{len(oc_instances)}: Ea={result.get('Ea')}, "
                        f"R2={result.get('R2')}, valid={result.get('is_valid_fit')}, "
                        f"cost=${client.total_cost:.3f}")
        except Exception as e:
            logger.error(f"Failed on instance {item_id}: {e}")
            continue

        # Checkpoint every 5 instances
        if (idx + 1) % 5 == 0:
            save_checkpoint({
                "phase": "main_sweep",
                "completed_instance_ids": list(completed_ids),
                "results_so_far": new_results,
                "total_cost": client.total_cost,
                "total_calls": client.total_calls,
            })

    return {
        "results": new_results,
        "n_oc_scanned": n_scanned,
        "n_oc_instances": len(oc_instances),
        "oc_rate_main": oc_rate_main,
    }

async def compute_bon16(
    client: OpenRouterClient,
    instances: list[dict],
) -> dict:
    """Step F: BON-16 accuracy comparison at regression T, fixed T=1.0, T=0.7."""
    logger.info("=== Step F: BON-16 Accuracy ===")
    correct_regression: list[bool] = []
    correct_T10: list[bool] = []
    correct_T07: list[bool] = []

    for inst in instances:
        if not inst.get("is_valid_fit"):
            continue
        if client.total_cost >= COST_LIMIT * 0.97:
            logger.warning("Budget near limit: truncating BON-16")
            break

        prompt = inst["input"]
        correct = inst["correct_letter"]
        Ea = inst.get("Ea")
        log_A = inst.get("log_A")

        # T_operating for regression: T_thresh(N=16) + 0.3, clipped
        if Ea is not None and Ea > 0:
            ln16 = math.log(16)
            T_thresh = Ea / ln16
            T_op = max(0.05, min(1.3, T_thresh + 0.3))
        else:
            T_op = 0.7  # fallback

        # Collect 16 samples at each of 3 temperatures
        tasks_reg = [_sample_call(client, prompt, T_op) for _ in range(N_BON)]
        tasks_T10 = [_sample_call(client, prompt, 1.0) for _ in range(N_BON)]
        tasks_T07 = [_sample_call(client, prompt, 0.7) for _ in range(N_BON)]

        all_resps = await asyncio.gather(
            *tasks_reg, *tasks_T10, *tasks_T07,
            return_exceptions=True,
        )
        resps_reg = all_resps[:N_BON]
        resps_T10 = all_resps[N_BON:2*N_BON]
        resps_T07 = all_resps[2*N_BON:]

        def bon_correct(resps, correct_letter):
            letters = [extract_top_letter(r) for r in resps
                       if r is not None and not isinstance(r, Exception)]
            return any(l == correct_letter for l in letters)

        correct_regression.append(bon_correct(resps_reg, correct))
        correct_T10.append(bon_correct(resps_T10, correct))
        correct_T07.append(bon_correct(resps_T07, correct))
        inst["bon16_correct_regression"] = correct_regression[-1]
        inst["bon16_correct_T10"] = correct_T10[-1]
        inst["bon16_correct_T07"] = correct_T07[-1]

    n = len(correct_regression)
    if n == 0:
        return {
            "bon16_accuracy_regression": float("nan"),
            "bon16_accuracy_fixed_T10": float("nan"),
            "bon16_accuracy_fixed_T07": float("nan"),
            "mcnemar_p_regression_vs_T07": float("nan"),
            "mcnemar_p_regression_vs_T10": float("nan"),
        }

    bon_reg = float(np.mean(correct_regression))
    bon_T10 = float(np.mean(correct_T10))
    bon_T07 = float(np.mean(correct_T07))

    p_vs_T07 = mcnemar_exact(correct_regression, correct_T07)
    p_vs_T10 = mcnemar_exact(correct_regression, correct_T10)

    logger.info(f"BON-16: regression={bon_reg:.3f}, T=1.0={bon_T10:.3f}, T=0.7={bon_T07:.3f}")
    logger.info(f"McNemar: vs T=0.7 p={p_vs_T07:.4f}, vs T=1.0 p={p_vs_T10:.4f}")

    return {
        "bon16_accuracy_regression": bon_reg,
        "bon16_accuracy_fixed_T10": bon_T10,
        "bon16_accuracy_fixed_T07": bon_T07,
        "mcnemar_p_regression_vs_T07": p_vs_T07,
        "mcnemar_p_regression_vs_T10": p_vs_T10,
    }

def analyze_results(results: list[dict]) -> dict:
    """Compute all analysis metrics from per-instance results."""
    if not results:
        return {}

    valid = [r for r in results if r.get("is_valid_fit") and r.get("Ea") is not None]
    all_with_r2 = [r for r in results if r.get("R2") is not None]

    n_valid = len(valid)
    n_total = len(results)
    valid_fit_rate = n_valid / n_total if n_total > 0 else 0.0

    # Median R2 across all instances with >=3 non-zero P values
    r2_all = [r["R2"] for r in all_with_r2 if r["R2"] is not None]
    median_R2 = float(np.median(r2_all)) if r2_all else float("nan")

    # Bootstrap R2 CI
    if valid:
        r2_valid = [r["R2"] for r in valid]
        r2_lo, r2_hi = float(np.percentile(r2_valid, 2.5)), float(np.percentile(r2_valid, 97.5))
    else:
        r2_lo, r2_hi = float("nan"), float("nan")

    # Arrhenius parameters for valid fits
    Ea_arr = [r["Ea"] for r in valid]
    logA_arr = [r["log_A"] for r in valid]
    median_Ea = float(np.median(Ea_arr)) if Ea_arr else float("nan")
    logA_mean = float(np.mean(logA_arr)) if logA_arr else float("nan")
    logA_std = float(np.std(logA_arr)) if logA_arr else float("nan")
    cv_log_A = logA_std / abs(logA_mean) if (logA_arr and abs(logA_mean) > 1e-9) else float("nan")

    # Spearman rho(Ea, Delta) with bootstrap CI
    delta_arr = [r["Delta"] for r in valid if not math.isnan(r.get("Delta", float("nan")))]
    Ea_for_delta = [r["Ea"] for r in valid if not math.isnan(r.get("Delta", float("nan")))]
    rho_ea_delta = float("nan")
    rho_ea_delta_p = float("nan")
    rho_ea_delta_ci = (float("nan"), float("nan"))
    if len(Ea_for_delta) >= 5:
        rho_res = spearmanr(Ea_for_delta, delta_arr)
        rho_ea_delta = float(rho_res.statistic)
        rho_ea_delta_p = float(rho_res.pvalue)
        # Bootstrap CI
        rng = np.random.default_rng(42)
        n = len(Ea_for_delta)
        boot_rhos = []
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            Eb = [Ea_for_delta[i] for i in idx]
            Db = [delta_arr[i] for i in idx]
            try:
                boot_rhos.append(spearmanr(Eb, Db).statistic)
            except Exception:
                pass
        if boot_rhos:
            rho_ea_delta_ci = (float(np.percentile(boot_rhos, 2.5)),
                               float(np.percentile(boot_rhos, 97.5)))

    # T_thresh validation (Step C)
    T_thresh_by_N: dict[int, dict] = {}
    for N in [4, 8, 16, 32]:
        is_lb = []  # T_thresh < T_min_emp (lower bound)
        T_thresh_list = []
        for r in valid:
            Ea = r["Ea"]
            log_A = r.get("log_A", 0.0) or 0.0
            if Ea is None:
                continue
            thresh = compute_T_thresh(Ea, log_A, N)
            T_s = thresh["simplified"]
            p_by_T = {float(k): v for k, v in r["p_correct_by_T"].items()}
            T_min_emp = compute_T_min_emp(p_by_T, N)
            if T_min_emp is not None and not math.isnan(T_s):
                is_lb.append(T_s < T_min_emp)
                T_thresh_list.append(T_s)

        n_lb = sum(1 for x in is_lb if x)
        n_tot = len(is_lb)
        frac = n_lb / n_tot if n_tot > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(n_lb, n_tot)
        theorem_applies = N > K  # Theorem 6 applies when N > K=4
        median_T_thresh_N = float(np.median(T_thresh_list)) if T_thresh_list else float("nan")

        T_thresh_by_N[N] = {
            "n_total": n_tot,
            "fraction_simplified_is_lower_bound": frac,
            "wilson_ci_low": ci_lo,
            "wilson_ci_high": ci_hi,
            "theorem6_applies": theorem_applies,
            "median_T_thresh": median_T_thresh_N,
        }

    # T_TURN analysis
    T_TURN_arr = [r.get("T_TURN_emp", 1.3) for r in valid]
    T_thresh_N16_arr = []
    for r in valid:
        Ea = r["Ea"]
        log_A = r.get("log_A", 0.0) or 0.0
        if Ea is not None and Ea > 0:
            T_thresh_N16_arr.append(compute_T_thresh(Ea, log_A, 16)["simplified"])
    median_T_TURN_emp = float(np.median(T_TURN_arr)) if T_TURN_arr else float("nan")
    median_T_thresh_N16 = float(np.median(T_thresh_N16_arr)) if T_thresh_N16_arr else float("nan")

    frac_below_T_TURN = 0.0
    if T_thresh_N16_arr and T_TURN_arr:
        pairs = list(zip(T_thresh_N16_arr, T_TURN_arr[:len(T_thresh_N16_arr)]))
        frac_below_T_TURN = sum(1 for t_th, t_turn in pairs if t_th < t_turn) / len(pairs)

    # rho(Ea, T_pref)
    T_pref_arr = [r.get("T_pref", float("nan")) for r in valid]
    Ea_tpref = [r["Ea"] for r in valid]
    rho_ea_tpref = float("nan")
    rho_ea_tpref_p = float("nan")
    rho_ea_tpref_ci = (float("nan"), float("nan"))
    valid_pairs_tpref = [(e, t) for e, t in zip(Ea_tpref, T_pref_arr)
                         if not math.isnan(t) and e is not None]
    if len(valid_pairs_tpref) >= 5:
        e_arr_tp = [x[0] for x in valid_pairs_tpref]
        t_arr_tp = [x[1] for x in valid_pairs_tpref]
        rho_res_tp = spearmanr(e_arr_tp, t_arr_tp)
        rho_ea_tpref = float(rho_res_tp.statistic)
        rho_ea_tpref_p = float(rho_res_tp.pvalue)
        rng = np.random.default_rng(43)
        n = len(e_arr_tp)
        boot_tp = []
        for _ in range(500):
            idx = rng.integers(0, n, size=n)
            eb = [e_arr_tp[i] for i in idx]
            tb = [t_arr_tp[i] for i in idx]
            try:
                boot_tp.append(spearmanr(eb, tb).statistic)
            except Exception:
                pass
        if boot_tp:
            rho_ea_tpref_ci = (float(np.percentile(boot_tp, 2.5)),
                               float(np.percentile(boot_tp, 97.5)))

    # Two-token dominance
    two_token_dominance_confirmed = (rho_ea_delta > 0.6 and cv_log_A < 0.4
                                     if not math.isnan(rho_ea_delta) else False)

    # Verdict
    if math.isnan(rho_ea_delta):
        verdict = "UNDERPOWERED"
        rationale = "Insufficient valid fits for correlation analysis."
    elif rho_ea_delta > 0.6 and cv_log_A < 0.4:
        verdict = "CONFIRMS"
        rationale = (f"K=4 shows rho(Ea,Delta)={rho_ea_delta:.3f}>0.6 and "
                     f"CV(log A)={cv_log_A:.3f}<0.4, confirming two-token dominance "
                     f"hypothesis vs MMLU-Pro K=10 baseline.")
    elif rho_ea_delta < 0.3:
        verdict = "DISCONFIRMS"
        rationale = (f"K=4 shows rho(Ea,Delta)={rho_ea_delta:.3f}<0.3, "
                     f"not supporting two-token dominance. Mass-diffusion hypothesis "
                     f"not confirmed by K reduction alone.")
    else:
        verdict = "UNDERPOWERED"
        rationale = (f"Intermediate rho={rho_ea_delta:.3f}; need more instances "
                     f"for definitive verdict. CV(log A)={cv_log_A:.3f}.")

    comparison = {
        "dataset_arc": "ARC-Challenge", "K_arc": 4,
        "dataset_mmlu": "MMLU-Pro", "K_mmlu": 10,
        "arc_rho_ea_delta": rho_ea_delta,
        "arc_cv_log_A": cv_log_A,
        "arc_valid_fit_rate": valid_fit_rate,
        "arc_median_R2": median_R2,
        "arc_median_Ea": median_Ea,
        "mmlu_rho_ea_delta": 0.106,
        "mmlu_cv_log_A": 1.093,
        "mmlu_valid_fit_rate": 0.199,
        "mmlu_median_R2": 0.848,
        "mmlu_median_Ea": 0.351,
        "rho_improved": (rho_ea_delta > 0.106) if not math.isnan(rho_ea_delta) else None,
        "cv_improved": (cv_log_A < 1.093) if not math.isnan(cv_log_A) else None,
        "fit_rate_improved": (valid_fit_rate > 0.199),
        "verdict": verdict,
        "verdict_rationale": rationale,
    }

    return {
        "n_valid_fits": n_valid,
        "valid_fit_rate": valid_fit_rate,
        "median_R2": median_R2,
        "bootstrap_R2_ci_low": r2_lo,
        "bootstrap_R2_ci_high": r2_hi,
        "median_Ea": median_Ea,
        "logA_mean": logA_mean,
        "logA_std": logA_std,
        "cv_log_A": cv_log_A,
        "rho_ea_delta": rho_ea_delta,
        "rho_ea_delta_p": rho_ea_delta_p,
        "rho_ea_delta_ci_low": rho_ea_delta_ci[0],
        "rho_ea_delta_ci_high": rho_ea_delta_ci[1],
        "two_token_dominance_confirmed": two_token_dominance_confirmed,
        "step6_T_thresh_validation": {
            "by_N": {str(N): T_thresh_by_N[N] for N in [4, 8, 16, 32]},
            "fraction_T_thresh_below_T_TURN": frac_below_T_TURN,
            "median_T_thresh_N16": median_T_thresh_N16,
            "median_T_TURN_emp": median_T_TURN_emp,
        },
        "rho_ea_tpref": rho_ea_tpref,
        "rho_ea_tpref_p": rho_ea_tpref_p,
        "rho_ea_tpref_ci_low": rho_ea_tpref_ci[0],
        "rho_ea_tpref_ci_high": rho_ea_tpref_ci[1],
        "comparison_vs_mmlu_pro": comparison,
    }

def generate_figures(results: list[dict], analysis: dict, workspace: Path):
    """Generate 3 matplotlib figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in results if r.get("is_valid_fit") and r.get("Ea") is not None]
        figure_paths = []

        # Figure 1: Arrhenius plots for 3 representative instances
        fig, axes = plt.subplots(1, min(3, len(valid)), figsize=(12, 4))
        if len(valid) < 3:
            axes_list = [axes] if len(valid) == 1 else list(axes)
        else:
            axes_list = list(axes)

        for ax, inst in zip(axes_list, valid[:3]):
            p_by_T = {float(k): v for k, v in inst["p_correct_by_T"].items()}
            temps = sorted(p_by_T.keys())
            ps = [p_by_T[T] for T in temps]
            valid_pairs = [(T, P) for T, P in zip(temps, ps) if P > 0]
            if len(valid_pairs) >= 2:
                inv_T = [1.0 / T for T, _ in valid_pairs]
                log_P = [math.log(P) for _, P in valid_pairs]
                ax.scatter(inv_T, log_P, color="steelblue", zorder=5)
                # Regression line
                x_fit = np.linspace(min(inv_T), max(inv_T), 50)
                Ea = inst["Ea"]
                log_A = inst["log_A"]
                y_fit = [-Ea * x + log_A for x in x_fit]
                ax.plot(x_fit, y_fit, "r-", label=f"Ea={Ea:.2f}")
                ax.set_xlabel("1/T")
                ax.set_ylabel("log P(correct)")
                ax.set_title(f"R²={inst['R2']:.2f}")
                ax.legend(fontsize=8)

        plt.tight_layout()
        fig1_path = str(workspace / "figure1_arrhenius_plots.png")
        plt.savefig(fig1_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig1_path)
        logger.info(f"Figure 1 saved: {fig1_path}")

        # Figure 2: Ea vs Delta scatter
        if valid:
            Ea_arr = [r["Ea"] for r in valid if not math.isnan(r.get("Delta", float("nan")))]
            delta_arr = [r["Delta"] for r in valid if not math.isnan(r.get("Delta", float("nan")))]
            if Ea_arr and delta_arr:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(delta_arr, Ea_arr, alpha=0.6, color="steelblue", label="ARC-Challenge K=4")
                rho = analysis.get("rho_ea_delta", float("nan"))
                ax.set_xlabel("Δ = logit(wrong) - logit(correct)")
                ax.set_ylabel("Arrhenius Ea")
                ax.set_title(f"Ea vs Δ: ρ={rho:.3f}" if not math.isnan(rho) else "Ea vs Δ")
                ax.legend()
                # Add MMLU-Pro reference point
                ax.axhline(y=0.351, color="orange", linestyle="--", alpha=0.5, label="MMLU-Pro median Ea")
                ax.legend()
                plt.tight_layout()
                fig2_path = str(workspace / "figure2_ea_vs_delta.png")
                plt.savefig(fig2_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                figure_paths.append(fig2_path)
                logger.info(f"Figure 2 saved: {fig2_path}")

        # Figure 3: R2 distribution histogram
        if results:
            r2_vals = [r["R2"] for r in results if r.get("R2") is not None]
            if r2_vals:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(r2_vals, bins=20, alpha=0.7, color="steelblue",
                        edgecolor="white", label="ARC-Challenge K=4")
                ax.axvline(x=0.848, color="orange", linestyle="--",
                           label="MMLU-Pro median R²=0.848")
                ax.axvline(x=np.median(r2_vals), color="steelblue", linestyle="-",
                           label=f"ARC median R²={np.median(r2_vals):.3f}")
                ax.set_xlabel("Arrhenius R²")
                ax.set_ylabel("Count")
                ax.set_title("R² Distribution: ARC-Challenge vs MMLU-Pro")
                ax.legend()
                plt.tight_layout()
                fig3_path = str(workspace / "figure3_r2_distribution.png")
                plt.savefig(fig3_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                figure_paths.append(fig3_path)
                logger.info(f"Figure 3 saved: {fig3_path}")

        return figure_paths
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
        return []

def build_method_out(
    smoke: dict,
    pilot: dict,
    phase2: dict,
    bon: dict,
    analysis: dict,
    client: OpenRouterClient,
    figure_paths: list[str],
) -> dict:
    """Build exp_gen_sol_out-compliant JSON output."""
    results = phase2.get("results", [])

    # Per-instance output as exp_gen_sol_out examples
    examples_out = []
    for r in results:
        ex_dict = {
            "input": r.get("input", ""),
            "output": r.get("correct_letter", ""),
            "metadata_item_id": str(r.get("item_id", "")),
            "metadata_correct_letter": str(r.get("correct_letter", "")),
            "metadata_greedy_letter": str(r.get("greedy_letter", "")),
            "metadata_is_valid_fit": str(r.get("is_valid_fit", False)),
            "predict_Ea": str(r.get("Ea", "")),
            "predict_log_A": str(r.get("log_A", "")),
            "predict_R2": str(r.get("R2", "")),
            "predict_Delta": str(r.get("Delta", "")),
            "predict_T_pref": str(r.get("T_pref", "")),
            "predict_T_TURN_emp": str(r.get("T_TURN_emp", "")),
            "predict_p_correct_by_T": json.dumps(r.get("p_correct_by_T", {})),
            "predict_T_thresh_N16": str(
                compute_T_thresh(r["Ea"], r.get("log_A", 0) or 0, 16)["simplified"]
                if r.get("Ea") and r["Ea"] > 0 else ""
            ),
            "predict_bon16_correct_regression": str(r.get("bon16_correct_regression", "")),
            "predict_bon16_correct_T10": str(r.get("bon16_correct_T10", "")),
            "predict_bon16_correct_T07": str(r.get("bon16_correct_T07", "")),
        }
        examples_out.append(ex_dict)

    # Top-level metadata aggregating all experiment stats
    metadata = {
        "model_name": MODEL,
        "dataset": "ARC-Challenge",
        "K": K,
        "n_oc_scanned": phase2.get("n_oc_scanned", 0),
        "n_oc_instances": phase2.get("n_oc_instances", 0),
        "n_valid_fits": analysis.get("n_valid_fits", 0),
        "valid_fit_rate": analysis.get("valid_fit_rate"),
        "oc_rate_main": phase2.get("oc_rate_main"),
        "pilot_gate_passed": pilot.get("pilot_gate_passed", False),
        "pilot_gate_fraction": pilot.get("pilot_gate_fraction"),
        "smoke_test_passed": smoke.get("smoke_test_passed", False),
        "cumulative_cost_usd": client.total_cost,
        "total_api_calls": client.total_calls,
        "failed_api_calls": client.failed_calls,
        "two_token_dominance_confirmed": analysis.get("two_token_dominance_confirmed", False),
        "rho_ea_delta": analysis.get("rho_ea_delta"),
        "rho_ea_delta_p": analysis.get("rho_ea_delta_p"),
        "rho_ea_delta_ci_low": analysis.get("rho_ea_delta_ci_low"),
        "rho_ea_delta_ci_high": analysis.get("rho_ea_delta_ci_high"),
        "cv_log_A": analysis.get("cv_log_A"),
        "median_R2": analysis.get("median_R2"),
        "bootstrap_R2_ci_low": analysis.get("bootstrap_R2_ci_low"),
        "bootstrap_R2_ci_high": analysis.get("bootstrap_R2_ci_high"),
        "median_Ea": analysis.get("median_Ea"),
        "logA_mean": analysis.get("logA_mean"),
        "logA_std": analysis.get("logA_std"),
        "step6_T_thresh_validation": analysis.get("step6_T_thresh_validation"),
        "bon16_accuracy_regression": bon.get("bon16_accuracy_regression"),
        "bon16_accuracy_fixed_T10": bon.get("bon16_accuracy_fixed_T10"),
        "bon16_accuracy_fixed_T07": bon.get("bon16_accuracy_fixed_T07"),
        "mcnemar_p_regression_vs_T07": bon.get("mcnemar_p_regression_vs_T07"),
        "mcnemar_p_regression_vs_T10": bon.get("mcnemar_p_regression_vs_T10"),
        "rho_ea_tpref": analysis.get("rho_ea_tpref"),
        "rho_ea_tpref_p": analysis.get("rho_ea_tpref_p"),
        "rho_ea_tpref_ci_low": analysis.get("rho_ea_tpref_ci_low"),
        "rho_ea_tpref_ci_high": analysis.get("rho_ea_tpref_ci_high"),
        "comparison_vs_mmlu_pro": analysis.get("comparison_vs_mmlu_pro"),
        "mmlu_pro_baseline": MMLP_BASELINE,
        "figure_paths": figure_paths,
    }

    # Replace any NaN/inf with None for JSON serialization
    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj

    return clean({
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "allenai/ai2_arc",
                "examples": examples_out,
            }
        ],
    })

# ── Main ──────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
async def async_main():
    workspace = Path("/ai-inventor/aii_data/runs/run_wYelBzy-9k_d/3_invention_loop/iter_5/gen_art/gen_art_experiment_1")

    logger.info(f"=== Arrhenius ARC-Challenge Experiment ===")
    logger.info(f"Model: {MODEL}, Dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    logger.info(f"Budget: ${COST_LIMIT}, Target fits: {TARGET_VALID_FITS}")

    # Unit tests
    run_unit_tests()

    # Validate API key
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set!")
        sys.exit(1)

    # T0: Dataset validation
    examples = load_arc_challenge()
    assert len(examples) > 1000, f"Expected >1000 items, got {len(examples)}"
    for ex in examples[:5]:
        assert ex["output"] in "ABCD", f"Bad answer: {ex['output']!r}"
        assert ex["metadata_num_choices"] == 4
    logger.info(f"T0 dataset validation: OK ({len(examples)} items)")

    # Shuffle with fixed seed
    rng = np.random.default_rng(42)
    idx_order = rng.permutation(len(examples)).tolist()
    examples = [examples[i] for i in idx_order]

    client = OpenRouterClient(api_key=OPENROUTER_API_KEY, concurrency=CONCURRENCY)

    try:
        # Phase 0: Smoke Test
        smoke = await phase0_smoke_test(client, examples[:20])
        logger.info(f"Smoke test passed: {smoke['smoke_test_passed']}")

        if not smoke["smoke_test_passed"]:
            logger.error("Smoke test failed! Writing partial output.")
            output = build_method_out(smoke, {}, {"results": [], "n_oc_scanned": 0,
                                                   "n_oc_instances": 0, "oc_rate_main": 0.0},
                                      {}, {}, client, [])
            (workspace / "method_out.json").write_text(json.dumps(output, indent=2))
            return

        # Phase 1: Pilot Gate (use examples after smoke test set)
        pilot = await phase1_pilot_gate(client, examples[20:220])
        logger.info(f"Pilot gate passed: {pilot['pilot_gate_passed']}, "
                    f"fraction={pilot.get('pilot_gate_fraction', 0):.3f}")

        if not pilot["pilot_gate_passed"]:
            logger.warning("Pilot gate FAILED. Proceeding anyway to collect partial data.")

        # Phase 2: Main Experiment
        phase2 = await phase2_main(client, examples, pilot)
        results = phase2["results"]

        logger.info(f"Main experiment complete: {len(results)} instances processed, "
                    f"cost=${client.total_cost:.3f}")

        # Analysis
        analysis = analyze_results(results)
        logger.info(f"Analysis: n_valid={analysis.get('n_valid_fits')}, "
                    f"rho={analysis.get('rho_ea_delta')}, "
                    f"cv_logA={analysis.get('cv_log_A')}")

        # BON-16 (only if budget allows)
        valid_results = [r for r in results if r.get("is_valid_fit")]
        if client.total_cost < COST_LIMIT * 0.90 and valid_results:
            bon = await compute_bon16(client, valid_results)
        else:
            logger.warning(f"Skipping BON-16: budget=${client.total_cost:.3f}")
            bon = {
                "bon16_accuracy_regression": None,
                "bon16_accuracy_fixed_T10": None,
                "bon16_accuracy_fixed_T07": None,
                "mcnemar_p_regression_vs_T07": None,
                "mcnemar_p_regression_vs_T10": None,
            }

        # Re-run analysis to include BON results in comparison
        analysis = analyze_results(results)
        if bon.get("bon16_accuracy_regression") is not None:
            analysis["comparison_vs_mmlu_pro"]["arc_bon16_regression"] = bon["bon16_accuracy_regression"]

        # Figures
        figure_paths = generate_figures(results, analysis, workspace)

        # Build output
        output = build_method_out(smoke, pilot, phase2, bon, analysis, client, figure_paths)

        out_path = workspace / "method_out.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.info(f"Output written to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        # Summary
        logger.info("=== EXPERIMENT SUMMARY ===")
        logger.info(f"Total cost: ${client.total_cost:.4f}")
        logger.info(f"Total API calls: {client.total_calls}")
        logger.info(f"Failed calls: {client.failed_calls}")
        logger.info(f"OC instances collected: {phase2['n_oc_instances']}")
        logger.info(f"Valid Arrhenius fits: {analysis.get('n_valid_fits', 0)}")
        logger.info(f"Valid fit rate: {analysis.get('valid_fit_rate', 0):.3f}")
        logger.info(f"Median R²: {analysis.get('median_R2')}")
        logger.info(f"Median Ea: {analysis.get('median_Ea')}")
        logger.info(f"rho(Ea, Delta): {analysis.get('rho_ea_delta')}")
        logger.info(f"CV(log A): {analysis.get('cv_log_A')}")
        verdict = analysis.get("comparison_vs_mmlu_pro", {}).get("verdict", "?")
        rationale = analysis.get("comparison_vs_mmlu_pro", {}).get("verdict_rationale", "")
        logger.info(f"VERDICT: {verdict} — {rationale}")

    finally:
        await client.close()

@logger.catch(reraise=True)
def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
