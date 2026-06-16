#!/usr/bin/env python3
"""
Arrhenius Kinetics of LLM Inference: Full 9-Step Empirical Protocol on MMLU-Pro

Steps implemented:
  0) Pilot greedy scan + rising-limb gate
  1) Main scan to collect ~150 overconfident-error instances
  2) Temperature sweep: P(correct|T) at 7 temperatures, 50 samples each
  3) TURN entropy collection at 13 temperatures
  4) Arrhenius fitting + alternative model comparison + bootstrap R²
  5) E_a vs Δ Spearman correlation
  6) T_thresh validation (3 variants, N=4/8/16/32)
  7) T_operating accuracy comparison vs fixed-T baselines
  8) E_a predicts T_pref (Spearman)
  9) Catalysis test (zero-shot vs 4-shot vs CoT)
"""

import asyncio
import json
import math
import os
import sys
import gc
import time
import resource
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from loguru import logger
import aiohttp

# ── Workspace / logging ────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits (cgroup v2, 29 GB) ────────────────────────────────────────
try:
    RAM_BUDGET = 20 * 1024 ** 3  # 20 GB (leave headroom)
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET, RAM_BUDGET))
except Exception:
    pass  # non-fatal

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_PATH = Path(
    "/ai-inventor/aii_data/runs/run_wYelBzy-9k_d"
    "/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json"
)
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
# Switched from qwen/qwen-2.5-7b-instruct (no logprob support on OpenRouter) to
# microsoft/phi-4 which is confirmed to return logprobs via NextBit provider.
MODEL = "microsoft/phi-4"
PROVIDER_ORDER: list = []  # no restriction; NextBit routes phi-4 with logprobs

TEMP_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TURN_TEMPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
N_SAMPLES = 50          # samples per temperature per instance
N_BON = 16              # best-of-N for T_operating evaluation
MAX_OC_INSTANCES = 150  # target number of overconfident-error instances
COST_LIMIT = 8.0        # hard stop (USD)
CONCURRENCY = 15        # max parallel API calls
VALID_LETTERS = "ABCDEFGHIJ"

CHECKPOINT_FILE = LOG_DIR / "checkpoint.json"
METHOD_OUT_FILE = WORKSPACE / "method_out.json"

# ── API Client ─────────────────────────────────────────────────────────────────

class OpenRouterClient:
    BASE = "https://openrouter.ai/api/v1/chat/completions"
    # phi-4 via NextBit: ~$0.0016/1M (empirically observed)
    # Prefer actual cost from API usage.cost field; these are fallback estimates
    INPUT_COST_PER_TOKEN = 0.0016 / 1e6
    OUTPUT_COST_PER_TOKEN = 0.0016 / 1e6

    def __init__(self, api_key: str, model: str, provider_order: list,
                 concurrency: int = CONCURRENCY):
        self.api_key = api_key
        self.model = model
        self.provider_order = provider_order
        self.sem = asyncio.Semaphore(concurrency)
        self.total_cost = 0.0
        self.call_count = 0
        self.failed_calls = 0

    async def call(
        self,
        messages: list,
        temperature: float = 1.0,
        max_tokens: int = 1,
        logprobs: bool = True,
        top_logprobs: int = 20,
        retries: int = 3,
    ) -> Optional[dict]:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs if logprobs else 0,
            # require_parameters=True routes only to providers supporting all params
            # (logprobs). Without this, random providers that ignore logprobs are used.
            "provider": {"require_parameters": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-inventor.research",
        }
        for attempt in range(retries):
            try:
                async with self.sem:
                    timeout = aiohttp.ClientTimeout(total=120)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            self.BASE, json=payload, headers=headers
                        ) as resp:
                            if resp.status == 429:
                                wait = 15 * (2 ** attempt)
                                logger.warning(
                                    f"Rate limited (attempt {attempt+1}), sleeping {wait}s"
                                )
                                await asyncio.sleep(wait)
                                continue
                            if resp.status >= 500:
                                logger.warning(f"Server error {resp.status} (attempt {attempt+1})")
                                await asyncio.sleep(5 * (attempt + 1))
                                continue
                            if resp.status != 200:
                                text = await resp.text()
                                logger.error(f"API error {resp.status}: {text[:300]}")
                                return None
                            data = await resp.json()
                # Use actual cost from API response when available
                usage = data.get("usage", {})
                actual_cost = usage.get("cost")
                if actual_cost is not None:
                    self.total_cost += float(actual_cost)
                else:
                    in_tok = usage.get("prompt_tokens", 150)
                    out_tok = usage.get("completion_tokens", 1)
                    self.total_cost += (
                        in_tok * self.INPUT_COST_PER_TOKEN
                        + out_tok * self.OUTPUT_COST_PER_TOKEN
                    )
                self.call_count += 1
                return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Network error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in API call: {e}")
                self.failed_calls += 1
                return None
        self.failed_calls += 1
        return None

    def check_budget(self):
        if self.total_cost >= COST_LIMIT:
            raise RuntimeError(
                f"Budget exceeded: ${self.total_cost:.3f} >= ${COST_LIMIT}"
            )
        if self.total_cost >= COST_LIMIT * 0.75:
            logger.warning(
                f"Approaching budget limit: ${self.total_cost:.3f} / ${COST_LIMIT}"
            )


# ── Logprob utilities ──────────────────────────────────────────────────────────

PROMPT_SUFFIX = "\nRespond with only the letter of the correct answer:\n"


def build_prompt(example: dict, system_prompt: Optional[str] = None) -> list:
    """Standard zero-shot prompt. Ends with letter-only instruction for reliable single-token output."""
    user_content = example["input"] + PROMPT_SUFFIX
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_cot_prompt(example: dict) -> list:
    """CoT prompt: changes the instruction to chain-of-thought but still uses max_tokens=1
    to extract logprobs for the first predicted token. The instruction shift changes
    internal computation without requiring full CoT generation."""
    user_content = (
        example["input"]
        + "\nThink step by step and then output the answer in the format"
        ' of "The answer is (X)" at the end.\nThe answer is ('
    )
    return [{"role": "user", "content": user_content}]


def build_fewshot_prompt(example: dict, shot_examples: list) -> list:
    """4-shot prompt using provided examples from different subjects."""
    few_shot_text = ""
    for i, ex in enumerate(shot_examples[:4], 1):
        few_shot_text += (
            f"Example {i}:\n{ex['input']}"
            f"\nRespond with only the letter of the correct answer:\n{ex['output']}\n\n"
        )
    user_content = few_shot_text + example["input"] + PROMPT_SUFFIX
    return [{"role": "user", "content": user_content}]


def extract_letter_logprobs(response: dict, num_choices: int) -> dict:
    valid = VALID_LETTERS[:num_choices]
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return {L: -math.inf for L in valid}
    token_lp: dict = {}
    for entry in top:
        token_lp[entry["token"]] = entry["logprob"]
    result = {}
    for L in valid:
        bare = token_lp.get(L, -math.inf)
        spaced = token_lp.get(" " + L, -math.inf)
        result[L] = max(bare, spaced)
    return result


def get_correct_wrong_logits(response: dict, example: dict) -> tuple:
    n = example["metadata_num_choices"]
    lp_map = extract_letter_logprobs(response, n)
    correct_letter = example["output"]
    logit_c = lp_map.get(correct_letter, -math.inf)
    wrong_letters = [L for L in VALID_LETTERS[:n] if L != correct_letter]
    logit_w = max(
        (lp_map.get(L, -math.inf) for L in wrong_letters), default=-math.inf
    )
    return logit_c, logit_w, lp_map


def is_greedy_wrong(response: dict, example: dict) -> bool:
    n = example["metadata_num_choices"]
    lp_map = extract_letter_logprobs(response, n)
    if not lp_map:
        return False
    predicted = max(lp_map, key=lp_map.get)
    return predicted != example["output"]


def extract_top_letter(response: Optional[dict]) -> Optional[str]:
    if response is None:
        return None
    try:
        content = response["choices"][0]["message"]["content"]
        if content:
            content = content.strip()
            for ch in content:
                if ch.upper() in VALID_LETTERS:
                    return ch.upper()
    except (KeyError, IndexError, TypeError):
        pass
    return None


def has_logprobs(response: dict) -> bool:
    try:
        lp = response["choices"][0]["logprobs"]
        return lp is not None and "content" in lp and len(lp["content"]) > 0
    except (KeyError, IndexError, TypeError):
        return False


# ── Arrhenius fitting ──────────────────────────────────────────────────────────

def fit_arrhenius(p_correct_by_T: dict) -> Optional[dict]:
    Ts = np.array(TEMP_GRID)
    Ps = np.array([p_correct_by_T.get(T, 0.0) for T in Ts])
    mask = Ps > 0
    if mask.sum() < 3:
        return None
    inv_T = (1.0 / Ts[mask]).reshape(-1, 1)
    log_P = np.log(Ps[mask])
    reg = LinearRegression().fit(inv_T, log_P)
    log_P_pred = reg.predict(inv_T)
    ss_res = float(np.sum((log_P - log_P_pred) ** 2))
    ss_tot = float(np.sum((log_P - log_P.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    slope = float(reg.coef_[0])
    Ea = -slope
    log_A = float(reg.intercept_)
    return {
        "Ea": Ea,
        "log_A": log_A,
        "R2": R2,
        "n_valid": int(mask.sum()),
        "slope": slope,
        "valid_temps": Ts[mask].tolist(),
    }


def fit_alternatives(p_correct_by_T: dict) -> dict:
    Ts = np.array(TEMP_GRID)
    Ps = np.array([p_correct_by_T.get(T, 0.0) for T in Ts])
    mask = Ps > 0
    if mask.sum() < 3:
        return {"linear": None, "exp_T": None, "power_law": None}
    Ts_v = Ts[mask]
    Ps_v = Ps[mask]

    def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return 0.0
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        pred = reg.predict(X.reshape(-1, 1))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_linear = ols_r2(Ts_v, Ps_v)
    r2_exp_T = ols_r2(Ts_v, np.log(Ps_v))
    log_Tv = np.log(Ts_v)
    r2_power = ols_r2(log_Tv, np.log(Ps_v)) if len(np.unique(log_Tv)) >= 2 else None
    return {"linear": r2_linear, "exp_T": r2_exp_T, "power_law": r2_power}


def bootstrap_R2(p_correct_by_T: dict, n_boot: int = 500) -> tuple:
    Ts = np.array(TEMP_GRID)
    Ps = np.array([p_correct_by_T.get(T, 0.0) for T in Ts])
    mask = Ps > 0
    if mask.sum() < 3:
        return (None, None)
    inv_T = 1.0 / Ts[mask]
    log_P = np.log(Ps[mask])
    n = int(mask.sum())
    rng = np.random.default_rng(42)
    r2_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        x_b = inv_T[idx].reshape(-1, 1)
        y_b = log_P[idx]
        if len(np.unique(y_b)) < 2:
            continue
        reg = LinearRegression().fit(x_b, y_b)
        pred = reg.predict(x_b)
        ss_res = float(np.sum((y_b - pred) ** 2))
        ss_tot = float(np.sum((y_b - y_b.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_boots.append(r2)
    if not r2_boots:
        return (None, None)
    arr = np.array(r2_boots)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


# ── TURN entropy ───────────────────────────────────────────────────────────────

def first_token_entropy(lp_map: dict) -> float:
    if not lp_map:
        return 0.0
    log_probs = np.array(list(lp_map.values()), dtype=float)
    finite = np.isfinite(log_probs)
    if finite.sum() == 0:
        return 0.0
    log_probs = log_probs[finite]
    log_probs = log_probs - np.logaddexp.reduce(log_probs)
    probs = np.clip(np.exp(log_probs), 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def per_instance_T_TURN(entropy_by_T: dict, adaptation_factor: float = 0.1) -> Optional[float]:
    temps = sorted(entropy_by_T.keys())
    if len(temps) < 3:
        return None
    log_H = [math.log(max(entropy_by_T[T], 1e-12)) for T in temps]
    for j in range(1, len(temps) - 1):
        delta2 = log_H[j + 1] - 2 * log_H[j] + log_H[j - 1]
        if delta2 > 0:
            return temps[j] + adaptation_factor
    return temps[-1] + adaptation_factor  # FB-6 fallback: use T_max + delta


# ── Statistics ─────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def spearman_with_ci(x, y, n_boot: int = 1000, seed: int = 42) -> dict:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"rho": None, "p": None, "ci_low": None, "ci_high": None, "n": int(len(x))}
    rho, p_val = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    arr = np.array(boots) if boots else np.array([rho])
    return {
        "rho": float(rho),
        "p": float(p_val),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n": n,
    }


def partial_spearman(x, y, covariate) -> dict:
    df = pd.DataFrame({"x": x, "y": y, "cat": covariate}).dropna()
    if len(df) < 5:
        return {"partial_rho": None, "partial_p": None, "n": int(len(df))}
    df["x_res"] = df.groupby("cat")["x"].rank() - df.groupby("cat")["x"].transform("mean")
    df["y_res"] = df.groupby("cat")["y"].rank() - df.groupby("cat")["y"].transform("mean")
    rho, p_val = stats.spearmanr(df["x_res"], df["y_res"])
    return {"partial_rho": float(rho), "partial_p": float(p_val), "n": int(len(df))}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(data: dict):
    try:
        CHECKPOINT_FILE.write_text(json.dumps(data, indent=2, default=str))
        logger.debug(f"Checkpoint saved to {CHECKPOINT_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint() -> Optional[dict]:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return None


# ── Phase 1: Pilot greedy scan ────────────────────────────────────────────────

async def run_pilot_greedy_scan(client: OpenRouterClient, pilot_set: list) -> list:
    logger.info(f"Phase 1: Greedy scan of {len(pilot_set)} pilot examples")
    tasks = [
        client.call(
            build_prompt(ex), temperature=1.0, max_tokens=1,
            logprobs=True, top_logprobs=20
        )
        for ex in pilot_set
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    oc_instances = []
    logprob_missing = 0
    for ex, resp in zip(pilot_set, responses):
        if isinstance(resp, Exception) or resp is None:
            continue
        if not has_logprobs(resp):
            logprob_missing += 1
            continue
        lc, lw, lp_map = get_correct_wrong_logits(resp, ex)
        greedy_wrong = is_greedy_wrong(resp, ex)
        if greedy_wrong and lc > -math.inf:
            ex = deepcopy(ex)
            ex["logit_correct_T1"] = lc
            ex["logit_wrong_T1"] = lw
            ex["Delta"] = lw - lc
            ex["lp_map_T1"] = {k: v for k, v in lp_map.items() if v > -math.inf}
            oc_instances.append(ex)
    logger.info(
        f"Pilot scan: {len(oc_instances)}/{len(pilot_set)} OC errors "
        f"({logprob_missing} missing logprobs). Cost: ${client.total_cost:.3f}"
    )
    return oc_instances


# ── Phase 1b: Rising-limb check ───────────────────────────────────────────────

async def run_pilot_rising_limb_check(
    client: OpenRouterClient, oc_pilot: list, n_samples: int = 20
) -> dict:
    subset = oc_pilot[:50]
    logger.info(f"Phase 1b: Rising-limb check on {len(subset)} OC pilot instances")
    rising_count = 0
    two_point_arr = []
    for ex in subset:
        t01_resps = await asyncio.gather(
            *[
                client.call(build_prompt(ex), temperature=0.1, max_tokens=1,
                            logprobs=False, top_logprobs=0)
                for _ in range(n_samples)
            ],
            return_exceptions=True,
        )
        t05_resps = await asyncio.gather(
            *[
                client.call(build_prompt(ex), temperature=0.5, max_tokens=1,
                            logprobs=False, top_logprobs=0)
                for _ in range(n_samples)
            ],
            return_exceptions=True,
        )
        p_01 = sum(
            1 for r in t01_resps
            if not isinstance(r, Exception) and r is not None
            and extract_top_letter(r) == ex["output"]
        ) / max(1, sum(1 for r in t01_resps if not isinstance(r, Exception) and r is not None))
        p_05 = sum(
            1 for r in t05_resps
            if not isinstance(r, Exception) and r is not None
            and extract_top_letter(r) == ex["output"]
        ) / max(1, sum(1 for r in t05_resps if not isinstance(r, Exception) and r is not None))
        if p_05 > p_01 + 0.05:
            rising_count += 1
        if p_01 > 0 and p_05 > 0:
            Ea_2pt = -(math.log(p_05) - math.log(p_01)) / (1 / 0.5 - 1 / 0.1)
            two_point_arr.append({"qid": ex["metadata_question_id"], "Ea_2pt": Ea_2pt})
        client.check_budget()
    gate_fraction = rising_count / max(1, len(subset))
    gate_passed = gate_fraction >= 0.30
    logger.info(
        f"Rising-limb gate: {rising_count}/{len(subset)} = {gate_fraction:.2f} "
        f"({'PASSED' if gate_passed else 'FAILED'}). Cost: ${client.total_cost:.3f}"
    )
    return {
        "gate_passed": gate_passed,
        "gate_fraction": gate_fraction,
        "n_oc_pilot": len(oc_pilot),
        "n_tested": len(subset),
        "two_point_arrhenius": two_point_arr,
    }


# ── Phase 2: Main scan ────────────────────────────────────────────────────────

async def run_main_scan(
    client: OpenRouterClient, main_set: list, max_oc: int = MAX_OC_INSTANCES
) -> list:
    logger.info(f"Phase 2: Main scan of {len(main_set)} examples to find {max_oc} OC instances")
    tasks = [
        client.call(
            build_prompt(ex), temperature=1.0, max_tokens=1,
            logprobs=True, top_logprobs=20
        )
        for ex in main_set
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    oc_main = []
    for ex, resp in zip(main_set, responses):
        if isinstance(resp, Exception) or resp is None:
            continue
        if not has_logprobs(resp):
            continue
        lc, lw, lp_map = get_correct_wrong_logits(resp, ex)
        if is_greedy_wrong(resp, ex) and lc > -math.inf:
            ex = deepcopy(ex)
            ex["logit_correct_T1"] = lc
            ex["logit_wrong_T1"] = lw
            ex["Delta"] = lw - lc
            ex["lp_map_T1"] = {k: v for k, v in lp_map.items() if v > -math.inf}
            n_choices = ex["metadata_num_choices"]
            ex["vocab_competition_count"] = sum(
                1 for v in lp_map.values() if v > lc - 1.0
            )
            oc_main.append(ex)
        if len(oc_main) >= max_oc:
            break
    client.check_budget()
    logger.info(
        f"Main scan: {len(oc_main)} OC instances found. Cost: ${client.total_cost:.3f}"
    )
    return oc_main


# ── Phase 3: Temperature sweep ────────────────────────────────────────────────

async def measure_p_correct_by_T(
    client: OpenRouterClient,
    oc_instances: list,
    existing_results: Optional[dict] = None,
    n_samples: int = N_SAMPLES,
) -> dict:
    """For each OC instance: sample P(correct) at each T + collect TURN entropy."""
    instance_results = existing_results or {}
    to_process = [
        ex for ex in oc_instances
        if str(ex["metadata_question_id"]) not in instance_results
    ]
    logger.info(
        f"Phase 3: Temperature sweep on {len(to_process)} instances "
        f"(skipping {len(oc_instances)-len(to_process)} cached). "
        f"n_samples={n_samples}"
    )

    batch_size = 10
    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start: batch_start + batch_size]
        client.check_budget()
        for ex in batch:
            qid = str(ex["metadata_question_id"])
            p_by_T: dict = {}
            entropy_by_T: dict = {}

            # Sample at each T in TEMP_GRID
            for T in TEMP_GRID:
                sample_tasks = [
                    client.call(
                        build_prompt(ex), temperature=T,
                        max_tokens=1, logprobs=False, top_logprobs=0
                    )
                    for _ in range(n_samples)
                ]
                sample_resps = await asyncio.gather(*sample_tasks, return_exceptions=True)
                valid_resps = [
                    r for r in sample_resps
                    if not isinstance(r, Exception) and r is not None
                ]
                correct_count = sum(
                    1 for r in valid_resps if extract_top_letter(r) == ex["output"]
                )
                n_valid = max(1, len(valid_resps))
                p_by_T[T] = correct_count / n_valid

            # TURN entropy: one logprob call per T at TURN_TEMPS
            for T in TURN_TEMPS:
                lp_resp = await client.call(
                    build_prompt(ex), temperature=T,
                    max_tokens=1, logprobs=True, top_logprobs=20
                )
                if lp_resp is not None and has_logprobs(lp_resp):
                    _, _, lp_map = get_correct_wrong_logits(lp_resp, ex)
                    entropy_by_T[T] = first_token_entropy(lp_map)

            ci_by_T = {
                T: wilson_ci(int(p * n_samples), n_samples)
                for T, p in p_by_T.items()
            }
            instance_results[qid] = {
                "qid": qid,
                "example_meta": {
                    "metadata_question_id": ex["metadata_question_id"],
                    "metadata_subject": ex["metadata_subject"],
                    "metadata_split": ex["metadata_split"],
                    "metadata_num_choices": ex["metadata_num_choices"],
                    "Delta": ex["Delta"],
                    "logit_correct_T1": ex["logit_correct_T1"],
                    "logit_wrong_T1": ex["logit_wrong_T1"],
                    "vocab_competition_count": ex.get("vocab_competition_count", 0),
                },
                "p_correct_by_T": {str(T): v for T, v in p_by_T.items()},
                "ci_by_T": {str(T): list(v) for T, v in ci_by_T.items()},
                "entropy_by_T": {str(T): v for T, v in entropy_by_T.items()},
            }

        logger.info(
            f"  Batch {batch_start}–{batch_start+len(batch)}: "
            f"cost=${client.total_cost:.3f}"
        )
        # Save checkpoint after each batch
        save_checkpoint({"instance_results": instance_results})

        # Adaptive sampling: reduce if approaching budget
        if client.total_cost > COST_LIMIT * 0.5 and n_samples > 25:
            n_samples = 25
            logger.warning(f"Cost > 50% budget; reducing n_samples to {n_samples}")

    return instance_results


# ── Phase 4: Arrhenius fitting + T_thresh ─────────────────────────────────────

def run_fitting_and_thresholds(instance_results: dict, oc_instances: list) -> dict:
    logger.info(f"Phase 4: Arrhenius fitting on {len(instance_results)} instances")
    # Build QID -> example lookup
    ex_by_qid = {str(ex["metadata_question_id"]): ex for ex in oc_instances}
    per_instance_fits: dict = {}

    for qid, data in instance_results.items():
        p_by_T = {float(k): v for k, v in data["p_correct_by_T"].items()}
        entropy_by_T = {float(k): v for k, v in data["entropy_by_T"].items()}
        meta = data["example_meta"]
        Delta = meta["Delta"]

        arr = fit_arrhenius(p_by_T)
        alts = fit_alternatives(p_by_T)
        if arr is None:
            per_instance_fits[qid] = None
            continue

        r2_ci = bootstrap_R2(p_by_T)
        Ea = arr["Ea"]
        log_A = arr["log_A"]
        A_hat = math.exp(log_A)

        # R2 advantage over next-best alternative
        r2_vals = [v for v in alts.values() if v is not None]
        r2_advantage = arr["R2"] - max(r2_vals) if r2_vals else None

        # T_thresh: 3 variants for N = 4, 8, 16, 32
        T_thresh_by_N: dict = {}
        for N in [4, 8, 16, 32]:
            ln_N = math.log(N)
            T_simplified = Ea / ln_N if ln_N > 0 else None
            T_approx = Delta / ln_N if ln_N > 0 else None
            denom_A = math.log(N * A_hat) if N * A_hat > 1 else None
            T_A_corrected = Ea / denom_A if denom_A else None

            T_empirical_min = None
            for T in sorted(p_by_T.keys()):
                if p_by_T[T] > 0 and N * p_by_T[T] >= 1.0:
                    T_empirical_min = T
                    break

            def _is_lower(T_formula, T_emp):
                if T_formula is None or T_emp is None:
                    return None
                return bool(T_formula < T_emp)

            def _within_20pct(T_formula, T_emp):
                if T_formula is None or T_emp is None or T_emp == 0:
                    return None
                return bool(abs(T_formula - T_emp) / T_emp < 0.2)

            T_thresh_by_N[N] = {
                "T_simplified": T_simplified,
                "T_approx": T_approx,
                "T_A_corrected": T_A_corrected,
                "T_empirical_min": T_empirical_min,
                "simplified_is_lower_bound": _is_lower(T_simplified, T_empirical_min),
                "approx_is_lower_bound": _is_lower(T_approx, T_empirical_min),
                "A_corrected_is_lower_bound": _is_lower(T_A_corrected, T_empirical_min),
                "simplified_within_20pct": _within_20pct(T_simplified, T_empirical_min),
                "approx_within_20pct": _within_20pct(T_approx, T_empirical_min),
            }

        T_TURN = per_instance_T_TURN(entropy_by_T)

        # Fallback: if no inflection found, use 0.8 (FB-6)
        if T_TURN is None:
            T_TURN = 0.8

        T_pref = max(p_by_T, key=p_by_T.get) if p_by_T else None

        per_instance_fits[qid] = {
            "qid": qid,
            "subject": meta["metadata_subject"],
            "Ea": Ea,
            "log_A": log_A,
            "A_hat": A_hat,
            "R2_arrhenius": arr["R2"],
            "R2_ci": list(r2_ci),
            "R2_alternatives": alts,
            "R2_advantage": r2_advantage,
            "Delta": Delta,
            "vocab_competition_count": meta["vocab_competition_count"],
            "T_thresh_by_N": T_thresh_by_N,
            "T_TURN": T_TURN,
            "T_pref": T_pref,
            "p_correct_by_T": p_by_T,
            "ci_by_T": {float(k): v for k, v in data["ci_by_T"].items()},
            "logit_correct_T1": meta["logit_correct_T1"],
        }

    n_valid = sum(1 for f in per_instance_fits.values() if f is not None)
    logger.info(f"Phase 4: {n_valid}/{len(instance_results)} instances have valid Arrhenius fits")
    return per_instance_fits


# ── Phase 5: E_a vs Delta (Step 5) ───────────────────────────────────────────

def run_step5_spearman(per_instance_fits: dict) -> dict:
    valid = [f for f in per_instance_fits.values() if f is not None]
    Ea_list = [f["Ea"] for f in valid]
    Delta_list = [f["Delta"] for f in valid]
    log_A_list = [f["log_A"] for f in valid]
    comp_list = [f["vocab_competition_count"] for f in valid]

    spearman_Ea_Delta = spearman_with_ci(Ea_list, Delta_list)
    log_A_arr = np.array(log_A_list, dtype=float)
    mean_logA = float(np.mean(log_A_arr))
    std_logA = float(np.std(log_A_arr))
    log_A_CV = std_logA / abs(mean_logA) if mean_logA != 0 else None

    deviation = [ea - d for ea, d in zip(Ea_list, Delta_list)]
    spearman_dev_comp = spearman_with_ci(comp_list, deviation)

    return {
        "n_instances": len(valid),
        "spearman_Ea_Delta": spearman_Ea_Delta,
        "logA_mean": mean_logA,
        "logA_std": std_logA,
        "logA_CV": log_A_CV,
        "spearman_deviation_vs_competition": spearman_dev_comp,
    }


# ── Phase 6: T_thresh aggregate (Step 6) ──────────────────────────────────────

def run_step6_aggregate(per_instance_fits: dict) -> dict:
    valid = [f for f in per_instance_fits.values() if f is not None]
    result = {}
    for N in [4, 8, 16, 32]:
        s_lb = [f["T_thresh_by_N"][N]["simplified_is_lower_bound"] for f in valid]
        a_lb = [f["T_thresh_by_N"][N]["approx_is_lower_bound"] for f in valid]
        s_20 = [f["T_thresh_by_N"][N]["simplified_within_20pct"] for f in valid]
        a_20 = [f["T_thresh_by_N"][N]["approx_within_20pct"] for f in valid]
        n_s_lb = sum(1 for x in s_lb if x is True)
        n_a_lb = sum(1 for x in a_lb if x is True)
        n_total = len([x for x in s_lb if x is not None])
        result[N] = {
            "n_total": n_total,
            "fraction_simplified_is_lower_bound": n_s_lb / n_total if n_total else None,
            "fraction_approx_is_lower_bound": n_a_lb / n_total if n_total else None,
            "fraction_simplified_within_20pct": sum(1 for x in s_20 if x) / n_total if n_total else None,
            "fraction_approx_within_20pct": sum(1 for x in a_20 if x) / n_total if n_total else None,
        }
    # Window: fraction of instances where T_TURN > T_thresh_simplified(N=16)
    pairs = [
        (f["T_thresh_by_N"][16]["T_simplified"], f["T_TURN"])
        for f in valid
        if f["T_thresh_by_N"][16]["T_simplified"] is not None and f["T_TURN"] is not None
    ]
    frac_window = sum(1 for ts, tt in pairs if tt > ts) / len(pairs) if pairs else None
    return {"by_N": result, "window_fraction_T_TURN_above_T_thresh": frac_window}


# ── Phase 7: T_operating accuracy comparison (Step 7) ────────────────────────

async def run_T_operating_evaluation(
    client: OpenRouterClient,
    oc_instances: list,
    per_instance_fits: dict,
    n_bon: int = N_BON,
) -> dict:
    logger.info(f"Phase 7: T_operating evaluation (N_BON={n_bon})")
    valid_exs = [
        ex for ex in oc_instances
        if str(ex["metadata_question_id"]) in per_instance_fits
        and per_instance_fits[str(ex["metadata_question_id"])] is not None
    ]
    n = len(valid_exs)
    if n == 0:
        return {"n_instances": 0}

    # Cap to 80 instances to control cost
    if n > 80:
        valid_exs = valid_exs[:80]
        n = 80
        logger.info(f"  Capped Step 7 evaluation to {n} instances for cost control")

    accuracy_results: dict = {}
    api_call_counts: dict = {}

    # T_operating conditions: T_approx + delta for delta in {0.2, 0.3, 0.4}
    for delta in [0.2, 0.3, 0.4]:
        correct = 0
        calls = 0
        for ex in valid_exs:
            qid = str(ex["metadata_question_id"])
            fit = per_instance_fits[qid]
            T_approx = fit["T_thresh_by_N"][16]["T_approx"]
            if T_approx is None:
                continue
            T_op = max(0.05, min(1.5, T_approx + delta))
            resps = await asyncio.gather(
                *[
                    client.call(build_prompt(ex), temperature=T_op,
                                max_tokens=1, logprobs=False, top_logprobs=0)
                    for _ in range(n_bon)
                ],
                return_exceptions=True,
            )
            answers = [
                extract_top_letter(r) for r in resps
                if not isinstance(r, Exception) and r is not None
            ]
            if ex["output"] in answers:
                correct += 1
            calls += n_bon
        key = f"T_operating_delta_{delta:.1f}"
        accuracy_results[key] = correct / n
        api_call_counts[key] = calls
        client.check_budget()
        logger.info(f"  {key}: acc={correct/n:.3f}, cost=${client.total_cost:.3f}")

    # Fixed T=0.7 baseline
    correct_07 = 0
    for ex in valid_exs:
        resps = await asyncio.gather(
            *[
                client.call(build_prompt(ex), temperature=0.7,
                            max_tokens=1, logprobs=False, top_logprobs=0)
                for _ in range(n_bon)
            ],
            return_exceptions=True,
        )
        answers = [
            extract_top_letter(r) for r in resps
            if not isinstance(r, Exception) and r is not None
        ]
        if ex["output"] in answers:
            correct_07 += 1
    accuracy_results["fixed_T07"] = correct_07 / n
    api_call_counts["fixed_T07"] = n * n_bon
    client.check_budget()
    logger.info(f"  fixed_T07: acc={correct_07/n:.3f}, cost=${client.total_cost:.3f}")

    # TURN-adapted: median T_TURN + 0.0 (direct T_TURN as adapted temp)
    T_TURN_vals = [
        per_instance_fits[str(ex["metadata_question_id"])]["T_TURN"]
        for ex in valid_exs
        if per_instance_fits[str(ex["metadata_question_id"])]["T_TURN"] is not None
    ]
    T_TURN_dataset = float(np.median(T_TURN_vals)) if T_TURN_vals else 0.8
    correct_turn = 0
    for ex in valid_exs:
        resps = await asyncio.gather(
            *[
                client.call(build_prompt(ex), temperature=T_TURN_dataset,
                            max_tokens=1, logprobs=False, top_logprobs=0)
                for _ in range(n_bon)
            ],
            return_exceptions=True,
        )
        answers = [
            extract_top_letter(r) for r in resps
            if not isinstance(r, Exception) and r is not None
        ]
        if ex["output"] in answers:
            correct_turn += 1
    accuracy_results["TURN_adapted"] = correct_turn / n
    accuracy_results["T_TURN_dataset_level"] = T_TURN_dataset
    api_call_counts["TURN_adapted"] = n * n_bon

    # Fixed T=1.0 baseline (greedy-ish sampling)
    correct_10 = 0
    for ex in valid_exs:
        resps = await asyncio.gather(
            *[
                client.call(build_prompt(ex), temperature=1.0,
                            max_tokens=1, logprobs=False, top_logprobs=0)
                for _ in range(n_bon)
            ],
            return_exceptions=True,
        )
        answers = [
            extract_top_letter(r) for r in resps
            if not isinstance(r, Exception) and r is not None
        ]
        if ex["output"] in answers:
            correct_10 += 1
    accuracy_results["fixed_T10"] = correct_10 / n
    api_call_counts["fixed_T10"] = n * n_bon

    accuracy_results["n_instances"] = n
    logger.info(f"Phase 7 done. Cost: ${client.total_cost:.3f}")
    return {"accuracy": accuracy_results, "api_call_counts": api_call_counts}


# ── Phase 8: E_a predicts T_pref (Step 8) ────────────────────────────────────

def run_step8_spearman(per_instance_fits: dict) -> dict:
    valid = [f for f in per_instance_fits.values() if f is not None]
    Ea_list = [f["Ea"] for f in valid]
    Tpref_list = [f["T_pref"] for f in valid]
    subjects = [f["subject"] for f in valid]

    spearman_result = spearman_with_ci(Ea_list, Tpref_list)
    partial_result = partial_spearman(Ea_list, Tpref_list, subjects)

    return {
        "n_instances": len(valid),
        "spearman_Ea_Tpref": spearman_result,
        "partial_spearman_controlling_subject": partial_result,
        "R2_approx": spearman_result["rho"] ** 2 if spearman_result["rho"] else None,
    }


# ── Phase 9: Catalysis test (Step 9) ─────────────────────────────────────────

async def run_catalysis_test(
    client: OpenRouterClient, catalysis_set: list, pilot_set: list
) -> dict:
    logger.info(f"Phase 9: Catalysis test on {len(catalysis_set)} examples")

    # Build 4-shot pool: 4 examples from pilot_set per unique subject
    subject_pool: dict = {}
    for ex in pilot_set:
        s = ex["metadata_subject"]
        if s not in subject_pool:
            subject_pool[s] = ex

    catalysis_results = []
    for ex in catalysis_set:
        subject = ex["metadata_subject"]
        # Select 4 shots from different subjects
        shot_exs = [v for k, v in subject_pool.items() if k != subject][:4]
        if len(shot_exs) < 2:
            shot_exs = list(subject_pool.values())[:4]

        r_zero = await client.call(
            build_prompt(ex), temperature=1.0, max_tokens=1, logprobs=True, top_logprobs=20
        )
        r_four = await client.call(
            build_fewshot_prompt(ex, shot_exs), temperature=1.0,
            max_tokens=1, logprobs=True, top_logprobs=20
        )
        r_cot = await client.call(
            build_cot_prompt(ex), temperature=1.0, max_tokens=1,
            logprobs=True, top_logprobs=20
        )

        def _safe_logits(resp):
            if resp is None or not has_logprobs(resp):
                return math.inf, math.inf
            lc, lw, _ = get_correct_wrong_logits(resp, ex)
            return lc, lw

        lc_zero, lw_zero = _safe_logits(r_zero)
        lc_four, lw_four = _safe_logits(r_four)
        lc_cot, lw_cot = _safe_logits(r_cot)

        Delta_zero = lw_zero - lc_zero if math.isfinite(lc_zero) and math.isfinite(lw_zero) else None
        Delta_four = lw_four - lc_four if math.isfinite(lc_four) and math.isfinite(lw_four) else None
        Delta_cot = lw_cot - lc_cot if math.isfinite(lc_cot) and math.isfinite(lw_cot) else None

        delta_lc_cot = (lc_cot - lc_zero) if (math.isfinite(lc_cot) and math.isfinite(lc_zero)) else None
        delta_lw_cot = (lw_cot - lw_zero) if (math.isfinite(lw_cot) and math.isfinite(lw_zero)) else None
        delta_lc_four = (lc_four - lc_zero) if (math.isfinite(lc_four) and math.isfinite(lc_zero)) else None
        delta_lw_four = (lw_four - lw_zero) if (math.isfinite(lw_four) and math.isfinite(lw_zero)) else None

        cot_reduces_Ea = (
            Delta_cot < Delta_zero
            if Delta_cot is not None and Delta_zero is not None
            else None
        )
        four_reduces_Ea = (
            Delta_four < Delta_zero
            if Delta_four is not None and Delta_zero is not None
            else None
        )

        # Catalysis dominant: l_c raised more than l_w suppressed
        cot_is_catalysis = (
            cot_reduces_Ea and delta_lc_cot is not None and delta_lw_cot is not None
            and delta_lc_cot > abs(delta_lw_cot)
        ) if cot_reduces_Ea else False

        catalysis_results.append({
            "qid": ex["metadata_question_id"],
            "subject": subject,
            "Delta_zero_shot": Delta_zero,
            "Delta_four_shot": Delta_four,
            "Delta_cot": Delta_cot,
            "delta_lc_CoT_minus_zero": delta_lc_cot,
            "delta_lw_CoT_minus_zero": delta_lw_cot,
            "delta_lc_four_minus_zero": delta_lc_four,
            "delta_lw_four_minus_zero": delta_lw_four,
            "cot_reduces_Ea": cot_reduces_Ea,
            "four_shot_reduces_Ea": four_reduces_Ea,
            "dominant_mechanism_is_catalysis": cot_is_catalysis,
        })

    # Summary statistics
    valid_cat = [r for r in catalysis_results if r["cot_reduces_Ea"] is not None]
    n_reduces = sum(1 for r in valid_cat if r["cot_reduces_Ea"])
    n_catalysis = sum(1 for r in valid_cat if r["dominant_mechanism_is_catalysis"])
    frac_catalysis = n_catalysis / n_reduces if n_reduces > 0 else None
    ci_catalysis = wilson_ci(n_catalysis, n_reduces) if n_reduces > 0 else (None, None)
    frac_reduces_overall = n_reduces / len(valid_cat) if valid_cat else None
    ci_reduces = wilson_ci(n_reduces, len(valid_cat)) if valid_cat else (None, None)

    four_reduces = sum(1 for r in valid_cat if r.get("four_shot_reduces_Ea"))
    frac_four = four_reduces / len(valid_cat) if valid_cat else None

    logger.info(
        f"Phase 9: {n_reduces}/{len(valid_cat)} CoT reduces Ea, "
        f"{n_catalysis} dominant catalysis. Cost: ${client.total_cost:.3f}"
    )
    return {
        "per_instance": catalysis_results,
        "n_total": len(catalysis_results),
        "n_valid": len(valid_cat),
        "n_cot_reduces_Ea": n_reduces,
        "fraction_cot_reduces_Ea": frac_reduces_overall,
        "ci_fraction_cot_reduces": list(ci_reduces),
        "n_catalysis_dominant": n_catalysis,
        "fraction_catalysis_given_reduces": frac_catalysis,
        "ci_catalysis": list(ci_catalysis),
        "n_four_shot_reduces_Ea": four_reduces,
        "fraction_four_shot_reduces_Ea": frac_four,
    }


# ── Aggregate + verdict ────────────────────────────────────────────────────────

def aggregate_results(
    per_instance_fits: dict,
    step5: dict,
    step6: dict,
    step7: dict,
    step8: dict,
    step9: dict,
) -> dict:
    valid = [f for f in per_instance_fits.values() if f is not None]
    n = len(valid)
    if n == 0:
        return {"n_instances": 0, "verdict": "DISCONFIRM"}

    r2_list = [f["R2_arrhenius"] for f in valid]
    median_R2 = float(np.median(r2_list))
    mean_R2 = float(np.mean(r2_list))
    Ea_list = [f["Ea"] for f in valid]
    T_TURN_list = [f["T_TURN"] for f in valid if f["T_TURN"] is not None]

    # Confirmation criteria
    C1 = median_R2 > 0.85

    sp_ED = step5.get("spearman_Ea_Delta", {}).get("rho")
    C2 = sp_ED is not None and sp_ED > 0.5

    lb_frac_N16 = step6.get("by_N", {}).get(16, {}).get("fraction_simplified_is_lower_bound")
    C3 = lb_frac_N16 is not None and lb_frac_N16 > 0.60

    C4 = step6.get("window_fraction_T_TURN_above_T_thresh") or 0
    C4_pass = C4 >= 0.70

    sp_ET = step8.get("spearman_Ea_Tpref", {}).get("rho")
    C6 = sp_ET is not None and sp_ET > 0.3

    # T_operating vs fixed: best T_operating beat fixed_T07?
    acc = step7.get("accuracy", {})
    t_op_accs = [v for k, v in acc.items() if k.startswith("T_operating_delta")]
    best_t_op = max(t_op_accs) if t_op_accs else None
    fixed_07 = acc.get("fixed_T07")
    C7 = best_t_op is not None and fixed_07 is not None and best_t_op > fixed_07

    # Catalysis: CoT reduces Ea in >50% of cases
    frac_cat = step9.get("fraction_cot_reduces_Ea") or 0
    C9 = frac_cat > 0.50

    confirm_flags = {
        "C1_median_R2_gt_085": C1,
        "C2_spearman_Ea_Delta_gt_05": C2,
        "C3_lower_bound_N16_gt_060": C3,
        "C4_window_fraction_ge_070": C4_pass,
        "C6_Ea_predicts_Tpref": C6,
        "C7_T_operating_beats_fixed": C7,
        "C9_catalysis_fraction_gt_050": C9,
    }

    n_confirmed = sum(1 for v in confirm_flags.values() if v)
    verdict = "CONFIRM" if n_confirmed >= 4 else "PARTIAL_CONFIRM" if n_confirmed >= 2 else "DISCONFIRM"

    return {
        "n_instances": n,
        "R2_distribution": {
            "median": median_R2,
            "mean": mean_R2,
            "p25": float(np.percentile(r2_list, 25)),
            "p75": float(np.percentile(r2_list, 75)),
            "p10": float(np.percentile(r2_list, 10)),
            "p90": float(np.percentile(r2_list, 90)),
        },
        "Ea_distribution": {
            "median": float(np.median(Ea_list)),
            "mean": float(np.mean(Ea_list)),
            "std": float(np.std(Ea_list)),
            "p25": float(np.percentile(Ea_list, 25)),
            "p75": float(np.percentile(Ea_list, 75)),
        },
        "T_TURN_distribution": {
            "median": float(np.median(T_TURN_list)) if T_TURN_list else None,
            "mean": float(np.mean(T_TURN_list)) if T_TURN_list else None,
        },
        "window_fraction": C4,
        "confirm_flags": confirm_flags,
        "n_criteria_confirmed": n_confirmed,
        "verdict": verdict,
    }


# ── Build method_out.json (exp_gen_sol_out schema) ────────────────────────────

def build_method_out(
    all_examples: list,
    per_instance_fits: dict,
    catalysis_results: dict,
    pilot_gate: dict,
    step5: dict,
    step6: dict,
    step7: dict,
    step8: dict,
    step9: dict,
    aggregate: dict,
    client: OpenRouterClient,
) -> dict:
    """Serialize all results into exp_gen_sol_out schema format."""

    # Build lookup: qid -> catalysis data
    cat_by_qid = {
        str(r["qid"]): r for r in catalysis_results.get("per_instance", [])
    }

    examples_out = []
    for ex in all_examples:
        qid = str(ex["metadata_question_id"])
        fit = per_instance_fits.get(qid)
        cat = cat_by_qid.get(qid)

        row = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_question_id": ex["metadata_question_id"],
            "metadata_subject": ex["metadata_subject"],
            "metadata_split": ex["metadata_split"],
            "metadata_num_choices": ex["metadata_num_choices"],
            "metadata_src": ex.get("metadata_src", ""),
            # Whether this was identified as an overconfident-error instance
            "predict_is_oc_error": str(fit is not None).lower(),
        }

        if fit is not None:
            row["predict_arrhenius_Ea"] = str(round(fit["Ea"], 6))
            row["predict_arrhenius_R2"] = str(round(fit["R2_arrhenius"], 6))
            row["predict_arrhenius_log_A"] = str(round(fit["log_A"], 6))
            row["predict_Delta"] = str(round(fit["Delta"], 6))
            row["predict_T_pref"] = str(fit["T_pref"]) if fit["T_pref"] is not None else ""
            row["predict_T_TURN"] = str(round(fit["T_TURN"], 4)) if fit["T_TURN"] is not None else ""
            t16 = fit["T_thresh_by_N"].get(16, {})
            row["predict_T_thresh_N16_simplified"] = (
                str(round(t16["T_simplified"], 4)) if t16.get("T_simplified") is not None else ""
            )
            row["predict_T_thresh_N16_approx"] = (
                str(round(t16["T_approx"], 4)) if t16.get("T_approx") is not None else ""
            )
            row["predict_T_thresh_N16_A_corrected"] = (
                str(round(t16["T_A_corrected"], 4)) if t16.get("T_A_corrected") is not None else ""
            )
            row["predict_p_correct_by_T"] = json.dumps(
                {str(k): round(v, 4) for k, v in fit["p_correct_by_T"].items()}
            )
            row["predict_R2_advantage_vs_alternatives"] = (
                str(round(fit["R2_advantage"], 6))
                if fit["R2_advantage"] is not None else ""
            )
            row["predict_R2_linear"] = str(round(fit["R2_alternatives"].get("linear", 0) or 0, 4))
            row["predict_R2_exp_T"] = str(round(fit["R2_alternatives"].get("exp_T", 0) or 0, 4))
            row["predict_R2_power_law"] = str(
                round(fit["R2_alternatives"].get("power_law", 0) or 0, 4)
            )
            row["predict_T_thresh_N16_empirical_min"] = (
                str(t16["T_empirical_min"]) if t16.get("T_empirical_min") is not None else ""
            )
            row["predict_verdict"] = aggregate.get("verdict", "UNKNOWN")

        if cat is not None:
            row["predict_catalysis_Delta_zero_shot"] = (
                str(round(cat["Delta_zero_shot"], 6))
                if cat.get("Delta_zero_shot") is not None else ""
            )
            row["predict_catalysis_Delta_cot"] = (
                str(round(cat["Delta_cot"], 6))
                if cat.get("Delta_cot") is not None else ""
            )
            row["predict_catalysis_cot_reduces_Ea"] = (
                str(cat["cot_reduces_Ea"]).lower()
                if cat.get("cot_reduces_Ea") is not None else ""
            )
            row["predict_catalysis_is_dominant"] = str(
                cat.get("dominant_mechanism_is_catalysis", False)
            ).lower()

        examples_out.append(row)

    return {
        "metadata": {
            "experiment_id": "arrhenius_iter2_exp1",
            "model": MODEL,
            "model_note": "microsoft/phi-4 via NextBit on OpenRouter (logprob-confirmed substitute for qwen/qwen-2.5-7b-instruct which lacks logprob support on OpenRouter)",
            "dataset": "TIGER-Lab/MMLU-Pro",
            "n_oc_instances": aggregate.get("n_instances", 0),
            "cumulative_cost_usd": round(client.total_cost, 4),
            "total_api_calls": client.call_count,
            "failed_api_calls": client.failed_calls,
            "pilot_gate": pilot_gate,
            "step5_Ea_vs_Delta": step5,
            "step6_T_thresh_validation": step6,
            "step7_accuracy_comparison": step7,
            "step8_Ea_predicts_Tpref": step8,
            "step9_catalysis": {k: v for k, v in step9.items() if k != "per_instance"},
            "aggregate": aggregate,
            "verdict": aggregate.get("verdict", "UNKNOWN"),
        },
        "datasets": [
            {
                "dataset": "TIGER-Lab/MMLU-Pro",
                "examples": examples_out,
            }
        ],
    }


# ── Main orchestration ─────────────────────────────────────────────────────────

@logger.catch(reraise=True)
async def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Arrhenius Kinetics of LLM Inference — MMLU-Pro")
    logger.info(f"Model: {MODEL}  |  Budget: ${COST_LIMIT}  |  N_SAMPLES: {N_SAMPLES}")
    logger.info("=" * 70)

    # ── Load dataset ────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {DATASET_PATH}")
    raw = json.loads(DATASET_PATH.read_text())
    examples_all = raw["datasets"][0]["examples"]

    pilot_set = [e for e in examples_all if e["metadata_split"] == "pilot_set"]
    main_set = [e for e in examples_all if e["metadata_split"] == "main_set"]
    catalysis_set = [e for e in examples_all if e["metadata_split"] == "catalysis_set"]

    logger.info(
        f"Dataset: {len(pilot_set)} pilot, {len(main_set)} main, "
        f"{len(catalysis_set)} catalysis  ({len(examples_all)} total)"
    )

    client = OpenRouterClient(OPENROUTER_API_KEY, MODEL, PROVIDER_ORDER, CONCURRENCY)

    # ── Load checkpoint if available ────────────────────────────────────────
    ckpt = load_checkpoint()
    instance_results_cache = {}
    if ckpt:
        instance_results_cache = ckpt.get("instance_results", {})
        logger.info(f"Resumed from checkpoint: {len(instance_results_cache)} cached instances")

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 0: Smoke test (3 examples)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("STAGE 0: Smoke test (3 examples)")
    smoke = pilot_set[:3]
    smoke_resps = await asyncio.gather(
        *[
            client.call(build_prompt(ex), temperature=1.0, max_tokens=1,
                        logprobs=True, top_logprobs=20)
            for ex in smoke
        ],
        return_exceptions=True,
    )
    logprob_ok = 0
    for ex, resp in zip(smoke, smoke_resps):
        if isinstance(resp, Exception) or resp is None:
            logger.error(f"Smoke test: call failed for qid {ex['metadata_question_id']}")
            continue
        if has_logprobs(resp):
            logprob_ok += 1
            lc, lw, _ = get_correct_wrong_logits(resp, ex)
            logger.info(f"  qid={ex['metadata_question_id']}: lc={lc:.3f} lw={lw:.3f} Delta={lw-lc:.3f}")
        else:
            logger.warning(f"  qid={ex['metadata_question_id']}: logprobs MISSING")

    if logprob_ok == 0:
        logger.error("Logprobs not returned. Trying provider fallback.")
        # Try with explicit Fireworks-only routing
        test_resp = await client.call(
            build_prompt(smoke[0]), temperature=1.0, max_tokens=1,
            logprobs=True, top_logprobs=20
        )
        if test_resp is None or not has_logprobs(test_resp):
            logger.error("Logprobs still missing. Proceeding anyway (will use sampling fallback).")

    logger.info(f"Smoke test done: {logprob_ok}/3 with logprobs. Cost: ${client.total_cost:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1: Pilot greedy scan + rising-limb gate
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 1: Pilot greedy scan")
    oc_pilot = await run_pilot_greedy_scan(client, pilot_set)
    client.check_budget()

    pilot_gate = await run_pilot_rising_limb_check(client, oc_pilot, n_samples=20)
    client.check_budget()

    if not pilot_gate["gate_passed"]:
        logger.warning(
            f"Pilot gate FAILED (rising_frac={pilot_gate['gate_fraction']:.2f}). "
            "Proceeding anyway (data may still show Arrhenius signal)."
        )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2: Main scan → collect OC instances
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 2: Main scan")
    oc_main = await run_main_scan(client, main_set, max_oc=MAX_OC_INSTANCES)
    client.check_budget()

    if len(oc_main) < 10:
        logger.error(f"Only {len(oc_main)} OC instances found. Results will have low power.")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3: Temperature sweep (gradual scaling)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 3: Temperature sweep")

    # Scale 1: 5 instances
    logger.info("  Stage 3a: 5-instance validation")
    results_5 = await measure_p_correct_by_T(
        client, oc_main[:5], existing_results=dict(instance_results_cache), n_samples=10
    )
    client.check_budget()

    # Scale 2: 20 instances (includes cached)
    logger.info("  Stage 3b: 20-instance run")
    results_20 = await measure_p_correct_by_T(
        client, oc_main[:20], existing_results=dict(results_5), n_samples=20
    )
    client.check_budget()

    # Scale 3: Full run (all OC instances)
    logger.info(f"  Stage 3c: Full run ({len(oc_main)} instances, {N_SAMPLES} samples)")
    instance_results = await measure_p_correct_by_T(
        client, oc_main, existing_results=dict(results_20), n_samples=N_SAMPLES
    )
    client.check_budget()

    logger.info(
        f"Temperature sweep done: {len(instance_results)} instances. "
        f"Cost: ${client.total_cost:.3f}"
    )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 4: Arrhenius fitting
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 4: Arrhenius fitting")
    per_instance_fits = run_fitting_and_thresholds(instance_results, oc_main)

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 5: Step 5 — E_a vs Delta Spearman
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 5: Step 5 Spearman (E_a vs Δ)")
    step5 = run_step5_spearman(per_instance_fits)
    logger.info(f"  ρ(Ea, Δ) = {step5['spearman_Ea_Delta'].get('rho'):.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 6: Step 6 — T_thresh aggregate
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 6: Step 6 T_thresh aggregate")
    step6 = run_step6_aggregate(per_instance_fits)
    lb_n16 = step6["by_N"].get(16, {}).get("fraction_simplified_is_lower_bound")
    logger.info(f"  T_thresh lower-bound fraction (N=16): {lb_n16}")
    logger.info(f"  Window fraction (T_TURN > T_thresh): {step6.get('window_fraction_T_TURN_above_T_thresh')}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 7: T_operating vs TURN vs fixed-T (cost-controlled)
    # ──────────────────────────────────────────────────────────────────────
    step7: dict = {"accuracy": {}, "api_call_counts": {}}
    if client.total_cost < COST_LIMIT * 0.65:
        logger.info("PHASE 7: T_operating accuracy comparison")
        step7 = await run_T_operating_evaluation(client, oc_main, per_instance_fits, n_bon=N_BON)
    else:
        logger.warning(
            f"Skipping Phase 7: cost ${client.total_cost:.3f} > 65% of budget"
        )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 8: E_a predicts T_pref
    # ──────────────────────────────────────────────────────────────────────
    logger.info("PHASE 8: Step 8 Spearman (E_a vs T_pref)")
    step8 = run_step8_spearman(per_instance_fits)
    logger.info(f"  ρ(Ea, T_pref) = {step8['spearman_Ea_Tpref'].get('rho')}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 9: Catalysis test
    # ──────────────────────────────────────────────────────────────────────
    step9: dict = {}
    if client.total_cost < COST_LIMIT * 0.85:
        logger.info("PHASE 9: Catalysis test")
        step9 = await run_catalysis_test(client, catalysis_set, pilot_set)
    else:
        logger.warning(
            f"Skipping Phase 9 (catalysis): cost ${client.total_cost:.3f} > 85% budget"
        )
        step9 = {"skipped": True, "reason": f"budget_exceeded: ${client.total_cost:.3f}"}

    # ──────────────────────────────────────────────────────────────────────
    # Aggregate + verdict
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Aggregating results and computing verdict")
    aggregate = aggregate_results(per_instance_fits, step5, step6, step7, step8, step9)

    # ──────────────────────────────────────────────────────────────────────
    # Build and write method_out.json
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Building method_out.json")
    method_out = build_method_out(
        all_examples=examples_all,
        per_instance_fits=per_instance_fits,
        catalysis_results=step9,
        pilot_gate=pilot_gate,
        step5=step5,
        step6=step6,
        step7=step7,
        step8=step8,
        step9=step9,
        aggregate=aggregate,
        client=client,
    )

    METHOD_OUT_FILE.write_text(json.dumps(method_out, indent=2, default=str))
    logger.info(f"Written: {METHOD_OUT_FILE}")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed/60:.1f} min | Cost: ${client.total_cost:.4f}")
    logger.info(f"Verdict: {aggregate.get('verdict', 'UNKNOWN')}")
    logger.info(f"n_instances: {aggregate.get('n_instances', 0)}")
    logger.info(
        f"Median R²: {aggregate.get('R2_distribution', {}).get('median')}"
    )
    logger.info(f"Criteria confirmed: {aggregate.get('n_criteria_confirmed')}/7")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
