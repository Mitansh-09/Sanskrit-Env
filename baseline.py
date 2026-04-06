"""
baseline.py — SanskritEnv Baseline Inference Script (Cloudflare Workers AI + ReAct + Memory)

Architecture: ReAct + Memory loop
    Think   -> agent reasons using full verse context + rolling_memory of prior decisions
    Act     -> agent selects one candidate_option verbatim
    Observe -> environment returns reward + feedback_message
    Update  -> agent appends a one-line referent summary to rolling_memory

LLM backend: Cloudflare Workers AI (OpenAI-compatible chat endpoint)

Usage:
        # Option 1: set env vars directly
        set CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
        set CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
        set SANSKRIT_ENV_URL=http://localhost:7860

        # Option 2: place values in .env (loaded automatically)
        # CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
        # CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
        # SANSKRIT_ENV_URL=http://localhost:7860

        python baseline.py
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from dotenv import load_dotenv

from client import SanskritEnv
from models import ManuscriptAction

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _first_nonempty_env(*names: str) -> tuple[str, Optional[str]]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip(), name
    return "", None


ENV_URL = os.environ.get("SANSKRIT_ENV_URL", "http://localhost:7860")
CLOUDFLARE_API_TOKEN, CLOUDFLARE_API_TOKEN_SOURCE = _first_nonempty_env(
    "CLOUDFLARE_API_TOKEN",
    "CF_API_TOKEN",
    "CLOUDFLARE_TOKEN",
    "CF_TOKEN",
)
CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCOUNT_ID_SOURCE = _first_nonempty_env(
    "CLOUDFLARE_ACCOUNT_ID",
    "CF_ACCOUNT_ID",
)

DEFAULT_MODEL = os.environ.get("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

_default_cf_chat_url = ""
if CLOUDFLARE_ACCOUNT_ID:
    _default_cf_chat_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
    )

CF_CHAT_COMPLETIONS_URL = (
    os.environ.get("CF_CHAT_COMPLETIONS_URL")
    or os.environ.get("CF_WORKERS_AI_URL")
    or _default_cf_chat_url
)

CF_MODEL_CANDIDATES = [
    model.strip()
    for model in os.environ.get(
        "CF_MODEL_CANDIDATES",
        DEFAULT_MODEL,
    ).split(",")
    if model.strip()
]
AUTO_MODEL_FALLBACK = _env_bool("CF_AUTO_MODEL_FALLBACK", True)

TEMPERATURE = _env_float("CF_TEMPERATURE", 0.0)
MAX_TOKENS = _env_int("CF_MAX_TOKENS", 512)
EPISODES_PER_TASK = _env_int("EPISODES_PER_TASK", 15)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
RETRY_WAIT = _env_int("RETRY_WAIT", 5)
REQUEST_TIMEOUT = _env_int("CF_REQUEST_TIMEOUT", 90)

BASELINE_TASK = ((os.environ.get("BASELINE_TASK", "all") or "all").strip().lower() or "all")
BASELINE_MODEL = ((os.environ.get("BASELINE_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL).strip() or DEFAULT_MODEL)
BASELINE_NO_AUTO_FALLBACK = _env_bool("BASELINE_NO_AUTO_FALLBACK", False)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Sanskrit manuscript interpreter with deep knowledge of:
- Classical Sanskrit grammar, phonology, and sandhi rules
- Paninian grammar: samasa (compound) classification — Tatpurusha, Karmadharaya, Dvigu, Dvandva, Bahuvrihi, Avyayibhava
- Ayurvedic texts: Charaka Samhita, Sushruta Samhita, Ashtanga Hridayam
- Astronomical texts: Aryabhatiya, Brahmasphutasiddhanta
- Philosophical texts: Bhagavad Gita, Upanishads, Vivekachudamani
- Narrative texts: Ramayana, Mahabharata, Arthashastra

For samasa classification, apply these rules:
- Tatpurusha: second member is the semantic head, first member qualifies it via a case relation
- Karmadharaya: adjective + noun referring to the SAME entity (subtype of tatpurusha)
- Dvigu: numeral + noun forming a collective (subtype of tatpurusha)
- Dvandva: both members are coordinate and equal — "A and B" — dual/plural number reflects sum
- Bahuvrihi: compound is adjectival, describes an EXTERNAL referent — neither member is the head
- Avyayibhava: whole compound is an indeclinable adverb, first member is usually a prefix/indeclinable

Your task each step:
1. THINK: reason carefully about the Sanskrit passage and question
2. SELECT: choose EXACTLY ONE option from the list provided
3. OUTPUT: respond with ONLY the exact text of your chosen option — nothing else

Rules:
- Your entire response must be one of the provided options, copied character-for-character
- Do not add explanation, punctuation, or any other text
- If unsure, pick the option that best fits the domain and grammatical context"""

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_user_prompt(obs, rolling_memory: str) -> str:
    """
    Build the full prompt for one ReAct step.

    Injects rolling_memory so the agent has access to all prior decisions
    established in this episode — critical for Task 4 coherence tracking.
    """
    lines = []

    # Source text
    if obs.source_text_iast:
        lines.append(f"Sanskrit (IAST): {obs.source_text_iast}")
    if obs.source_text_devanagari:
        lines.append(f"Devanagari:      {obs.source_text_devanagari}")
    if obs.english_context:
        lines.append(f"Source context:  {obs.english_context}")
    if obs.domain:
        lines.append(f"Domain:          {obs.domain}")

    # Task-specific fields
    if obs.target_term_iast:
        lines.append(f"Term to interpret: {obs.target_term_iast}")
    if obs.compound_iast:
        label = "Compound to classify" if obs.task_id == "samasa_classification" else "Compound to split"
        lines.append(f"{label}: {obs.compound_iast}")

    # Task 4: full verse history
    if obs.verses_so_far:
        lines.append("")
        lines.append("Verses in this passage:")
        for v in obs.verses_so_far:
            lines.append(f"  [{v['verse_num']}] IAST:    {v['iast']}")
            lines.append(f"       English: {v['english']}")

    # ReAct Memory — inject everything established so far in this episode
    if rolling_memory.strip():
        lines.append("")
        lines.append("── What you have established so far in this episode ──")
        lines.append(rolling_memory.strip())
        lines.append("── Use this to stay consistent ──")

    # Reward signal from last step (helps agent self-correct)
    if obs.step_reward and obs.step_reward > 0:
        lines.append("")
        lines.append(f"Your last answer was CORRECT (reward: {obs.step_reward:.2f}).")
    elif obs.step_reward == 0.0 and obs.feedback_message:
        lines.append("")
        lines.append(f"Feedback: {obs.feedback_message}")

    # The decision
    lines.append("")
    lines.append(f"Question: {obs.decision_prompt}")
    lines.append("")
    lines.append("Options (choose one exactly as written):")
    for i, opt in enumerate(obs.candidate_options):
        lines.append(f"  {i + 1}. {opt}")
    lines.append("")
    lines.append("Your answer (exact option text only):")

    return "\n".join(lines)


def update_rolling_memory(rolling_memory: str, obs, selected_option: str) -> str:
    """
    Append a one-line summary of the just-completed decision to rolling_memory.

    This is the 'Update' phase of the ReAct + Memory loop.
    For Task 4 specifically, this records which character/entity each
    pronoun or implicit subject refers to, so future steps can stay consistent.
    """
    if not obs.decision_prompt:
        return rolling_memory

    # Build a concise one-liner
    summary = f"• {obs.decision_prompt.strip().rstrip('?')} → {selected_option}"

    # Cap at 10 lines to prevent prompt bloat
    lines = [l for l in rolling_memory.strip().split("\n") if l.strip()]
    lines.append(summary)
    if len(lines) > 10:
        lines = lines[-10:]

    return "\n".join(lines)


# ── Cloudflare Workers AI API call with retry ────────────────────────────────


def _extract_chat_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        # Some models return content as an array of chunks.
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "".join(chunks).strip()

        return str(content).strip()

    # Cloudflare native envelope (ai/run) often returns result.response
    result = payload.get("result")
    if isinstance(result, dict):
        response_text = result.get("response")
        if isinstance(response_text, str):
            return response_text.strip()

    return ""


def _is_openai_chat_endpoint(url: str) -> bool:
    return "/ai/v1/chat/completions" in (url or "")


def _build_worker_url(base_url: str, model: str) -> str:
    url = (base_url or "").strip()
    if not url:
        return url

    encoded_model = urllib.parse.quote(model, safe="@/-._")
    if "{model}" in url:
        return url.replace("{model}", encoded_model)

    # If user supplies /ai/run without the model suffix, append it.
    if "/ai/run" in url and url.rstrip("/").endswith("/ai/run"):
        return f"{url.rstrip('/')}/{encoded_model}"

    return url


def _build_cf_payload(model: str, system: str, user: str, max_tokens: int) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if _is_openai_chat_endpoint(CF_CHAT_COMPLETIONS_URL):
        payload["model"] = model
    return payload


def _parse_api_error_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "unknown provider error"

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            # OpenAI-style error envelope
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            if isinstance(err, str) and err.strip():
                return err.strip()

            # Cloudflare API envelope
            errs = payload.get("errors")
            if isinstance(errs, list):
                messages = []
                for item in errs:
                    if isinstance(item, dict):
                        msg = item.get("message")
                        if isinstance(msg, str) and msg.strip():
                            messages.append(msg.strip())
                if messages:
                    return "; ".join(messages)
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    title_start = lowered.find("<title>")
    if title_start != -1:
        title_end = lowered.find("</title>", title_start)
        if title_end != -1:
            title = text[title_start + len("<title>"):title_end].strip()
            if title:
                return title

    return " ".join(text.split())[:220]


def _is_quota_exhausted(message: str) -> bool:
    msg = (message or "").lower()
    checks = (
        "insufficient credits",
        "depleted your monthly included credits",
        "purchase pre-paid credits",
        "quota",
        "billing",
    )
    return any(token in msg for token in checks)


def _cf_headers() -> dict:
    return {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }


def _probe_model_access(model: str) -> tuple[bool, str]:
    payload = _build_cf_payload(
        model=model,
        system="Reply with OK only.",
        user="OK",
        max_tokens=4,
    )

    request_body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _build_worker_url(CF_CHAT_COMPLETIONS_URL, model),
        data=request_body,
        headers=_cf_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=min(REQUEST_TIMEOUT, 30)) as response:
            if 200 <= response.status < 300:
                return True, "ok"
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        parsed = _parse_api_error_text(error_body)
        return False, f"{exc.code}: {parsed}"
    except urllib.error.URLError as exc:
        return False, f"network: {exc.reason}"
    except Exception as exc:
        return False, f"error: {exc}"


def select_model_for_run(requested_model: str) -> str:
    ordered: List[str] = [requested_model]
    ordered.extend(m for m in CF_MODEL_CANDIDATES if m != requested_model)

    print("Checking Cloudflare Workers AI model availability...")
    unavailable = []

    for model in ordered:
        ok, detail = _probe_model_access(model)
        if ok:
            if model == requested_model:
                print(f"  Using requested model: {model}")
            else:
                print(f"  Requested model unavailable; auto-fallback to: {model}")
            return model

        unavailable.append((model, detail))
        print(f"  Unavailable: {model} ({detail})")

        if _is_quota_exhausted(detail):
            raise RuntimeError(
                f"Cloudflare Workers AI quota appears exhausted for this token/account. {detail}"
            )

    summary = "; ".join(f"{model} -> {reason}" for model, reason in unavailable[:4])
    raise RuntimeError(
        "No available Cloudflare Workers AI chat model found for this token/account setup. "
        f"Tried: {summary}"
    )


def call_llm(model: str, system: str, user: str) -> str:
    """
    Call Cloudflare Workers AI OpenAI-compatible chat completions endpoint.
    """
    payload = _build_cf_payload(
        model=model,
        system=system,
        user=user,
        max_tokens=MAX_TOKENS,
    )

    request_body = json.dumps(payload).encode("utf-8")

    for attempt in range(4):
        req = urllib.request.Request(
            _build_worker_url(CF_CHAT_COMPLETIONS_URL, model),
            data=request_body,
            headers=_cf_headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                return _extract_chat_text(data)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            if exc.code in (429, 500, 502, 503, 504):
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"    [cloudflare {exc.code}] waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
                continue
            parsed = _parse_api_error_text(error_body)
            raise RuntimeError(f"Workers AI request failed ({exc.code}): {parsed}")
        except urllib.error.URLError as exc:
            wait = RETRY_WAIT * (2 ** attempt)
            print(f"    [network] {exc.reason}; waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Workers AI returned non-JSON response: {exc}")

    return ""


# ── Option matching ───────────────────────────────────────────────────────────

def match_to_option(raw_answer: str, candidate_options: list) -> str:
    """
    Match the LLM's raw output to the closest candidate_option.

    Priority:
    1. Exact match
    2. Candidate starts with the raw answer (model truncated)
    3. Raw answer starts with the candidate (model added padding)
    4. Random fallback (prevents crash, penalised by grader)
    """
    raw = raw_answer.strip()

    # 1. Exact
    for opt in candidate_options:
        if raw == opt:
            return opt

    # 2. Prefix: model gave first N chars of option
    for opt in candidate_options:
        if opt.lower().startswith(raw.lower()[:30]):
            return opt

    # 3. Contains: raw answer contains the option
    for opt in candidate_options:
        if opt.lower() in raw.lower():
            return opt

    # 4. Random fallback
    print(f"    [warn] could not match '{raw[:60]}' to any option — random fallback")
    return random.choice(candidate_options)


# ── Episode runner (ReAct + Memory loop) ─────────────────────────────────────

def run_episode(env, model: str, task_id: str, seed: int, verbose: bool = True) -> float:
    """
    Run one complete episode using the ReAct + Memory architecture.

    Loop:
        while not done:
            user_prompt = build_user_prompt(obs, rolling_memory)
            raw_answer  = call_llm(model, system, user_prompt)
            selected    = match_to_option(raw_answer, obs.candidate_options)
            result      = env.step(ManuscriptAction(selected_option=selected, reasoning=raw_answer))
            rolling_memory = update_rolling_memory(rolling_memory, obs, selected)
            obs = result.observation

    Returns the final episode score (0.0–1.0).
    """
    result = env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    rolling_memory = ""   # starts empty every episode
    step = 0

    while not obs.done:
        step += 1
        user_prompt = build_user_prompt(obs, rolling_memory)
        raw_answer  = call_llm(model, SYSTEM_PROMPT, user_prompt)
        selected    = match_to_option(raw_answer, obs.candidate_options)

        if verbose:
            print(f"    Step {step}: selected → '{selected[:60]}'")

        # Update memory BEFORE stepping (so the summary is of current decision)
        rolling_memory = update_rolling_memory(rolling_memory, obs, selected)

        result = env.step(ManuscriptAction(
            selected_option=selected,
            confidence=0.8,
            reasoning=raw_answer,
        ))
        obs = result.observation

        if obs.step_reward is not None and verbose:
            print(f"            reward: {obs.step_reward:.2f} | cumulative: {obs.cumulative_score:.2f}")

    final_score = obs.reward if obs.reward is not None else 0.0
    if verbose:
        print(f"    Episode done — final score: {final_score:.4f}")

    return final_score


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, label: str, model: str) -> dict:
    # Task 1 — glossary_anchoring  (Easy)
    # Task 2 — sandhi_resolution   (Medium)
    # Task 3 — samasa_classification (Medium)
    # Task 4 — referential_coherence (Hard)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Model: {model} | Episodes: {EPISODES_PER_TASK} | Seed base: {RANDOM_SEED}")
    print(f"{'='*65}")

    scores = []
    with SanskritEnv(base_url=ENV_URL).sync() as env:
        for i in range(EPISODES_PER_TASK):
            seed = RANDOM_SEED + i
            print(f"\n  Episode {i + 1}/{EPISODES_PER_TASK} (seed={seed})")
            try:
                score = run_episode(env, model, task_id, seed, verbose=True)
                scores.append(score)
            except Exception as exc:
                message = str(exc)
                print(f"  ERROR: {message}")
                scores.append(0.0)
                if _is_quota_exhausted(message):
                    remaining = EPISODES_PER_TASK - len(scores)
                    if remaining > 0:
                        scores.extend([0.0] * remaining)
                    print("  Stopping task early: Workers AI quota appears exhausted.")
                    break

    mean   = sum(scores) / len(scores)
    stddev = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    print(f"\n  Scores:  {[round(s, 3) for s in scores]}")
    print(f"  Mean:    {mean:.4f}")
    print(f"  Std dev: {stddev:.4f}")

    return {
        "task_id":  task_id,
        "label":    label,
        "model":    model,
        "episodes": EPISODES_PER_TASK,
        "seed":     RANDOM_SEED,
        "scores":   scores,
        "mean":     round(mean, 4),
        "stddev":   round(stddev, 4),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task_choice = BASELINE_TASK
    model_choice = BASELINE_MODEL
    no_auto_fallback = BASELINE_NO_AUTO_FALLBACK

    valid_tasks = {
        "glossary_anchoring",
        "sandhi_resolution",
        "samasa_classification",
        "referential_coherence",
        "all",
    }
    if task_choice not in valid_tasks:
        print(
            "ERROR: BASELINE_TASK must be one of: "
            "glossary_anchoring, sandhi_resolution, samasa_classification, referential_coherence, all"
        )
        print(f"Current BASELINE_TASK: {task_choice}")
        sys.exit(1)

    if not CLOUDFLARE_API_TOKEN:
        print("ERROR: CLOUDFLARE_API_TOKEN (or CF_API_TOKEN) is not set.")
        print("  Create an API token with Workers AI access in Cloudflare dashboard.")
        sys.exit(1)

    if not CF_CHAT_COMPLETIONS_URL:
        print("ERROR: Workers AI URL could not be resolved.")
        print("  Set CLOUDFLARE_ACCOUNT_ID (or CF_ACCOUNT_ID), or set CF_CHAT_COMPLETIONS_URL directly.")
        sys.exit(1)

    effective_model = model_choice
    auto_fallback_enabled = AUTO_MODEL_FALLBACK and not no_auto_fallback
    if auto_fallback_enabled:
        try:
            effective_model = select_model_for_run(model_choice)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            print("Hint: set BASELINE_MODEL to a supported model, e.g. @cf/meta/llama-3.1-8b-instruct")
            sys.exit(1)
    elif no_auto_fallback:
        print("Model fallback disabled via BASELINE_NO_AUTO_FALLBACK=1")
    else:
        print("Model fallback disabled via CF_AUTO_MODEL_FALLBACK=0")

    print(f"\nSanskritEnv Baseline — Cloudflare Workers AI + ReAct + Memory")
    print(f"Environment: {ENV_URL}")
    print(f"Model:       {effective_model}")
    print(f"Task scope:  {task_choice}")
    print(f"Episodes:    {EPISODES_PER_TASK}")
    print(f"Seed base:   {RANDOM_SEED}")
    print(f"Workers URL: {CF_CHAT_COMPLETIONS_URL}")
    if CLOUDFLARE_API_TOKEN_SOURCE:
        print(f"Token source: {CLOUDFLARE_API_TOKEN_SOURCE}")
    if CLOUDFLARE_ACCOUNT_ID_SOURCE:
        print(f"Account source: {CLOUDFLARE_ACCOUNT_ID_SOURCE}")
    print(f"Architecture: ReAct + rolling_memory (Think->Act->Observe->Update)")

    # Canonical task ordering:
    #   Task 1 — glossary_anchoring       (Easy)
    #   Task 2 — sandhi_resolution        (Medium)
    #   Task 3 — samasa_classification    (Medium)
    #   Task 4 — referential_coherence    (Hard)
    tasks_to_run = {
        "glossary_anchoring":    "Task 1 — Glossary Anchoring (Easy)",
        "sandhi_resolution":     "Task 2 — Sandhi Resolution (Medium)",
        "samasa_classification": "Task 3 — Samasa Classification (Medium)",
        "referential_coherence": "Task 4 — Referential Coherence (Hard)",
    }

    if task_choice != "all":
        tasks_to_run = {task_choice: tasks_to_run[task_choice]}

    results = []
    for task_id, label in tasks_to_run.items():
        results.append(run_task(task_id, label, effective_model))

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL BASELINE RESULTS")
    print(f"{'='*65}")
    for r in results:
        bar = "█" * int(r["mean"] * 20)
        print(f"  {r['label']}")
        print(f"    {bar:<20} {r['mean']:.4f} ± {r['stddev']:.4f}")
    print(f"{'='*65}\n")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    print("Copy the mean scores into README.md baseline table.")
