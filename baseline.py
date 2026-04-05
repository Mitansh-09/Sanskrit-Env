"""
baseline.py — SanskritEnv Baseline Inference Script (HF Inference API + ReAct + Memory)

Architecture: ReAct + Memory loop
  Think  → agent reasons using full verse context + rolling_memory of prior decisions
  Act    → agent selects one candidate_option verbatim
  Observe→ environment returns reward + feedback_message
  Update → agent appends a one-line referent summary to rolling_memory

LLM: meta-llama/Llama-3.3-70B-Instruct via HuggingFace Serverless Inference API (free tier)
Free token: https://huggingface.co/settings/tokens  (no billing required)

Usage:
    export OPENAI_API_KEY=your_hf_token_here
    export SANSKRIT_ENV_URL=http://localhost:7860    # or HF Space URL
    python baseline.py

    # Run a specific task only:
    python baseline.py --task glossary_anchoring
    python baseline.py --task sandhi_resolution
    python baseline.py --task samasa_classification
    python baseline.py --task referential_coherence

    # Swap models:
    python baseline.py --model meta-llama/Llama-3.3-70B-Instruct

Expected baseline scores (meta-llama/Llama-3.3-70B-Instruct, temp=0.0):
    Task 1 — Glossary Anchoring:      1.000
    Task 2 — Sandhi Resolution:       1.000
    Task 3 — Samasa Classification:   ~0.80 (estimated — run python baseline.py --task samasa_classification to generate)
    Task 4 — Referential Coherence:   0.840
"""

import os
import sys
import time
import json
import random
import argparse
from openai import OpenAI

from client import SanskritEnv
from models import ManuscriptAction

# ── Configuration ────────────────────────────────────────────────────────────

HF_API_KEY    = os.environ.get("OPENAI_API_KEY")   # HF token — reuse OPENAI_API_KEY for spec compliance
ENV_URL       = os.environ.get("SANSKRIT_ENV_URL", "http://localhost:7860")
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TEMPERATURE   = 0.0          # deterministic — required for reproducible scores
MAX_TOKENS    = 512
EPISODES_PER_TASK = 15
RANDOM_SEED   = 42
RETRY_WAIT    = 5            # seconds between retries on rate-limit hit

HF_BASE_URL   = "https://api-inference.huggingface.co/v1"

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


# ── HF Inference API call with retry ─────────────────────────────────────────

def call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    """
    Call HuggingFace Serverless Inference API with exponential backoff on rate-limit errors.
    Free tier: uses HF token via OpenAI-compatible endpoint at api-inference.huggingface.co/v1.
    """
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            err_str = str(exc).lower()
            # Handle HF rate limit (429) and similar transient errors
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"    [rate limit] waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
            else:
                # Non-rate-limit error — re-raise immediately
                raise
    # Final fallback after all retries exhausted
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

def run_episode(env, client: OpenAI, model: str, task_id: str, seed: int, verbose: bool = True) -> float:
    """
    Run one complete episode using the ReAct + Memory architecture.

    Loop:
        while not done:
            user_prompt = build_user_prompt(obs, rolling_memory)
            raw_answer  = call_llm(client, model, system, user_prompt)
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
        raw_answer  = call_llm(client, model, SYSTEM_PROMPT, user_prompt)
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

def run_task(task_id: str, label: str, client: OpenAI, model: str) -> dict:
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
                score = run_episode(env, client, model, task_id, seed, verbose=True)
                scores.append(score)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                scores.append(0.0)

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
    parser = argparse.ArgumentParser(description="SanskritEnv baseline inference (HF Inference API + ReAct + Memory)")
    parser.add_argument(
        "--task",
        choices=["glossary_anchoring", "sandhi_resolution", "samasa_classification", "referential_coherence", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model ID to use (default: meta-llama/Llama-3.3-70B-Instruct)",
    )
    args = parser.parse_args()

    if not HF_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  Get a free HF token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Instantiate OpenAI client pointed at HF Serverless Inference API
    hf_client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=HF_API_KEY,
    )

    print(f"\nSanskritEnv Baseline — HF Inference API + ReAct + Memory")
    print(f"Environment: {ENV_URL}")
    print(f"Model:       {args.model}")
    print(f"Endpoint:    {HF_BASE_URL}")
    print(f"Architecture: ReAct + rolling_memory (Think→Act→Observe→Update)")

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

    if args.task != "all":
        tasks_to_run = {args.task: tasks_to_run[args.task]}

    results = []
    for task_id, label in tasks_to_run.items():
        results.append(run_task(task_id, label, hf_client, args.model))

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
