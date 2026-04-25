"""
SanskritEnv Baseline Runner.

Runs episodes against the local or HF Space environment using LLM inference.
Outputs results to baseline_results.json and prints a summary table.

Usage:
    python baseline.py
    python baseline.py --task referential_coherence
    python baseline.py --task manuscript_restoration --difficulty hard
    python baseline.py --model "@cf/meta/llama-3.1-70b-instruct"
    python baseline.py --episodes 20
"""

import argparse
import json
import os
import sys
import time
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

TASK_IDS = [
    "glossary_anchoring",
    "sandhi_resolution",
    "samasa_classification",
    "referential_coherence",
    "manuscript_restoration",
    "full_manuscript_session",
]

DEFAULT_MODEL = os.environ.get(
    "BASELINE_MODEL", "@cf/meta/llama-3.2-3b-instruct"
)
DEFAULT_TASK = os.environ.get("BASELINE_TASK", "all")
DEFAULT_EPISODES = int(os.environ.get("EPISODES_PER_TASK", "5"))
DEFAULT_SEED = int(os.environ.get("RANDOM_SEED", "42"))

CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ROUTER_URL = os.environ.get(
    "HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions"
)

ENV_TARGET = os.environ.get("SANSKRIT_ENV_TARGET", "local")
HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL", "https://adityahars-sanskrit-env.hf.space"
)
ENV_URL = os.environ.get("SANSKRIT_ENV_URL", "http://localhost:7860")

RESULTS_FILE = "baseline_results.json"

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Sanskrit philologist with knowledge of:
- Classical Sanskrit grammar, phonology, Paninian sandhi rules
- Samasa classification (Tatpurusha, Karmadharaya, Dvigu, Dvandva, Bahuvrihi, Avyayibhava)
- Vedic and classical Sanskrit literature across Ayurveda, astronomy, philosophy, narrative
- Textual criticism methodology

For standard questions (tasks 1-4): reply with EXACTLY one candidate option, verbatim.

For manuscript restoration (task 5):
  If you need more evidence: reply in format:
    TOOL: <tool_name>
    INPUT: <tool_input>

  If you are ready to commit: reply in format:
    COMMIT: <your final interpretation (must match one candidate option exactly)>

Available tools: lexicon_lookup, sandhi_parser, meter_checker, commentary_fetch, witness_compare, referent_tracker

Rules:
- Never repeat a tool call with identical input
- Only commit when you have consulted at least 2 relevant tools
- For wrong answers: 0.0 reward regardless of evidence. Answer correctly."""


# ── LLM Call Functions ───────────────────────────────────────────────────────

def call_cloudflare(messages, model, max_retries=3):
    """Call Cloudflare Workers AI."""
    if not CF_API_TOKEN or not CF_ACCOUNT_ID:
        raise RuntimeError("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID required")
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Cloudflare call failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Cloudflare call failed after retries")


def call_huggingface(messages, model, max_retries=4):
    """Call HuggingFace Router."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required for HuggingFace provider")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_ROUTER_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"HF call failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("HF call failed after retries")


def call_llm(messages, model):
    """Route to the appropriate LLM provider."""
    if CF_API_TOKEN and CF_ACCOUNT_ID:
        return call_cloudflare(messages, model)
    elif HF_TOKEN:
        return call_huggingface(messages, model)
    else:
        raise RuntimeError("No LLM provider configured. Set CLOUDFLARE_API_TOKEN or HF_TOKEN.")


# ── Environment Client ──────────────────────────────────────────────────────

def get_env_url():
    """Determine environment URL."""
    if ENV_TARGET == "space":
        return HF_SPACE_URL
    return ENV_URL


def env_reset(base_url, task_id, seed=None, episode_id=None, difficulty=None):
    """Call /reset on the environment."""
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    if episode_id:
        payload["episode_id"] = episode_id
    if difficulty:
        payload["difficulty"] = difficulty
    resp = requests.post(f"{base_url}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "observation" in data:
        obs = data.get("observation") or {}
        if isinstance(obs, dict):
            obs.setdefault("reward", data.get("reward"))
            obs.setdefault("done", data.get("done", False))
            return obs
    return data


def env_step(base_url, action):
    """Call /step on the environment."""
    resp = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
    if resp.status_code >= 400:
        print(f"Server error for {{'action': action}} {resp.status_code}: {resp.text}")
    
    if resp.status_code == 422:
        resp2 = requests.post(f"{base_url}/step", json=action, timeout=30)
        if resp2.status_code >= 400:
            print(f"Server error for raw action {resp2.status_code}: {resp2.text}")
        resp = resp2

    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "observation" in data:
        obs = data.get("observation") or {}
        if isinstance(obs, dict):
            obs.setdefault("reward", data.get("reward"))
            obs.setdefault("done", data.get("done", False))
            return obs
    return data


# ── Response Parsing ─────────────────────────────────────────────────────────

def match_to_option(text, candidate_options):
    """Match response text to the closest candidate option."""
    text_clean = text.strip()
    # Exact match
    for opt in candidate_options:
        if text_clean == opt:
            return opt
    # Prefix match
    for opt in candidate_options:
        if opt.startswith(text_clean) or text_clean.startswith(opt):
            return opt
    # Substring match
    for opt in candidate_options:
        if text_clean.lower() in opt.lower() or opt.lower() in text_clean.lower():
            return opt
    return candidate_options[0] if candidate_options else text_clean


def parse_restoration_response(raw_text, candidate_options):
    """Parse a Task 5 response into action dict."""
    raw = raw_text.strip()
    if raw.startswith("TOOL:"):
        lines = raw.split("\n")
        tool_name = lines[0].replace("TOOL:", "").strip()
        tool_input = lines[1].replace("INPUT:", "").strip() if len(lines) > 1 else ""
        return {"action_type": "tool_call", "tool_name": tool_name, "tool_input": tool_input}
    elif raw.startswith("COMMIT:"):
        answer = raw.replace("COMMIT:", "").strip()
        matched = match_to_option(answer, candidate_options)
        return {"action_type": "commit", "final_answer": matched}
    else:
        matched = match_to_option(raw, candidate_options)
        return {"action_type": "commit", "final_answer": matched}


# ── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(base_url, task_id, model, seed, difficulty=None, verbose=False):
    """Run a single episode and return the result."""
    obs = env_reset(base_url, task_id, seed=seed, difficulty=difficulty)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_trace = []
    total_steps = 0
    tools_used = []

    while not obs.get("done", False):
        # Build user message
        candidates = obs.get("candidate_options", [])
        user_msg = f"Task: {task_id}\nPassage: {obs.get('source_text_iast', '')}\n"
        user_msg += f"Question: {obs.get('decision_prompt', '')}\n"
        user_msg += f"Options:\n" + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(candidates))

        if obs.get("tool_call_history"):
            user_msg += f"\n\nTool history:\n"
            for tc in obs["tool_call_history"]:
                user_msg += f"  - {tc.get('tool', '')}: {json.dumps(tc.get('output', {}))[:200]}\n"

        if obs.get("steps_remaining") is not None:
            user_msg += f"\nSteps remaining: {obs['steps_remaining']}"

        messages.append({"role": "user", "content": user_msg})

        try:
            response = call_llm(messages, model)
        except RuntimeError as e:
            print(f"  LLM error: {e}")
            break

        messages.append({"role": "assistant", "content": response})

        # Parse and step
        if task_id in ["manuscript_restoration", "full_manuscript_session"]:
            parsed = parse_restoration_response(response, candidates)
            action = {
                "action_type": parsed["action_type"],
                "selected_option": parsed.get("final_answer", ""),
                "tool_name": parsed.get("tool_name"),
                "tool_input": parsed.get("tool_input"),
                "final_answer": parsed.get("final_answer"),
                "reasoning": response[:200],
            }
            if parsed["action_type"] == "tool_call":
                tools_used.append(parsed["tool_name"])
        else:
            selected = match_to_option(response, candidates)
            action = {"selected_option": selected, "reasoning": response[:200]}

        if verbose:
            print(f"  Step {total_steps}: {json.dumps(action)[:120]}")

        obs = env_step(base_url, action)
        total_steps += 1
        step_trace.append({"step": total_steps, "action": action, "reward": obs.get("step_reward")})

        # Safety: max steps
        if total_steps > 20:
            break

    return {
        "task_id": task_id,
        "episode_id": obs.get("episode_id", ""),
        "score": obs.get("reward", 0.0) or 0.0,
        "steps": total_steps,
        "tools_used": tools_used,
        "step_trace": step_trace,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SanskritEnv Baseline Runner")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_url = get_env_url()
    tasks = TASK_IDS if args.task == "all" else [args.task]

    print(f"Running baseline: {args.model}")
    print(f"Environment: {base_url}")
    print("-" * 65)

    all_results = []
    summary = {}

    for task_id in tasks:
        scores = []
        for ep_num in range(args.episodes):
            seed = args.seed + ep_num
            try:
                result = run_episode(
                    base_url, task_id, args.model, seed,
                    difficulty=args.difficulty, verbose=args.verbose,
                )
                scores.append(result["score"])
                all_results.append(result)
                status = "ok" if result["score"] > 0 else "X"
                print(f"  [{task_id}] ep {ep_num+1}/{args.episodes}: {result['score']:.3f} {status}")
            except Exception as e:
                print(f"  [{task_id}] ep {ep_num+1}/{args.episodes}: ERROR - {e}")
                scores.append(0.0)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0.0
        summary[task_id] = {"mean": round(mean_score, 3), "std": round(std_score, 3), "episodes": len(scores)}
        print(f"  [{task_id}] Mean: {mean_score:.3f} +/- {std_score:.3f}")
        print()

    # Overall
    all_scores = [r["score"] for r in all_results]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall_std = (sum((s - overall_mean) ** 2 for s in all_scores) / len(all_scores)) ** 0.5 if all_scores else 0.0

    print("-" * 65)
    print(f"OVERALL: {overall_mean:.3f} +/- {overall_std:.3f}")

    # Save results
    output = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "summary": {
            "overall_mean": round(overall_mean, 3),
            "overall_std": round(overall_std, 3),
            "tasks": summary,
        },
        "episodes": all_results,
    }

    # Merge with existing results
    results_path = Path(RESULTS_FILE)
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                existing.append(output)
            else:
                existing = [existing, output]
            results_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            results_path.write_text(json.dumps([output], indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        results_path.write_text(json.dumps([output], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
