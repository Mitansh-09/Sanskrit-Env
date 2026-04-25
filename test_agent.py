"""
SanskritEnv Test Agent — local evaluation script.

Evaluates LLM performance on all SanskritEnv tasks, with special handling
for the manuscript_restoration tool-use POMDP.

Usage:
    python test_agent.py --task manuscript_restoration --episodes 3 --difficulty beginner
    python test_agent.py --task all --episodes 5
    python test_agent.py --provider cloudflare --model "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
"""

import argparse
import json
import os
import sys
import time
import random
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

DEFAULTS = {
    "model": os.environ.get("TEST_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    "provider": os.environ.get("TEST_PROVIDER", "cloudflare"),
    "task": os.environ.get("TEST_TASK", "all"),
    "episodes": int(os.environ.get("TEST_EPISODES", "10")),
    "seed": int(os.environ.get("TEST_SEED", "42")),
    "env_url": os.environ.get("TEST_ENV_URL", "http://localhost:7860"),
    "difficulty": os.environ.get("TEST_DIFFICULTY", "auto"),
    "output": os.environ.get("TEST_OUTPUT_FILE", "test_results.json"),
}

CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", os.environ.get("CF_API_TOKEN", ""))
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", os.environ.get("CF_ACCOUNT_ID", ""))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ROUTER_URL = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")

SYSTEM_PROMPT = """You are an expert Sanskrit philologist with knowledge of:
- Classical Sanskrit grammar, phonology, Paninian sandhi rules
- Samasa classification (Tatpurusha, Karmadharaya, Dvigu, Dvandva, Bahuvrihi, Avyayibhava)
- Vedic and classical Sanskrit literature across Ayurveda, astronomy, philosophy, narrative
- Textual criticism methodology: use evidence, check meter, consult commentary, compare witnesses

For standard questions (tasks 1-4): reply with EXACTLY one candidate option, verbatim.

For manuscript restoration (task 5):
  If you need more evidence: reply in format:
    TOOL: <tool_name>
    INPUT: <tool_input>

  If you are ready to commit: reply in format:
    COMMIT: <your final interpretation (must match one candidate option exactly)>

Available tools: lexicon_lookup, sandhi_parser, meter_checker, commentary_fetch,
                 witness_compare, referent_tracker

Rules:
- Never repeat a tool call with identical input
- Use meter_checker after sandhi_parser for the same compound
- Use commentary_fetch to verify lexicon findings
- Only commit when you have consulted at least 2 relevant tools
- For wrong answers: 0.0 reward regardless of evidence. Answer correctly."""


# ── LLM Providers ───────────────────────────────────────────────────────────

def call_cloudflare(messages, model, temperature=0.0, max_tokens=512):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Cloudflare call failed")


def call_huggingface(messages, model, temperature=0.0, max_tokens=512):
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(4):
        try:
            resp = requests.post(HF_ROUTER_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("HF call failed")


def call_llm(messages, model, provider):
    if provider == "cloudflare":
        return call_cloudflare(messages, model)
    elif provider == "huggingface":
        return call_huggingface(messages, model)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")


# ── Environment Client ──────────────────────────────────────────────────────

def env_reset(base_url, task_id, seed=None, difficulty=None):
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    if difficulty and difficulty != "auto":
        payload["difficulty"] = difficulty
    resp = requests.post(f"{base_url}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # OpenEnv servers may return wrapped StepResult payload:
    # {"observation": {...}, "reward": ..., "done": ...}
    if isinstance(data, dict) and "observation" in data:
        obs = data.get("observation") or {}
        if isinstance(obs, dict):
            obs.setdefault("reward", data.get("reward"))
            obs.setdefault("done", data.get("done", False))
            return obs
    return data


def env_step(base_url, action):
    # New OpenEnv API expects body shape: {"action": {...}}
    resp = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)

    # Backward-compat fallback for older servers expecting raw action payload.
    if resp.status_code == 422:
        resp = requests.post(f"{base_url}/step", json=action, timeout=30)

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

def match_to_option(text, candidates):
    text_clean = text.strip()
    for opt in candidates:
        if text_clean == opt:
            return opt
    for opt in candidates:
        if opt.startswith(text_clean) or text_clean.startswith(opt):
            return opt
    for opt in candidates:
        if text_clean.lower() in opt.lower() or opt.lower() in text_clean.lower():
            return opt
    return candidates[0] if candidates else text_clean


def parse_restoration_response(raw_text, candidates):
    raw = raw_text.strip()
    if raw.startswith("TOOL:"):
        lines = raw.split("\n")
        tool_name = lines[0].replace("TOOL:", "").strip()
        tool_input = lines[1].replace("INPUT:", "").strip() if len(lines) > 1 else ""
        return {"action_type": "tool_call", "tool_name": tool_name, "tool_input": tool_input}
    elif raw.startswith("COMMIT:"):
        answer = raw.replace("COMMIT:", "").strip()
        return {"action_type": "commit", "final_answer": match_to_option(answer, candidates)}
    else:
        return {"action_type": "commit", "final_answer": match_to_option(raw, candidates)}


# ── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(base_url, task_id, model, provider, seed, difficulty=None, verbose=False):
    obs = env_reset(base_url, task_id, seed=seed, difficulty=difficulty)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_trace = []
    total_steps = 0
    tools_used = []

    while not obs.get("done", False):
        candidates = obs.get("candidate_options", [])
        user_msg = f"Passage: {obs.get('source_text_iast', '')}\n"
        user_msg += f"Question: {obs.get('decision_prompt', '')}\n"
        user_msg += "Options:\n" + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(candidates))

        if obs.get("tool_call_history"):
            user_msg += "\n\nPrevious tool results:\n"
            for tc in obs["tool_call_history"][-3:]:  # last 3 for context window
                user_msg += f"  {tc.get('tool','')}: {json.dumps(tc.get('output',{}))[:150]}\n"

        if obs.get("steps_remaining") is not None:
            user_msg += f"\nBudget remaining: {obs['steps_remaining']}"

        messages.append({"role": "user", "content": user_msg})

        try:
            response = call_llm(messages, model, provider)
        except Exception as e:
            if verbose:
                print(f"    LLM error: {e}")
            break

        messages.append({"role": "assistant", "content": response})

        if task_id == "manuscript_restoration":
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
            print(f"    Step {total_steps}: {response[:80]}...")

        obs = env_step(base_url, action)
        total_steps += 1
        step_trace.append({"step": total_steps, "reward": obs.get("step_reward")})

        if total_steps > 20:
            break

    return {
        "task_id": task_id,
        "episode_id": obs.get("episode_id", ""),
        "difficulty": obs.get("difficulty", ""),
        "score": obs.get("reward", 0.0) or 0.0,
        "steps": total_steps,
        "tools_used": tools_used,
        "committed_correctly": (obs.get("reward", 0) or 0) > 0.5,
        "step_trace": step_trace,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SanskritEnv Test Agent")
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--provider", default=DEFAULTS["provider"], choices=["cloudflare", "huggingface"])
    parser.add_argument("--task", default=DEFAULTS["task"])
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--difficulty", default=DEFAULTS["difficulty"])
    parser.add_argument("--env-url", default=DEFAULTS["env_url"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = TASK_IDS if args.task == "all" else [args.task]
    rng = random.Random(args.seed)

    print(f"Testing {args.model} on SanskritEnv")
    print("-" * 65)

    all_episodes = []
    task_summaries = {}

    for idx, task_id in enumerate(tasks):
        scores = []
        task_tools = []
        task_steps = []

        for ep_num in range(args.episodes):
            seed = args.seed + ep_num
            diff = args.difficulty if task_id == "manuscript_restoration" else None
            try:
                result = run_episode(
                    args.env_url, task_id, args.model, args.provider, seed,
                    difficulty=diff, verbose=args.verbose,
                )
                scores.append(result["score"])
                task_tools.append(len(result["tools_used"]))
                task_steps.append(result["steps"])
                all_episodes.append(result)
            except Exception as e:
                print(f"  ERROR [{task_id}] ep {ep_num+1}: {e}")
                scores.append(0.0)

        mean = sum(scores) / len(scores) if scores else 0.0
        std = (sum((s - mean)**2 for s in scores) / len(scores))**0.5 if scores else 0.0

        task_summary = {"mean": round(mean, 3), "std": round(std, 3), "episodes": len(scores)}
        if task_id == "manuscript_restoration" and task_tools:
            task_summary["mean_tools_used"] = round(sum(task_tools) / len(task_tools), 1)
            task_summary["mean_steps_used"] = round(sum(task_steps) / len(task_steps), 1)

        task_summaries[task_id] = task_summary
        check = "ok" if mean > 0.3 else "!!"
        print(f"[Task {idx+1}/{len(tasks)}] {task_id:30s} | Episodes: {len(scores):2d} | Mean: {mean:.3f} {check}")

        if task_id == "manuscript_restoration" and task_tools:
            avg_tools = sum(task_tools) / len(task_tools)
            print(f"  -> Avg tools used: {avg_tools:.1f}  |  Avg steps: {sum(task_steps)/len(task_steps):.1f}")

    # Overall
    all_scores = [e["score"] for e in all_episodes]
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall_std = (sum((s - overall_mean)**2 for s in all_scores) / len(all_scores))**0.5 if all_scores else 0.0

    print("-" * 65)
    print(f"OVERALL: {overall_mean:.3f} +/- {overall_std:.3f}")

    # Save
    output = {
        "model": args.model,
        "provider": args.provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "summary": {
            "overall_mean": round(overall_mean, 3),
            "overall_std": round(overall_std, 3),
            "tasks": task_summaries,
        },
        "episodes": all_episodes,
    }

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
