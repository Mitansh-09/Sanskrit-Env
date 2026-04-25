"""
GRPO training script for SanskritEnv.

Trains a small instruct LLM (default: Qwen/Qwen2.5-1.5B-Instruct) end-to-end
across all six SanskritEnv tasks using HuggingFace TRL's GRPOTrainer.

Pipeline:
  1. Connect to a running SanskritEnv server (HTTP, OpenEnv API).
  2. Collect prompts per task by repeatedly calling /reset with deterministic
     seeds, capturing decision prompt + candidate options.
  3. Define a reward function that, for each model completion, re-resets the
     env with the same (task_id, seed) and calls /step. The returned
     `step_reward` is the GRPO scalar reward for that completion.
  4. Wrap the policy in 4-bit QLoRA (PEFT) and train with GRPOTrainer.

Default footprint targets a single 16 GB GPU (T4/V100/A10) on Google Colab.

Example usage (locally or in Colab after env server is up):

    python training/train_grpo.py \
        --env-url http://localhost:7860 \
        --model-id Qwen/Qwen2.5-1.5B-Instruct \
        --episodes-per-task 1500 \
        --output-dir runs/qwen25-1p5b-grpo \
        --group-size 4 \
        --lr 5e-6
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

TASK_IDS = [
    "glossary_anchoring",
    "sandhi_resolution",
    "samasa_classification",
    "referential_coherence",
    "manuscript_restoration",
    "full_manuscript_session",
]
# “Easy” decision tasks (1–4); 5–6 are longer multi-step / manuscript runs.
TASK_IDS_EASY = TASK_IDS[:4]


# ───────────────────────── Env HTTP client ─────────────────────────

# Pacing: deployed HF Spaces rate-limit bursty /reset and /step traffic.
_last_env_request_mono: float = 0.0


def _pace_env_request(url: str) -> None:
    """Enforce a minimum interval between env HTTP calls to reduce HTTP 429 on hf.space."""
    global _last_env_request_mono
    raw = (os.environ.get("SANSKRIT_ENV_MIN_INTERVAL") or "").strip()
    if raw:
        interval = max(0.0, float(raw))
    else:
        u = url.lower()
        interval = 0.35 if ("hf.space" in u or "huggingface.co" in u) else 0.0
    if interval <= 0.0:
        return
    now = time.monotonic()
    wait = interval - (now - _last_env_request_mono)
    if wait > 0:
        time.sleep(wait)
    _last_env_request_mono = time.monotonic()


def _http_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    max_attempts = int(os.environ.get("SANSKRIT_ENV_HTTP_RETRIES", "12"))
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "SanskritEnv-train_grpo/1.0",
    }
    for attempt in range(max_attempts):
        _pace_env_request(url)
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_attempts - 1:
                ra = e.headers.get("Retry-After", "") if e.headers else ""
                if ra and ra.isdigit():
                    time.sleep(min(float(ra), 90.0))
                else:
                    time.sleep(min(2.0**attempt * 0.4 + random.random() * 0.5, 60.0))
                continue
            raise
        except (urllib.error.URLError, OSError) as e:
            if attempt < max_attempts - 1:
                time.sleep(min(0.5 * (2**attempt), 20.0))
                continue
            raise
    raise RuntimeError("unreachable")


def _unwrap_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict) and "observation" in data:
        obs = data.get("observation") or {}
        if isinstance(obs, dict):
            obs.setdefault("reward", data.get("reward"))
            obs.setdefault("done", data.get("done", False))
            return obs
    return data


def env_reset(base_url: str, task_id: str, seed: int, difficulty: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_id": task_id, "seed": int(seed)}
    if difficulty and difficulty != "auto":
        payload["difficulty"] = difficulty
    return _unwrap_observation(_http_post(f"{base_url}/reset", payload))


def env_step(base_url: str, action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _unwrap_observation(_http_post(f"{base_url}/step", {"action": action}))
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            return _unwrap_observation(_http_post(f"{base_url}/step", action))
        raise


# ───────────────────────── Prompt formatting ─────────────────────────

SYSTEM_INSTRUCTION = (
    "You are an expert Sanskrit philologist trained on classical grammar, "
    "Paninian sandhi rules, samasa classification, and Vedic/Ayurvedic/"
    "philosophical/narrative literature. For every question reply with EXACTLY "
    "one option from the provided list, copied verbatim, with no extra text."
)


def build_user_prompt(obs: Dict[str, Any]) -> str:
    lines: List[str] = []
    if obs.get("source_text_iast"):
        lines.append(f"Sanskrit (IAST): {obs['source_text_iast']}")
    if obs.get("source_text_devanagari"):
        lines.append(f"Devanagari:      {obs['source_text_devanagari']}")
    if obs.get("english_context"):
        lines.append(f"Context:         {obs['english_context']}")
    if obs.get("domain"):
        lines.append(f"Domain:          {obs['domain']}")
    if obs.get("target_term_iast"):
        lines.append(f"Target term:     {obs['target_term_iast']}")
    if obs.get("compound_iast"):
        lines.append(f"Compound:        {obs['compound_iast']}")

    verses = obs.get("verses_so_far") or []
    if verses:
        lines.append("")
        lines.append("Verses so far:")
        for verse in verses:
            lines.append(f"  [{verse.get('verse_num', '?')}] IAST: {verse.get('iast', '')}")
            if verse.get("english"):
                lines.append(f"       English: {verse['english']}")

    decision_prompt = obs.get("decision_prompt", "")
    if decision_prompt:
        lines.append("")
        lines.append(f"Question: {decision_prompt}")

    options = obs.get("candidate_options") or []
    if options:
        lines.append("")
        lines.append("Options (reply with exactly one, verbatim):")
        for i, opt in enumerate(options):
            lines.append(f"  {i + 1}. {opt}")

    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def format_chat_prompt(tokenizer, user_prompt: str) -> str:
    """Render the chat template into a single string the trainer can tokenize."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def match_to_option(raw: str, options: List[str]) -> str:
    text = (raw or "").strip()
    if not options:
        return text
    for opt in options:
        if text == opt:
            return opt
    lowered = text.lower()
    for opt in options:
        if opt.lower() == lowered:
            return opt
    for opt in options:
        if opt.lower().startswith(lowered[:30]) and lowered:
            return opt
    for opt in options:
        if opt.lower() in lowered or lowered in opt.lower():
            return opt
    return options[0]


# ───────────────────────── Dataset collection ─────────────────────────


def _env_positive_int(name: str) -> Optional[int]:
    """Parse EPISODES_PER_TASK_EASY style env; empty or non-positive => None."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    v = int(raw)
    return None if v <= 0 else v


def resolve_training_episode_counts(
    tasks: List[str],
    uniform: int,
    easy: Optional[int],
) -> Union[int, Dict[str, int]]:
    """If `easy` is set, use it for TASK_IDS_EASY (tasks 1–4); all others use `uniform`."""
    if easy is None:
        return uniform
    easy_set = set(TASK_IDS_EASY)
    return {t: (easy if t in easy_set else uniform) for t in tasks}


def collect_prompt_dataset(
    env_url: str,
    episodes_per_task: Union[int, Dict[str, int]],
    base_seed: int,
    tasks: List[str],
    difficulty: Optional[str] = None,
):
    """Reset the env once per (task, episode) and capture the first decision step.

    Returns a list of dicts with keys: prompt, options, task_id, seed.
    The same (task_id, seed) is replayed at reward time to score completions
    deterministically against the env grader.

    `episodes_per_task` is either a single int (same for every task) or a map
    task_id -> episode count.
    """
    from datasets import Dataset

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        n = episodes_per_task[task] if isinstance(episodes_per_task, dict) else episodes_per_task
        print(f"[collect] task={task} episodes={n}", flush=True)
        for ep in range(n):
            seed = base_seed + ep * 7919 + (abs(hash(task)) % 9999)
            try:
                obs = env_reset(env_url, task, seed=seed, difficulty=difficulty)
            except Exception as exc:
                print(f"  [warn] reset failed seed={seed}: {exc}", file=sys.stderr)
                continue
            options = obs.get("candidate_options") or []
            prompt = build_user_prompt(obs)
            if not options or not prompt:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "options": options,
                    "task_id": task,
                    "seed": seed,
                }
            )
    print(f"[collect] total prompts: {len(rows)}", flush=True)
    return Dataset.from_list(rows)


# ───────────────────────── Reward function ─────────────────────────


def make_reward_function(env_url: str, difficulty: Optional[str] = None):
    """Return a TRL-compatible reward function bound to a SanskritEnv server.

    GRPOTrainer passes lists of equal length to `reward_funcs`:
      - completions: list[str]
      - extra dataset columns expanded as keyword args (here: task_id, seed, options)

    For each completion we re-reset the env with the matching (task_id, seed),
    parse the model's text into one of `options`, then call /step. The env's
    `step_reward` becomes the GRPO scalar reward for that sample.
    """

    def reward_fn(completions, task_id, seed, options, **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion, task, sd, opts in zip(completions, task_id, seed, options):
            try:
                obs = env_reset(env_url, task, seed=int(sd), difficulty=difficulty)
                live_options = obs.get("candidate_options") or list(opts) or []
                selected = match_to_option(str(completion), live_options)
                action: Dict[str, Any] = {
                    "selected_option": selected,
                    "confidence": 0.6,
                    "reasoning": str(completion)[:200],
                }
                if task in ("manuscript_restoration", "full_manuscript_session"):
                    action.update(
                        {
                            "action_type": "commit",
                            "final_answer": selected,
                        }
                    )
                next_obs = env_step(env_url, action)
                reward = float(next_obs.get("step_reward") or 0.0)
            except Exception as exc:
                print(f"  [reward] error task={task} seed={sd}: {exc}", file=sys.stderr)
                reward = 0.0
            rewards.append(reward)
        return rewards

    return reward_fn


# ───────────────────────── In-process evaluation ─────────────────────────


def evaluate_policy(
    model,
    tokenizer,
    env_url: str,
    tasks: List[str],
    episodes_per_task: int,
    base_seed: int,
    difficulty: Optional[str] = None,
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    """Run a deterministic evaluation pass across all tasks.

    Returns a dict with per-task aggregates and an overall summary, mirroring
    the layout of `test_results.json` so it can be diffed against baselines.

    Notes:
      - Uses greedy decoding (`do_sample=False`) for reproducibility.
      - Reuses (task_id, seed) pairs that are disjoint from training seeds
        when called with `base_seed=args.eval_base_seed` (default 10_000).
    """
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    per_task: Dict[str, Dict[str, Any]] = {}
    all_scores: List[float] = []

    for task in tasks:
        scores: List[float] = []
        steps_list: List[int] = []
        for ep in range(episodes_per_task):
            seed = base_seed + ep * 7919 + (abs(hash(task)) % 9999)
            try:
                obs = env_reset(env_url, task, seed=seed, difficulty=difficulty)
            except Exception as exc:
                print(f"  [eval] reset failed task={task} seed={seed}: {exc}", file=sys.stderr)
                continue

            options = obs.get("candidate_options") or []
            if not options:
                continue

            prompt_text = format_chat_prompt(tokenizer, build_user_prompt(obs))
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            selected = match_to_option(completion, options)
            action: Dict[str, Any] = {
                "selected_option": selected,
                "confidence": 0.7,
                "reasoning": completion[:200],
            }
            if task in ("manuscript_restoration", "full_manuscript_session"):
                action.update({"action_type": "commit", "final_answer": selected})

            try:
                next_obs = env_step(env_url, action)
                reward = float(next_obs.get("step_reward") or 0.0)
            except Exception as exc:
                print(f"  [eval] step failed task={task} seed={seed}: {exc}", file=sys.stderr)
                reward = 0.0

            scores.append(reward)
            steps_list.append(1)

        n = max(len(scores), 1)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = var ** 0.5
        success = sum(1 for s in scores if s >= 0.5) / n
        full_credit = sum(1 for s in scores if s >= 0.95) / n

        per_task[task] = {
            "n_episodes": len(scores),
            "score_mean": round(mean, 4),
            "score_std": round(std, 4),
            "success_rate": round(success, 4),
            "full_credit_rate": round(full_credit, 4),
            "mean_steps": round(sum(steps_list) / max(len(steps_list), 1), 4),
        }
        all_scores.extend(scores)

    if was_training:
        model.train()

    overall_n = max(len(all_scores), 1)
    overall_mean = sum(all_scores) / overall_n
    overall_var = sum((s - overall_mean) ** 2 for s in all_scores) / overall_n
    overall_std = overall_var ** 0.5
    overall_success = sum(1 for s in all_scores if s >= 0.5) / overall_n

    return {
        "tasks": per_task,
        "overall": {
            "n_episodes": len(all_scores),
            "score_mean": round(overall_mean, 4),
            "score_std": round(overall_std, 4),
            "success_rate": round(overall_success, 4),
        },
    }


def _save_metrics_history(metrics_path: Path, history: List[Dict[str, Any]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, ensure_ascii=False, indent=2)


# ───────────────────────── Training entrypoint ─────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training on SanskritEnv.")
    parser.add_argument(
        "--env-url",
        default=(
            os.environ.get("ENV_URL")
            or os.environ.get("HF_SPACE_URL")
            or "https://adityahars-sanskrit-env.hf.space"
        ),
        help="SanskritEnv HTTP base (OpenAPI /reset, /step). Default: ENV_URL, else HF_SPACE_URL, else the project HF Space.",
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=int(os.environ.get("EPISODES_PER_TASK", "250")),
        help="Episodes for tasks 5–6 (and for all tasks if --episodes-per-task-easy is not set).",
    )
    parser.add_argument(
        "--episodes-per-task-easy",
        type=int,
        default=_env_positive_int("EPISODES_PER_TASK_EASY"),
        help=(
            "Episodes for tasks 1–4 (glossary_anchoring, sandhi, samasa, referential). "
            "Omit or set a non-positive value (or unset EPISODES_PER_TASK_EASY) to use "
            "the same count as --episodes-per-task for every task. Env: EPISODES_PER_TASK_EASY."
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=TASK_IDS, help="Subset of task ids to train on.")
    parser.add_argument("--difficulty", default="auto", help="Used by tasks 5/6 (beginner|intermediate|hard|expert|auto).")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/qwen25-1p5b-grpo")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO num_generations per prompt (smaller = faster, default 4).")
    parser.add_argument("--per-device-batch", type=int, default=2, help="Per-device batch size (A100 default).")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (A100 default).")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of full passes over the prompt set.")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Token-level truncation of each training prompt (applied to the dataset; not a GRPOConfig arg in recent trl).",
    )
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit QLoRA. Off by default; useful only on T4/V100 (<24 GB).",
    )
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--dataset-cache", default=None, help="Optional path to cache the prompt dataset (jsonl).")
    parser.add_argument("--dry-run", action="store_true", help="Only collect dataset and print stats.")
    parser.add_argument(
        "--eval-episodes-per-task",
        type=int,
        default=20,
        help=(
            "Episodes per task for per-epoch and post-training eval. Use 0 to disable those. "
            "Pre-training baseline still runs 1 episode/task if --baseline-eval (default) so the "
            "model is exercised before GRPO."
        ),
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=10_000,
        help="Seed offset for evaluation episodes (kept disjoint from training seeds).",
    )
    parser.add_argument(
        "--baseline-eval",
        action="store_true",
        default=True,
        help="Run a pre-training evaluation pass to capture baseline metrics.",
    )
    parser.add_argument("--no-baseline-eval", dest="baseline_eval", action="store_false")
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Path for the metrics history JSON. Defaults to <output_dir>/metrics_history.json.",
    )
    args = parser.parse_args()
    if args.episodes_per_task_easy is not None and args.episodes_per_task_easy <= 0:
        args.episodes_per_task_easy = None
    return args


def truncate_dataset_prompts(
    dataset: Any,
    tokenizer: Any,
    max_length: int,
):
    """TRL recent GRPOConfig no longer takes max_prompt_length; apply truncation on the `prompt` column."""
    if max_length <= 0 or len(dataset) == 0 or "prompt" not in (dataset.column_names or []):
        return dataset

    def _row(ex: Dict[str, Any]) -> Dict[str, Any]:
        p = ex.get("prompt") or ""
        if not p:
            return ex
        enc = tokenizer(
            p,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        ex["prompt"] = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        return ex

    return dataset.map(_row, desc=f"truncate prompts (max {max_length} tok)")


def maybe_load_or_save_dataset(args, build_fn):
    if args.dataset_cache and os.path.exists(args.dataset_cache):
        from datasets import Dataset

        with open(args.dataset_cache, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(f"[cache] loaded {len(rows)} prompts from {args.dataset_cache}", flush=True)
        return Dataset.from_list(rows)

    dataset = build_fn()
    if args.dataset_cache:
        Path(args.dataset_cache).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dataset_cache, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[cache] saved {len(dataset)} prompts to {args.dataset_cache}", flush=True)
    return dataset


def main() -> None:
    args = parse_args()

    print(f"[info] env_url={args.env_url} model={args.model_id}", flush=True)
    ep_counts = resolve_training_episode_counts(
        args.tasks, args.episodes_per_task, args.episodes_per_task_easy
    )
    if isinstance(ep_counts, dict):
        print(
            f"[info] tasks={args.tasks} episodes (1–4 easy, rest --episodes-per-task)={ep_counts}",
            flush=True,
        )
    else:
        print(
            f"[info] tasks={args.tasks} episodes_per_task={ep_counts} (uniform)",
            flush=True,
        )

    try:
        ping = _http_post(f"{args.env_url}/reset", {"task_id": "glossary_anchoring", "seed": args.base_seed})
        ping_done = ping.get("done", "?") if isinstance(ping, dict) else "?"
        print(f"[info] env reachable, sample reset done={ping_done}", flush=True)
    except Exception as exc:
        print(f"[fatal] cannot reach env at {args.env_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def build_dataset_fn():
        raw = collect_prompt_dataset(
            env_url=args.env_url,
            episodes_per_task=ep_counts,
            base_seed=args.base_seed,
            tasks=args.tasks,
            difficulty=args.difficulty,
        )
        templated = raw.map(
            lambda ex: {"prompt": format_chat_prompt(tokenizer, ex["prompt"])},
            desc="apply chat template",
        )
        return templated

    dataset = maybe_load_or_save_dataset(args, build_dataset_fn)
    dataset = truncate_dataset_prompts(dataset, tokenizer, args.max_prompt_length)

    if args.dry_run:
        print("[dry-run] dataset preview:")
        if len(dataset) > 0:
            print(json.dumps({k: dataset[0][k] for k in dataset.column_names}, ensure_ascii=False, indent=2)[:2000])
        return

    quant_config = None
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        import torch

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    from peft import LoraConfig, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.group_size,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    metrics_path = Path(args.metrics_file or os.path.join(args.output_dir, "metrics_history.json"))
    metrics_history: List[Dict[str, Any]] = []

    eval_kwargs = dict(
        env_url=args.env_url,
        tasks=args.tasks,
        episodes_per_task=args.eval_episodes_per_task,
        base_seed=args.eval_base_seed,
        difficulty=args.difficulty,
        max_new_tokens=args.max_completion_length,
    )
    # At least one episode before training so base+LoRA is callable (generate + env step), even
    # when per-epoch eval is disabled (eval_episodes_per_task==0).
    if args.baseline_eval:
        baseline_episodes = args.eval_episodes_per_task if args.eval_episodes_per_task > 0 else 1
    else:
        baseline_episodes = 0
    baseline_eval_kwargs = {**eval_kwargs, "episodes_per_task": baseline_episodes}

    from transformers import TrainerCallback

    class EpochEvalCallback(TrainerCallback):
        """Runs an eval pass at the end of every epoch and persists metrics_history.json."""

        def on_epoch_end(self, callback_args, state, control, **kwargs):
            if args.eval_episodes_per_task <= 0:
                return control
            current_model = kwargs.get("model")
            current_tokenizer = kwargs.get("processing_class") or tokenizer
            print(
                f"[eval] epoch={state.epoch:.2f} step={state.global_step} "
                f"running {args.eval_episodes_per_task} episodes/task...",
                flush=True,
            )
            t0 = time.time()
            metrics = evaluate_policy(current_model, current_tokenizer, **eval_kwargs)
            elapsed = time.time() - t0
            entry = {
                "phase": "post_epoch",
                "epoch": float(state.epoch or 0.0),
                "global_step": int(state.global_step or 0),
                "elapsed_seconds": round(elapsed, 2),
                "metrics": metrics,
            }
            metrics_history.append(entry)
            _save_metrics_history(metrics_path, metrics_history)
            overall = metrics["overall"]
            print(
                f"[eval] epoch={state.epoch:.2f} overall_mean={overall['score_mean']:.3f} "
                f"std={overall['score_std']:.3f} success_rate={overall['success_rate']:.3f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            return control

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=make_reward_function(args.env_url, difficulty=args.difficulty),
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[EpochEvalCallback()] if args.eval_episodes_per_task > 0 else None,
    )

    if args.baseline_eval and baseline_episodes > 0:
        print(
            f"[eval] running baseline pass ({baseline_episodes} episodes/task, pre-training sanity)...",
            flush=True,
        )
        t0 = time.time()
        baseline_metrics = evaluate_policy(trainer.model, tokenizer, **baseline_eval_kwargs)
        elapsed = time.time() - t0
        metrics_history.append(
            {
                "phase": "baseline",
                "epoch": 0.0,
                "global_step": 0,
                "elapsed_seconds": round(elapsed, 2),
                "metrics": baseline_metrics,
            }
        )
        _save_metrics_history(metrics_path, metrics_history)
        print(
            f"[eval] baseline overall_mean={baseline_metrics['overall']['score_mean']:.3f} "
            f"std={baseline_metrics['overall']['score_std']:.3f} ({elapsed:.1f}s)",
            flush=True,
        )

    print(f"[info] starting GRPO training, prompts={len(dataset)}", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"[info] training finished in {time.time() - t0:.1f}s", flush=True)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[info] saved adapter + tokenizer to {args.output_dir}", flush=True)

    if args.eval_episodes_per_task > 0 and (
        not metrics_history or metrics_history[-1].get("phase") != "post_epoch"
    ):
        print("[eval] running final post-training pass...", flush=True)
        final_metrics = evaluate_policy(trainer.model, tokenizer, **eval_kwargs)
        metrics_history.append(
            {
                "phase": "final",
                "epoch": float(args.epochs),
                "global_step": int(trainer.state.global_step or 0),
                "metrics": final_metrics,
            }
        )
        _save_metrics_history(metrics_path, metrics_history)

    if metrics_history:
        baseline_entry = next((h for h in metrics_history if h["phase"] == "baseline"), None)
        last_entry = metrics_history[-1]
        if baseline_entry:
            b = baseline_entry["metrics"]["overall"]
            f = last_entry["metrics"]["overall"]
            delta = f["score_mean"] - b["score_mean"]
            rel = (delta / max(b["score_mean"], 1e-3)) * 100
            print(
                f"[summary] baseline overall_mean={b['score_mean']:.3f} -> "
                f"final overall_mean={f['score_mean']:.3f} "
                f"(delta {delta:+.3f}, {rel:+.1f}%)",
                flush=True,
            )

    print(f"[info] metrics history written to {metrics_path}", flush=True)

    if args.push_to_hub and args.hub_model_id:
        trainer.push_to_hub()
        print(f"[info] pushed to hub: {args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
