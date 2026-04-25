"""
Standalone evaluation script for SanskritEnv.

Loads a base model (optionally with a LoRA adapter) and runs N evaluation
episodes per task against a running SanskritEnv server. Writes a JSON summary
that can be diffed against another run via `compare_evals.py`.

Usage examples:

    # Evaluate the untrained base model (run BEFORE training).
    python training/evaluate.py \
        --env-url http://localhost:7860 \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --episodes-per-task 30 \
        --output runs/eval_baseline.json

    # Evaluate a trained adapter (run AFTER training).
    python training/evaluate.py \
        --env-url http://localhost:7860 \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/qwen25-1p5b-grpo \
        --episodes-per-task 30 \
        --output runs/eval_post.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the env client + prompt formatting from train_grpo.py without
# pulling in the heavy training-only imports.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from train_grpo import (  # noqa: E402  (import path needs sys.path tweak above)
    TASK_IDS,
    build_user_prompt,
    env_reset,
    env_step,
    format_chat_prompt,
    match_to_option,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SanskritEnv policy.")
    parser.add_argument(
        "--env-url",
        default=(
            os.environ.get("ENV_URL")
            or os.environ.get("HF_SPACE_URL")
            or "https://adityahars-sanskrit-env.hf.space"
        ),
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional path to a PEFT/LoRA adapter directory.")
    parser.add_argument("--tasks", nargs="*", default=TASK_IDS)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--difficulty", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Use 4-bit weights for the base model. Off by default; useful on T4/V100.",
    )
    parser.add_argument("--output", default="runs/eval.json")
    parser.add_argument("--label", default=None, help="Optional human-readable label stored in the JSON.")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model, tokenizer


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    model, tokenizer = load_model(args)
    device = next(model.parameters()).device

    per_task: Dict[str, Any] = {}
    all_scores: List[float] = []
    per_episode: List[Dict[str, Any]] = []

    for task in args.tasks:
        scores: List[float] = []
        for ep in range(args.episodes_per_task):
            seed = args.base_seed + ep * 7919 + (abs(hash(task)) % 9999)
            try:
                obs = env_reset(args.env_url, task, seed=seed, difficulty=args.difficulty)
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
                    max_new_tokens=args.max_new_tokens,
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
                next_obs = env_step(args.env_url, action)
                reward = float(next_obs.get("step_reward") or 0.0)
            except Exception as exc:
                print(f"  [eval] step failed task={task} seed={seed}: {exc}", file=sys.stderr)
                reward = 0.0

            scores.append(reward)
            per_episode.append(
                {
                    "task_id": task,
                    "seed": seed,
                    "score": reward,
                    "completion": completion[:200],
                    "selected_option": selected,
                }
            )

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
        }
        all_scores.extend(scores)
        print(
            f"  [eval] {task:30s} mean={mean:.3f} std={std:.3f} success={success:.3f}",
            flush=True,
        )

    overall_n = max(len(all_scores), 1)
    overall_mean = sum(all_scores) / overall_n
    overall_var = sum((s - overall_mean) ** 2 for s in all_scores) / overall_n
    overall_std = overall_var ** 0.5
    overall_success = sum(1 for s in all_scores if s >= 0.5) / overall_n

    return {
        "label": args.label or ("post-train" if args.adapter else "baseline"),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "env_url": args.env_url,
        "tasks": args.tasks,
        "episodes_per_task": args.episodes_per_task,
        "base_seed": args.base_seed,
        "difficulty": args.difficulty,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "overall_mean": round(overall_mean, 4),
            "overall_std": round(overall_std, 4),
            "overall_success_rate": round(overall_success, 4),
            "tasks": per_task,
        },
        "episodes": per_episode,
    }


def main() -> None:
    args = parse_args()
    print(f"[info] env_url={args.env_url} base={args.base_model} adapter={args.adapter}", flush=True)
    print(f"[info] tasks={args.tasks} episodes_per_task={args.episodes_per_task}", flush=True)

    t0 = time.time()
    result = evaluate(args)
    elapsed = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = result["summary"]
    print(
        f"[done] overall_mean={summary['overall_mean']:.3f} "
        f"std={summary['overall_std']:.3f} success={summary['overall_success_rate']:.3f} "
        f"in {elapsed:.1f}s -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
