#!/usr/bin/env python3
"""
Build GRPO prompt cache JSONL from local data/*.json (in-process SanskritEnvironment).

No HTTP / no rate limits. Output matches --dataset-cache format for train_grpo.py.

Usage (repo root):
  python training/collect_prompts_jsonl.py --output runs/prompts.jsonl --episodes-per-task 250
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _ROOT / "training"
for p in (str(_ROOT), str(_TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_grpo  # noqa: E402
from server.environment import SanskritEnvironment  # noqa: E402

build_user_prompt = train_grpo.build_user_prompt
format_chat_prompt = train_grpo.format_chat_prompt
resolve_training_episode_counts = train_grpo.resolve_training_episode_counts
TASK_IDS = train_grpo.TASK_IDS


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()  # type: ignore[no-untyped-call]
    raise TypeError(f"Unknown observation type: {type(obs)}")


def build_rows(
    env: SanskritEnvironment,
    ep_counts: Any,
    tasks: List[str],
    base_seed: int,
    difficulty: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        n = ep_counts[task] if isinstance(ep_counts, dict) else ep_counts
        print(f"[collect-local] task={task} episodes={n}", flush=True)
        for ep in range(n):
            seed = base_seed + ep * 7919 + (abs(hash(task)) % 9999)
            kwargs: Dict[str, Any] = {"seed": seed, "task_id": task}
            if difficulty and difficulty != "auto":
                kwargs["difficulty"] = difficulty
            obs = env.reset(**kwargs)
            o = _obs_to_dict(obs)
            options = o.get("candidate_options") or []
            prompt = build_user_prompt(o)
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
    print(f"[collect-local] total raw prompts: {len(rows)}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GRPO dataset JSONL from local data/ JSON (in-process env).")
    parser.add_argument(
        "--output",
        default=os.environ.get("DATASET_CACHE", "runs/prompts.jsonl"),
        help="Output JSONL (same as train_grpo --dataset-cache).",
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--episodes-per-task", type=int, default=int(os.environ.get("EPISODES_PER_TASK", "250")))
    parser.add_argument(
        "--episodes-per-task-easy",
        type=int,
        default=int(os.environ.get("EPISODES_PER_TASK_EASY", "0") or 0),
        help="If >0, use for tasks 1-4; else uniform --episodes-per-task.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--difficulty", default="auto")
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of task ids; default: all 6.")
    args = parser.parse_args()

    easy: int | None = args.episodes_per_task_easy if args.episodes_per_task_easy > 0 else None
    tasks = list(args.tasks) if args.tasks else list(TASK_IDS)
    ep_counts = resolve_training_episode_counts(tasks, args.episodes_per_task, easy)
    if isinstance(ep_counts, dict):
        print(f"[info] per-task episode counts: {ep_counts}", flush=True)
    else:
        print(f"[info] uniform episodes_per_task={ep_counts}", flush=True)

    try:
        env = SanskritEnvironment()
    except (FileNotFoundError, OSError, KeyError) as e:
        print(f"[error] could not load SanskritEnvironment (is data/ complete?): {e}", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    rows = build_rows(env, ep_counts, tasks, args.base_seed, str(args.difficulty))
    if not rows:
        print("[error] no rows; check data files and episode counts.", file=sys.stderr)
        return 1

    print(f"[info] chat template (model_id={args.model_id})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(out_path.parent),
        prefix=".prompts-",
        suffix=".jsonl.tmp",
    ) as tmp:
        for row in rows:
            templated = {
                "prompt": format_chat_prompt(tokenizer, row["prompt"]),
                "options": row["options"],
                "task_id": row["task_id"],
                "seed": row["seed"],
            }
            tmp.write(json.dumps(templated, ensure_ascii=False) + "\n")
        tmp_path = tmp.name
    os.replace(tmp_path, out_path)
    print(f"[cache] saved {len(rows)} prompts to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
