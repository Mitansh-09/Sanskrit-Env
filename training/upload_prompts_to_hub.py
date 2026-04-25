#!/usr/bin/env python3
"""
Upload or download the GRPO prompts JSONL on the Hugging Face Hub.

  # After you have a local prompts.jsonl (e.g. from a Job or collect_prompts_jsonl.py):
  export HF_TOKEN=hf_...
  python training/upload_prompts_to_hub.py upload runs/prompts.jsonl

  # Download (next job: set PULL_PROMPTS_FROM_HUB=1 or run download):
  python training/upload_prompts_to_hub.py download --output runs/prompts.jsonl

  # Try to extract JSONL lines from a log where each line is a full JSON object
  # (e.g. if you redirected structured lines). Messy real logs are often unusable;
  # prefer upload from the file produced on disk during the job.
  python training/upload_prompts_to_hub.py from-log job_log.txt -o recovered.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, List, Optional


def _default_repo() -> str:
    return (os.environ.get("HUB_PROMPTS_REPO") or "Adityahars/sanskrit-grpo-prompts").strip()


def _default_path() -> str:
    return (os.environ.get("HUB_PROMPTS_PATH_IN_REPO") or "data/prompts.jsonl").strip()


def cmd_upload(
    file: Path,
    repo_id: str,
    path_in_repo: str,
    private: bool,
    commit_message: str,
) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        print("error: pip install huggingface_hub", e, file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: set HF_TOKEN or run hf auth login", file=sys.stderr)
        return 1

    if not file.is_file():
        print(f"error: not a file: {file}", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )
    api.upload_file(
        path_or_fileobj=str(file.resolve()),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print(f"[ok] uploaded to https://huggingface.co/datasets/{repo_id} → {path_in_repo}", flush=True)
    return 0


def cmd_download(
    output: Path,
    repo_id: str,
    path_in_repo: str,
) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("error: pip install huggingface_hub", e, file=sys.stderr)
        return 1

    from shutil import copy2

    token = os.environ.get("HF_TOKEN")
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        token=token,
    )
    copy2(cached, output)
    print(f"[ok] saved {output} from Hub {repo_id}/{path_in_repo}", flush=True)
    return 0


def _is_prompt_row(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    return "prompt" in d and "task_id" in d and "seed" in d and "options" in d


def _try_parse_json_line(s: str) -> Optional[dict]:
    s = s.strip()
    i = s.find("{")
    if i < 0:
        return None
    dec = json.JSONDecoder()
    try:
        d, _ = dec.raw_decode(s[i:])
    except json.JSONDecodeError:
        return None
    if _is_prompt_row(d):
        return d
    return None


def _iter_log_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            d = _try_parse_json_line(line)
            if d is not None:
                yield d


def cmd_from_log(log_file: Path, output: Path) -> int:
    rows: List[dict] = list(_iter_log_jsonl(log_file))
    if not rows:
        print(
            "error: no valid prompt rows (need JSON lines with prompt, options, task_id, seed).",
            file=sys.stderr,
        )
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(rows)} lines to {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload/download GRPO prompts JSONL on the Hub")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("upload", help="Upload a JSONL file to a dataset repo on the Hub")
    p_up.add_argument("file", type=Path, help="Local prompts.jsonl")
    p_up.add_argument("--repo-id", default=_default_repo())
    p_up.add_argument("--path-in-repo", default=_default_path())
    p_up.add_argument("--private", action="store_true")
    p_up.add_argument("--message", default="Update GRPO prompts JSONL", dest="message")

    p_down = sub.add_parser("download", help="Download prompts JSONL from a dataset repo")
    p_down.add_argument("--output", "-o", type=Path, default=Path("runs/prompts.jsonl"))
    p_down.add_argument("--repo-id", default=_default_repo())
    p_down.add_argument("--path-in-repo", default=_default_path())

    p_log = sub.add_parser("from-log", help="Best-effort extract from log text; prefer uploading the real file")
    p_log.add_argument("log_file", type=Path)
    p_log.add_argument("--output", "-o", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "upload":
        return cmd_upload(
            args.file, args.repo_id, args.path_in_repo, args.private, args.message
        )
    if args.command == "download":
        return cmd_download(args.output, args.repo_id, args.path_in_repo)
    if args.command == "from-log":
        return cmd_from_log(args.log_file, args.output)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
