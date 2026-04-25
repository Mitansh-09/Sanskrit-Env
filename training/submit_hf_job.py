"""
Submit a Hugging Face Job that clones this repo and runs training/scripts/hf_job_entrypoint.sh
on hosted GPU. See https://huggingface.co/docs/huggingface_hub/guides/jobs

Usage (from your machine; never commit tokens):
  export HF_TOKEN=...          # or: hf auth login
  # Default --namespace is Adityahars; set HF_JOB_NAMESPACE to override (avoids /whoami-v2 429 on resubmits).
  python training/submit_hf_job.py
  python training/submit_hf_job.py --flavor a10g-small --smoke --timeout 45m
  python training/submit_hf_job.py --flavor a100-large --timeout 12h
  # Full pipeline check: 5 train ep/task, 2 eval ep/task (baseline + post + compare), separate e2e artifacts
  python training/submit_hf_job.py --e2e-pipeline --flavor a100-large --timeout 3h
  # Default full run: 250 ep/task, group_size=4, upload prompts to datasets/Adityahars/sanskrit-grpo-prompts, push model
  python training/submit_hf_job.py --push-prompts --push-to-hub --flavor a100-large --timeout 12h
  # Reuse cached prompts from Hub (skip local collection)
  python training/submit_hf_job.py --pull-prompts --flavor a100-large --timeout 8h
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys


def _default_repo_url() -> str:
    return os.environ.get(
        "SANSKRIT_ENV_REPO_URL",
        "https://github.com/aditya-raj9125/Sanskrit-Env.git",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit SanskritEnv GRPO training to Hugging Face Jobs.")
    parser.add_argument(
        "--flavor",
        default=os.environ.get("HF_JOB_FLAVOR", "a100-large"),
        help="GPU flavor, e.g. t4-small, a10g-small, a100-large",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get(
            "HF_JOB_IMAGE",
            "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        ),
        help="Docker image with PyTorch+CUDA (job container).",
    )
    parser.add_argument(
        "--timeout",
        default=os.environ.get("HF_JOB_TIMEOUT", "12h"),
        help='Max job wall time, e.g. "12h", "6h" (Hub default is often 30m if omitted in API).',
    )
    parser.add_argument(
        "--repo-url",
        default=_default_repo_url(),
        help="Public git URL to clone. For private GitHub, set SANSKRIT_ENV_REPO_URL in env (PAT in URL; never commit).",
    )
    parser.add_argument(
        "--repo-branch",
        default=(os.environ.get("SANSKRIT_ENV_REPO_BRANCH", "main") or "main").strip(),
        help="Branch to clone (default: main). Set SANSKRIT_ENV_REPO_BRANCH to override.",
    )
    parser.add_argument(
        "--env-url",
        default=os.environ.get("ENV_URL", "https://adityahars-sanskrit-env.hf.space"),
        help="SanskritEnv HTTP base (deployed Space).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Set SMOKE_TEST=1 in the job (tiny episode count, 0.1 epoch, no standalone baseline).",
    )
    parser.add_argument(
        "--e2e-pipeline",
        action="store_true",
        help="Set E2E_PIPELINE_TEST=1: baseline (2 ep/task) + train (5 ep/task) + post + compare; uses separate cache/output paths.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=os.environ.get("PUSH_TO_HUB") == "1",
        help=(
            "Push the trained LoRA adapter to the Hugging Face Hub after training. "
            "Requires --hub-model-id or HUB_MODEL_ID env. Env: PUSH_TO_HUB=1."
        ),
    )
    parser.add_argument(
        "--hub-model-id",
        default=(os.environ.get("HUB_MODEL_ID") or "Adityahars/sanskrit-qwen-grpo"),
        help="Hub repo for the adapter, e.g. Adityahars/sanskrit-qwen-grpo. Env: HUB_MODEL_ID.",
    )
    parser.add_argument(
        "--push-prompts",
        action="store_true",
        default=os.environ.get("PUSH_PROMPTS_TO_HUB") == "1",
        help="Upload prompts.jsonl to a Hub dataset (HUB_PROMPTS_REPO). Env: PUSH_PROMPTS_TO_HUB=1.",
    )
    parser.add_argument(
        "--pull-prompts",
        action="store_true",
        default=os.environ.get("PULL_PROMPTS_FROM_HUB") == "1",
        help="Download prompts from Hub; skips local collect. Env: PULL_PROMPTS_FROM_HUB=1.",
    )
    parser.add_argument(
        "--hub-prompts-repo",
        default=(os.environ.get("HUB_PROMPTS_REPO") or "Adityahars/sanskrit-grpo-prompts").strip(),
        help="Hugging Face dataset id for shared prompts.jsonl. Env: HUB_PROMPTS_REPO.",
    )
    parser.add_argument(
        "--namespace",
        default=(os.environ.get("HF_JOB_NAMESPACE") or "Adityahars").strip(),
        help="Hub username or org for the job URL (skips /whoami-v2; avoids 429 on rapid resubmits). Default: Adityahars. Env: HF_JOB_NAMESPACE.",
    )
    args = parser.parse_args()

    if args.smoke and args.e2e_pipeline:
        print("error: use either --smoke or --e2e-pipeline, not both.", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "error: set HF_TOKEN in the environment, or run `hf auth login` so the hub can use your credentials.",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import run_job
    except ImportError as e:
        print("error: pip install -U huggingface_hub", e, file=sys.stderr)
        return 1

    # Inline shlex-quoted clone (reliable in job JSON). No `set -u` in bootstrap (Hub env can omit vars).
    repo_quoted = shlex.quote(args.repo_url.strip())
    branch_quoted = shlex.quote(args.repo_branch.strip())
    cmd = (
        "echo \"[hf-job] bootstrap: installing git + cloning repo\"; "
        "set -eo pipefail; "
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq; "
        "apt-get install -y -qq --no-install-recommends ca-certificates git; "
        f"git clone --depth 1 -b {branch_quoted} {repo_quoted} /tmp/sanskrit-env; "
        "test -f /tmp/sanskrit-env/training/scripts/hf_job_entrypoint.sh; "
        "exec bash /tmp/sanskrit-env/training/scripts/hf_job_entrypoint.sh"
    )

    env: dict[str, str] = {
        "ENV_URL": args.env_url.rstrip("/"),
        "HF_SPACE_URL": args.env_url.rstrip("/"),
    }
    if args.smoke or os.environ.get("SMOKE_TEST") == "1":
        env["SMOKE_TEST"] = "1"
    elif args.e2e_pipeline or os.environ.get("E2E_PIPELINE_TEST") == "1":
        env["E2E_PIPELINE_TEST"] = "1"
    if args.push_to_hub:
        env["PUSH_TO_HUB"] = "1"
        env["HUB_MODEL_ID"] = args.hub_model_id
    if args.push_prompts:
        env["PUSH_PROMPTS_TO_HUB"] = "1"
        env["HUB_PROMPTS_REPO"] = args.hub_prompts_repo
    if args.pull_prompts:
        env["PULL_PROMPTS_FROM_HUB"] = "1"
        env["HUB_PROMPTS_REPO"] = args.hub_prompts_repo
    for key in (
        "EPISODES_PER_TASK",
        "EPISODES_PER_TASK_EASY",
        "E2E_PIPELINE_TEST",
        "PUSH_TO_HUB",
        "HUB_MODEL_ID",
        "PUSH_PROMPTS_TO_HUB",
        "PULL_PROMPTS_FROM_HUB",
        "HUB_PROMPTS_REPO",
        "HUB_PROMPTS_PATH_IN_REPO",
        "TRAIN_EPOCHS",
        "EVAL_EPISODES",
        "EVAL_DURING_TRAIN",
        "NO_BASELINE_EVAL",
        "MODEL_ID",
        "OUTPUT_DIR",
        "DATASET_CACHE",
        "GROUP_SIZE",
        "PER_DEVICE_BATCH",
        "GRAD_ACCUM",
        "LR",
        "MAX_COMPLETION_LENGTH",
        "SANSKRIT_ENV_MIN_INTERVAL",
        "SANSKRIT_ENV_HTTP_RETRIES",
    ):
        v = os.environ.get(key)
        if v is not None and v != "":
            env[key] = v

    print("[info] submitting job...", flush=True)
    print(f"  image:   {args.image}", flush=True)
    print(f"  flavor:  {args.flavor}", flush=True)
    print(f"  timeout: {args.timeout}", flush=True)
    print(f"  env_url: {env['ENV_URL']}", flush=True)
    print(f"  branch:  {args.repo_branch}", flush=True)
    print(f"  clone:   {args.repo_url}", flush=True)
    if args.namespace:
        print(f"  namespace: {args.namespace} (skips whoami; avoids /whoami-v2 429)", flush=True)
    if env.get("SMOKE_TEST") == "1":
        print("  mode:    smoke (minimal, no standalone baseline by default)", flush=True)
    elif env.get("E2E_PIPELINE_TEST") == "1":
        print("  mode:    e2e-pipeline (5 ep/task train, 2 ep/task eval, full baseline+train+post+compare)", flush=True)
    if env.get("PUSH_TO_HUB") == "1":
        print(f"  push:    Hub model -> {env.get('HUB_MODEL_ID', 'Adityahars/sanskrit-qwen-grpo')}", flush=True)
    if env.get("PULL_PROMPTS_FROM_HUB") == "1":
        print(
            f"  prompts: pull from datasets/{env.get('HUB_PROMPTS_REPO', 'Adityahars/sanskrit-grpo-prompts')}",
            flush=True,
        )
    if env.get("PUSH_PROMPTS_TO_HUB") == "1":
        print(
            f"  prompts: push to datasets/{env.get('HUB_PROMPTS_REPO', 'Adityahars/sanskrit-grpo-prompts')}",
            flush=True,
        )

    job = run_job(
        image=args.image,
        command=["bash", "-c", cmd],
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        env=env,
        namespace=args.namespace,
        token=token,
    )
    print(f"[ok] job id: {job.id}", flush=True)
    print(f"[ok] url:    {job.url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
