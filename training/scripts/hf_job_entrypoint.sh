#!/usr/bin/env bash
# Hugging Face Job entrypoint: optional baseline eval -> GRPO train -> post eval -> compare.
# HF_TOKEN must be set (job secret). ENV_URL points at the deployed SanskritEnv Space.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Pass it as a job secret (see submit_hf_job.py)."
  exit 1
fi

export ENV_URL="${ENV_URL:-${HF_SPACE_URL:-https://adityahars-sanskrit-env.hf.space}}"
export ENV_URL="${ENV_URL%/}"

# SMOKE_TEST=1: minimal run, skips baseline by default, fast sanity check
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export EPISODES_PER_TASK_EASY="${EPISODES_PER_TASK_EASY:-2}"
  export EPISODES_PER_TASK="${EPISODES_PER_TASK:-2}"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-0.1}"
  export EVAL_EPISODES="${EVAL_EPISODES:-2}"
  export EVAL_DURING_TRAIN="${EVAL_DURING_TRAIN:-0}"
  export NO_BASELINE_EVAL="${NO_BASELINE_EVAL:-1}"
  echo "[smoke] EPISODES_PER_TASK_EASY=$EPISODES_PER_TASK_EASY EPISODES_PER_TASK=$EPISODES_PER_TASK EVAL_EPISODES=$EVAL_EPISODES TRAIN_EPOCHS=$TRAIN_EPOCHS"
# E2E_PIPELINE_TEST=1: full path baseline -> train -> post -> compare, tiny counts (5 ep/task train, 2 ep/task eval)
elif [[ "${E2E_PIPELINE_TEST:-0}" == "1" ]]; then
  export EPISODES_PER_TASK_EASY="${EPISODES_PER_TASK_EASY:-5}"
  export EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1.0}"
  export EVAL_EPISODES="${EVAL_EPISODES:-2}"
  export EVAL_DURING_TRAIN="${EVAL_DURING_TRAIN:-2}"
  export NO_BASELINE_EVAL="${NO_BASELINE_EVAL:-0}"
  # Separate paths so a full run does not load this small cache or overwrite eval artifacts
  export DATASET_CACHE="${DATASET_CACHE:-$ROOT/runs/prompts_e2e_pipeline.jsonl}"
  export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/qwen25-1p5b-grpo-e2e-pipeline}"
  export BASELINE_JSON="${BASELINE_JSON:-$ROOT/runs/eval_baseline_e2e.json}"
  export POST_JSON="${POST_JSON:-$ROOT/runs/eval_post_e2e.json}"
  export IMPROVE_MD="${IMPROVE_MD:-$ROOT/runs/improvement_table_e2e.md}"
  echo "[e2e-pipeline] train 5 ep/task, eval 2 ep/task, baseline+post+compare; DATASET_CACHE=$DATASET_CACHE OUTPUT_DIR=$OUTPUT_DIR"
else
  # Full run: 250 ep/task, lean eval, GRPO group_size=4 in train for speed. 1 epoch.
  export EPISODES_PER_TASK_EASY="${EPISODES_PER_TASK_EASY:-250}"
  export EPISODES_PER_TASK="${EPISODES_PER_TASK:-250}"
  export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1.0}"
  export EVAL_EPISODES="${EVAL_EPISODES:-30}"
  export EVAL_DURING_TRAIN="${EVAL_DURING_TRAIN:-10}"
  export NO_BASELINE_EVAL="${NO_BASELINE_EVAL:-0}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/qwen25-1p5b-grpo}"
DATASET_CACHE="${DATASET_CACHE:-$ROOT/runs/prompts.jsonl}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/runs/eval_baseline.json}"
POST_JSON="${POST_JSON:-$ROOT/runs/eval_post.json}"
IMPROVE_MD="${IMPROVE_MD:-$ROOT/runs/improvement_table.md}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
# Hub push: set PUSH_TO_HUB=1 + HUB_MODEL_ID=...; prompts JSONL dataset: PUSH_PROMPTS_TO_HUB=1, HUB_PROMPTS_REPO, etc.
HUB_MODEL_ID="${HUB_MODEL_ID:-Adityahars/sanskrit-qwen-grpo}"
HUB_PROMPTS_REPO="${HUB_PROMPTS_REPO:-Adityahars/sanskrit-grpo-prompts}"
HUB_PROMPTS_PATH_IN_REPO="${HUB_PROMPTS_PATH_IN_REPO:-data/prompts.jsonl}"

echo "[info] repo root: $ROOT"
echo "[info] ENV_URL=$ENV_URL"
echo "[info] MODEL_ID=$MODEL_ID OUTPUT_DIR=$OUTPUT_DIR"
echo "[info] EPISODES_PER_TASK_EASY=$EPISODES_PER_TASK_EASY EPISODES_PER_TASK(hard)=$EPISODES_PER_TASK"
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  echo "[info] PUSH_TO_HUB=1 -> adapter will be pushed to Hub as $HUB_MODEL_ID after training"
fi
if [[ "${PUSH_PROMPTS_TO_HUB:-0}" == "1" ]]; then
  echo "[info] PUSH_PROMPTS_TO_HUB=1 -> prompts JSONL -> datasets/$HUB_PROMPTS_REPO ($HUB_PROMPTS_PATH_IN_REPO)"
fi

echo "[info] pip install (SanskritEnv + training)..."
python -m pip install -q -U pip
python -m pip install -q -r "$ROOT/requirements.txt"
python -m pip install -q -r "$ROOT/training/requirements-train.txt"

mkdir -p "$(dirname "$BASELINE_JSON")" "$(dirname "$POST_JSON")" "$(dirname "$DATASET_CACHE")" "$OUTPUT_DIR"

# JSONL: pull from dataset repo, or build from local data/ (no HTTP), then optionally push dataset to Hub.
if [[ "${PULL_PROMPTS_FROM_HUB:-0}" == "1" ]]; then
  echo "[info] pull prompts: Hub $HUB_PROMPTS_REPO / $HUB_PROMPTS_PATH_IN_REPO -> $DATASET_CACHE"
  python "$ROOT/training/upload_prompts_to_hub.py" download \
    --output "$DATASET_CACHE" \
    --repo-id "$HUB_PROMPTS_REPO" \
    --path-in-repo "$HUB_PROMPTS_PATH_IN_REPO"
elif [[ "${USE_HTTP_DATASET:-0}" != "1" ]]; then
  if [[ "${SKIP_DATASET_REBUILD:-0}" != "1" ]] || [[ ! -f "$DATASET_CACHE" ]]; then
    echo "[info] collect_prompts_jsonl.py (local data/ -> $DATASET_CACHE)..."
    python "$ROOT/training/collect_prompts_jsonl.py" \
      --output "$DATASET_CACHE" \
      --model-id "$MODEL_ID" \
      --episodes-per-task "$EPISODES_PER_TASK" \
      --episodes-per-task-easy "$EPISODES_PER_TASK_EASY" \
      --base-seed 42 \
      --difficulty "${DIFFICULTY:-auto}"
  else
    echo "[info] SKIP_DATASET_REBUILD=1 and cache exists: $DATASET_CACHE"
  fi
else
  echo "[info] USE_HTTP_DATASET=1: train will HTTP-collect if no cache at train time"
fi

if [[ "${PUSH_PROMPTS_TO_HUB:-0}" == "1" ]] && [[ -f "$DATASET_CACHE" ]]; then
  echo "[info] upload prompts JSONL to Hub dataset $HUB_PROMPTS_REPO..."
  python "$ROOT/training/upload_prompts_to_hub.py" upload "$DATASET_CACHE" \
    --repo-id "$HUB_PROMPTS_REPO" \
    --path-in-repo "$HUB_PROMPTS_PATH_IN_REPO" \
    --message "GRPO prompts JSONL (HF Job)"
fi

echo "[info] wait for remote env /health..."
for i in $(seq 1 90); do
  if python -c "import requests; r=requests.get('$ENV_URL/health', timeout=30); assert r.status_code==200" 2>/dev/null; then
    echo "[info] env healthy after ${i} attempt(s)"
    break
  fi
  echo "[info] waiting... ($i/90)"
  sleep 2
done
if ! python -c "import requests; r=requests.get('$ENV_URL/health', timeout=30); assert r.status_code==200"; then
  echo "ERROR: $ENV_URL/health did not return 200."
  exit 1
fi

# At least 1 episode/task so the HF-downloaded model actually runs generate+step (not only load).
EVAL_RUN_EPISODES="${EVAL_EPISODES:-30}"
if [[ "$EVAL_RUN_EPISODES" =~ ^[0-9]+$ ]] && [[ "$EVAL_RUN_EPISODES" -lt 1 ]]; then
  EVAL_RUN_EPISODES=1
  echo "[info] EVAL_EPISODES<1; using 1 for evaluate.py (model must run >=1 episode per task)"
fi

BASE_EVAL_ARGS=()
if [[ "$NO_BASELINE_EVAL" == "1" ]]; then
  echo "[info] skip standalone baseline"
else
  echo "[info] baseline evaluate.py..."
  python "$ROOT/training/evaluate.py" \
    --env-url "$ENV_URL" \
    --base-model "$MODEL_ID" \
    --episodes-per-task "$EVAL_RUN_EPISODES" \
    --base-seed 10000 \
    --output "$BASELINE_JSON" \
    --label baseline-untrained
  BASE_EVAL_ARGS=("$BASELINE_JSON")
fi

TRAIN_EXTRAS=()
if [[ "$NO_BASELINE_EVAL" == "1" ]]; then
  TRAIN_EXTRAS+=(--no-baseline-eval)
fi
if [[ "$EVAL_DURING_TRAIN" == "0" ]]; then
  TRAIN_EXTRAS+=(--eval-episodes-per-task 0)
else
  TRAIN_EXTRAS+=(--eval-episodes-per-task "$EVAL_DURING_TRAIN")
fi
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  TRAIN_EXTRAS+=(--push-to-hub --hub-model-id "$HUB_MODEL_ID")
fi

echo "[info] train_grpo.py..."
python "$ROOT/training/train_grpo.py" \
  --env-url "$ENV_URL" \
  --model-id "$MODEL_ID" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --episodes-per-task-easy "$EPISODES_PER_TASK_EASY" \
  --base-seed 42 \
  --dataset-cache "$DATASET_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --group-size "${GROUP_SIZE:-4}" \
  --per-device-batch "${PER_DEVICE_BATCH:-2}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --epochs "$TRAIN_EPOCHS" \
  --lr "${LR:-5e-6}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-96}" \
  --eval-base-seed 10000 \
  "${TRAIN_EXTRAS[@]}"

echo "[info] post-training evaluate.py..."
python "$ROOT/training/evaluate.py" \
  --env-url "$ENV_URL" \
  --base-model "$MODEL_ID" \
  --adapter "$OUTPUT_DIR" \
  --episodes-per-task "$EVAL_RUN_EPISODES" \
  --base-seed 10000 \
  --output "$POST_JSON" \
  --label post-train

if [[ ${#BASE_EVAL_ARGS[@]} -gt 0 ]]; then
  echo "[info] compare_evals.py..."
  python "$ROOT/training/compare_evals.py" \
    "$BASELINE_JSON" \
    "$POST_JSON" \
    --markdown "$IMPROVE_MD"
  cat "$IMPROVE_MD"
else
  echo "[info] skip compare (no baseline json)"
fi

echo "[done] artifacts under $ROOT/runs/"

# Upload adapter to Hub (if PUSH_TO_HUB=1).
# Re-upload is a fast no-op when train_grpo --push-to-hub already pushed the same files.
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  echo "[info] uploading adapter to Hub: $HUB_MODEL_ID ..."
  python - <<'PYEOF'
import os, sys
from pathlib import Path
output_dir = os.environ.get("OUTPUT_DIR", "/tmp/sanskrit-env/runs/qwen25-1p5b-grpo")
hub_model_id = os.environ.get("HUB_MODEL_ID", "Adityahars/sanskrit-qwen-grpo")
hf_token = os.environ.get("HF_TOKEN")
if not Path(output_dir).exists():
    print(f"[warn] OUTPUT_DIR={output_dir} not found; skipping hub upload", file=sys.stderr)
    sys.exit(0)
try:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_model_id, exist_ok=True, private=False)
    api.upload_folder(
        repo_id=hub_model_id,
        folder_path=output_dir,
        repo_type="model",
        commit_message="GRPO LoRA adapter upload from HF Job",
    )
    print(f"[ok] adapter uploaded to https://huggingface.co/{hub_model_id}")
except Exception as e:
    print(f"[warn] hub upload failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
fi
