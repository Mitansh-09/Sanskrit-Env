# Training with the deployed Hugging Face environment and account credits

This guide assumes you use the **hosted SanskritEnv** at `https://adityahars-sanskrit-env.hf.space` (not a local `localhost` server) for `train_grpo.py` and `evaluate.py` HTTP calls. Training **still runs on a GPU** (Colab, HF Jobs, or your own machine); only the **environment** is remote.

## Security: tokens

- **Never** commit a Hugging Face token, paste it into the repo, or share it in chat. If a token is exposed, open [Access Tokens](https://huggingface.co/settings/tokens), **revoke** it, and create a new one (read for Hub pulls; **write** if you push models).
- Set the token only via environment: `HF_TOKEN`, Colab Secrets, or GitHub / HF Job secrets. The scripts read it via `huggingface_hub.login` or the Hub API when you push adapters.

## What “HF credits” can pay for

- **Hugging Face subscription / balance**: Depending on your plan, you may have **monthly compute credits** or a **balance** usable for paid hardware (e.g. **Spaces** with A100, **Hugging Face Jobs** with GPU, or other hosted GPU products). Check [Settings → Billing](https://huggingface.co/settings/billing) and current product docs; naming and prices change over time.
- **Credits do not replace a training GPU** for the policy model: the env Space only serves `/reset` and `/step`. You still need a **separate** GPU session where PyTorch and `train_grpo.py` run. Credits are typically used to **keep the Space GPU awake** (if the Space is GPU-backed) and/or to run **your training job** on HF-managed GPU (Jobs) or to pay for **Colab Pro** separately (that is Google, not HF).

**Important:** Hitting a **free CPU Space** for thousands of `step` calls will be **slow** and may hit limits. If your training is heavy, prefer: **remote env on a small always-on plan** *or* accept longer wall time *or* run env locally. The defaults in this repo point at your **deployed** URL as requested.

## One-line mental model

| Process | Where it runs | Paid by |
|--------|----------------|--------|
| **SanskritEnv** (FastAPI) | `https://adityahars-sanskrit-env.hf.space` | Your Space (HF billing / credits if the Space uses paid hardware) |
| **GRPO + Qwen** (GPU) | Colab A100, HF Job `a100-large`, or local | Colab / HF Jobs / your cloud |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Download Qwen, push LoRA, HF CLI `login` (never commit) |
| `ENV_URL` or `HF_SPACE_URL` | Base URL for env HTTP API (no trailing path). Default in code: `https://adityahars-sanskrit-env.hf.space` if unset. |
| `MODEL_ID` | e.g. `Qwen/Qwen2.5-1.5B-Instruct` |

**Windows (PowerShell):** there is no `export` command. Set variables for the current session with `$env:HF_TOKEN = "hf_..."` (same for `$env:ENV_URL`). In **Command Prompt** use `set HF_TOKEN=hf_...`.

## Verify the remote env is up

```bash
curl -sS "https://adityahars-sanskrit-env.hf.space/health"
```

If the Space was sleeping, the first request can take **30–120s**. Re-run until you get HTTP 200.

## Option A: Google Colab (A100) + remote env

1. Open `training/Sanskrit_GRPO_Colab.ipynb` on Colab, select **A100** (Pro/Pro+).
2. In **Secrets**, add `HF_TOKEN` (and optionally set `ENV_URL` if you use a different Space). The notebook does **not** start `localhost` uvicorn; it uses the **deployed** Space.
3. Run all cells in order. Dataset collection and training will call your Space over HTTPS; expect **higher latency** than localhost.

**Tip:** If you see timeouts, increase patience on first `/health` or run the curl check in a cell before the dry-run.

## Option B: Hugging Face Jobs (GPU) + remote env

Use this when you want training to run on **Hugging Face managed GPU** (billed to your [Pro/Team](https://huggingface.co/docs/hub/jobs) plan / credits). The job **clones the repo** and runs `training/scripts/hf_job_entrypoint.sh`: pip install → wait for `ENV_URL/health` → optional baseline `evaluate.py` → `train_grpo.py` → post `evaluate.py` → `compare_evals.py`.

### Prerequisites

- `pip install -U "huggingface_hub>=0.26"` (or your stack’s version that exposes `run_job`)
- A Hugging Face **access token** with read access to models; never commit it
- A **public** git URL for this repo, **or** a private URL with a GitHub PAT only in your shell: `SANSKRIT_ENV_REPO_URL=https://x-access-token:TOKEN@github.com/org/repo.git` (do not save that string in the repo)
- [Jobs overview and pricing](https://huggingface.co/docs/hub/jobs) — set **`timeout` to several hours** for full runs; the Hub default is often **30 minutes** if omitted

### Submit from your laptop (recommended)

```bash
cd /path/to/sanskrit-env
export HF_TOKEN=hf_...   # or run: hf auth login
# Quick wiring test (minutes):
python training/submit_hf_job.py --smoke --flavor a10g-small --timeout 45m
# Full GRPO (long; increase --timeout as needed, e.g. 6h+):
python training/submit_hf_job.py --flavor a100-large --timeout 6h
```

The script calls [`run_job()`](https://huggingface.co/docs/huggingface_hub/guides/jobs) with `secrets={"HF_TOKEN": ...}` and passes `ENV_URL` to the container. It prints a **Job URL** where you can follow logs.

**429 on `/whoami-v2`:** The Hub rate-limits identity checks. Submitting several jobs in a row can hit this. Set your Hub **username** once so the client does not call `whoami` for every submit:

- Bash: `export HF_JOB_NAMESPACE=YourHFUsername`
- PowerShell: `$env:HF_JOB_NAMESPACE = "YourHFUsername"`

Or pass `python training/submit_hf_job.py --namespace YourHFUsername ...`. Wait a few minutes if you still see 429, then retry.

**Forwarded environment variables** (optional): set any of `EPISODES_PER_TASK`, `TRAIN_EPOCHS`, `MODEL_ID`, `OUTPUT_DIR`, `GROUP_SIZE`, etc., before `submit_hf_job.py`; they are passed into the job as plain `env` (not secrets). Training knobs are the same as `train_grpo.py`.

**Smoke test:** `SMOKE_TEST=1` (or `python training/submit_hf_job.py --smoke`) uses tiny episode counts and `TRAIN_EPOCHS=0.1` to verify connectivity to the env and Hub download.

### Artifacts in the job

Checkpoints and JSON live **inside the job filesystem** and are **lost when the job ends** unless you add a [volume](https://huggingface.co/docs/huggingface_hub/guides/jobs#mount-a-volume) (e.g. Hub bucket) or `huggingface_hub` upload in a follow-up step. For production, set `--push-to-hub` in `train_grpo.py` or upload `OUTPUT_DIR` with `HfApi().upload_folder` after training.

## Option C: Your laptop / VM (GPU) + remote env

```bash
cd /path/to/sanskrit-env
pip install -r training/requirements-train.txt
export HF_TOKEN=...          # for model download / hub push
export ENV_URL=https://adityahars-sanskrit-env.hf.space
python training/evaluate.py --episodes-per-task 30 --output runs/eval_baseline.json
python training/train_grpo.py --output-dir runs/qwen25-1p5b-grpo
python training/evaluate.py --adapter runs/qwen25-1p5b-grpo --episodes-per-task 30 --output runs/eval_post.json
python training/compare_evals.py runs/eval_baseline.json runs/eval_post.json
```

`train_grpo.py` already defaults `ENV_URL` from `ENV_URL` → `HF_SPACE_URL` → the Space URL above, so you can omit `export` if you use that Space.

## After training

- Artifacts: `runs/.../metrics_history.json`, `eval_baseline.json`, `eval_post.json`, optional `improvement_table.md` from `compare_evals.py`.
- Push the adapter with `HF_TOKEN` (write) via `HfApi().upload_folder` or `train_grpo.py --push-to-hub` if you enable it.

## If training feels too slow

The remote env adds **per-step** HTTP latency. To speed up: cache prompts (`--dataset-cache` in `train_grpo.py`), reduce `--episodes-per-task` for debugging, or run the **env server** on the same machine as the GPU (the opposite of this doc’s “always HF URL” request).
