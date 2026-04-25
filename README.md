# SanskritEnv - RL Environment for Sanskrit Manuscript Interpretation

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue?logo=huggingface)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/Adityahars/Sanskrit-env)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

## The Real-World Problem

India has approximately 10 million Sanskrit manuscripts. The Union Budget 2025-26 allocated INR 60 crore for digitization under the Gyan Bharatam Mission. Digitization, however, is only the first bottleneck.

The deeper bottleneck is translation and interpretation. Expert Sanskrit scholars with manuscript-level training are retiring faster than new scholars are being trained. Most current AI systems fail on this problem because they process words independently, without a structured world model of how Sanskrit ambiguity propagates across linguistic layers.

## The Four Linguistic Layers

| Layer | Core ambiguity | Why this is hard |
|---|---|---|
| Lexical | Polysemy (for example, `agni` shifts meaning across Ayurveda, ritual, and philosophy) | Correct answer depends on domain-grounded context, not dictionary frequency |
| Phonological | Sandhi splitting admits multiple valid parses | Requires jointly reasoning over grammar, meter, and local semantics |
| Morphological | Samasa compounds must be classified before interpretation | Type errors upstream corrupt downstream parsing |
| Discourse | Pronoun/antecedent links span multiple verses with implicit references | Requires long-range coherence and memory across steps |

## How SanskritEnv Solves This

SanskritEnv defines six escalating tasks:

- Tasks 1-4 are single-step MCQ skill drills.
- Task 5 is a full tool-use POMDP (manuscript restoration).
- Task 6 is a long-horizon full session chaining all five skills with cross-phase consistency scoring.

| Task | ID | Type |
|---|---|---|
| 1 | `glossary_anchoring` | Single-step lexical disambiguation |
| 2 | `sandhi_resolution` | Single-step phonological parsing |
| 3 | `referential_coherence` | Multi-step discourse tracking |
| 4 | `samasa_classification` | Single-step morphological classification |
| 5 | `manuscript_restoration` | Tool-use POMDP with commit action |
| 6 | `full_manuscript_session` | Long-horizon phase chain with consistency checks |

## Reward Structure - DETAILED

Wrong answers always return exactly `0.0`. This is intentional for GRPO compatibility, not a floor.

`A_i = (r_i - mean(group)) / (std(group) + epsilon)`

With a floor around `0.5`, group standard deviation typically collapses near `0.10-0.15`, producing weak training signal. With true zero for wrong answers, standard deviation typically rises near `0.35-0.45`, yielding stronger group-relative advantages.

Reward shaping in the environment:

- Wrong (`raw = 0.0`) -> `0.0` (unchanged)
- `0.0 < raw < 0.40` -> linear identity mapping (`shaped = raw`)
- `0.40 <= raw <= 1.00` -> `shaped = 0.50 + (raw - 0.40) * (0.45 / 0.60)`

### Tasks 1-4 (Single-Step MCQ)

| Outcome | Raw | Shaped |
|---|---|---|
| Full credit | 1.00 | 0.95 |
| Partial credit | 0.40 | 0.50 |
| Adjacent sandhi | 0.25 | 0.25 |
| Wrong | 0.00 | 0.00 |

### Task 5 (Manuscript Restoration POMDP)

Per-step tool reward:

`tool_reward = relevance_bonus + workflow_bonus - redundancy_penalty`

- PRIMARY tool for episode type: `+0.08`
- SECOND tool:
  - If PRIMARY already used: `+0.05`
  - Otherwise: `+0.04`
- Supporting tool: `+0.04`
- Redundant call (same tool + same input): `-0.05`
- Irrelevant tool example (`meter_checker` on prose): `-0.05`

Workflow pair bonuses:

- `sandhi_parser -> meter_checker`: `+0.05`
- `lexicon_lookup -> commentary_fetch`: `+0.05`
- `witness_compare -> referent_tracker`: `+0.03`

Terminal commit reward:

`terminal_reward = r_correctness * M_evidence - P_budget`

Where:

- `M_evidence = 0.60 + 0.40 * (distinct_relevant_tools_used / tools_needed)` (range `0.60` to `1.00`)
- `P_budget = 0.10 * max(0, steps_used - ideal_steps) / tool_budget` (range `0.00` to `0.10`)

Critical invariant:

- Wrong commit -> `0.0` regardless of evidence gathered.
- Episode score is terminal commit reward only.
- Tool rewards are dense intermediate signals and are not added into final episode score.

### Task 6 (Full Manuscript Session)

Session score:

`session_score = mean(phase_rewards) - consistency_penalty + consistency_bonus`

- Each contradiction between phases: `-0.05` per violation
- Zero violations across phases: `+0.05` bonus
- Violation rule: if restoration-phase final interpretation contradicts earlier phase answers, it counts as a violation

### Tool Relevance Matrix (Task 5)

| Episode type | `lexicon_lookup` | `sandhi_parser` | `meter_checker` | `commentary_fetch` | `witness_compare` | `referent_tracker` |
|---|---|---|---|---|---|---|
| glossary | PRIMARY | support | none | SECOND | support | none |
| sandhi | support | PRIMARY | SECOND | none | support | none |
| samasa | support | PRIMARY | SECOND | none | none | none |
| coherence | support | support | none | support | support | PRIMARY |

## The Six Philological Tools

| Tool | Input type | Returns | PRIMARY for episode types |
|---|---|---|---|
| `lexicon_lookup` | Sanskrit lemma/term | Domain-conditioned meanings and glosses | glossary |
| `sandhi_parser` | Compound form | Candidate sandhi splits with rule-level structure | sandhi, samasa |
| `meter_checker` | Candidate split/text span | Meter compatibility signal | none (SECOND in sandhi/samasa) |
| `commentary_fetch` | Term, phrase, or verse reference | Commentary snippets linked to interpretation | none (SECOND in glossary) |
| `witness_compare` | Verse/manuscript locus | Variant witness readings and differences | none (support in glossary/sandhi/coherence) |
| `referent_tracker` | Pronoun/entity cue | Candidate antecedents and discourse links | coherence |

## Adaptive Difficulty Curriculum (Task 5)

Difficulty levels:

- Beginner: clean text, full commentary, `budget=8`
- Intermediate: 10% OCR noise, partial commentary, `budget=6`
- Hard: 25% OCR noise, partial commentary, `budget=5`
- Expert: 40% OCR noise, conflicting witnesses, no commentary, `budget=4`

Promotion and fallback:

- Escalation threshold: mean of last 10 scores `> 0.80` with at least 5 episodes
- De-escalation threshold: mean `< 0.45`

## Data Sources

| Text | Domain | Tasks |
|---|---|---|
| Sushruta Samhita | Ayurveda | 1, 5 |
| Bhagavad Gita | Philosophy / Vedanta | 1, 2, 4, 5 |
| Charaka Samhita | Ayurveda | 1, 3 |
| Ramayana | Narrative | 2, 3, 4, 5 |
| Arthashastra | Political philosophy | 1, 3, 4, 5 |
| Surya Siddhanta | Astronomy | 5 |
| Aryabhatiya | Astronomy | 1, 3 |

## Project Setup - Local Development

1. Clone the repository:

   ```bash
   git clone https://huggingface.co/spaces/Adityahars/Sanskrit-env
   cd sanskrit-env
   ```

2. Create and activate a Python 3.11+ virtual environment:

   ```bash
   python -m venv .venv
   # PowerShell
   .venv\Scripts\Activate.ps1
   # bash/zsh
   # source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Fill `.env` with your Cloudflare and/or HuggingFace credentials.

5. Start the server locally:

   ```bash
   python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
   ```

6. Validate health endpoint:

   ```bash
   curl http://localhost:7860/health
   ```

7. Run the test agent against localhost:

   ```bash
   python test_agent.py --local --task all --episodes 5
   ```

## Project Setup - Docker

Build image:

```bash
docker build -t sanskrit-env:local .
```

Run container:

```bash
docker run --rm -p 7860:7860 sanskrit-env:local
```

Check health:

```bash
curl http://localhost:7860/health
```

## Running `inference.py` (Submission Script)

`inference.py` is the submission artifact and is designed to follow OpenEnv output constraints strictly:

- Stdout contains only `[START]`, `[STEP]`, and `[END]` lines.
- Debug/error details are written to stderr.
- Model/router settings are pulled from environment variables.

Required environment variables:

- `HF_TOKEN`
- `API_BASE_URL` (for example `https://router.huggingface.co/v1`)
- `MODEL_NAME` (for example `Qwen/Qwen2.5-72B-Instruct`)

Example:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Running `test_agent.py` (Development Evaluation)

`test_agent.py` is the development/evaluation runner and supports both remote and local environments:

- `--env-url` defaults to the HuggingFace Space URL
- `--local` is a shortcut for `--env-url http://localhost:7860`

Examples:

Run all tasks against HuggingFace Space (default):

```bash
python test_agent.py --task all --episodes 5
```

Run all tasks against localhost:

```bash
python test_agent.py --local --task all --episodes 5
```

Run a single task with verbose multi-step output:

```bash
python test_agent.py --task referential_coherence --episodes 1 --verbose
```

Run manuscript restoration at hard difficulty with 10 episodes:

```bash
python test_agent.py --task manuscript_restoration --difficulty hard --episodes 10
```

## Baseline Results

| Model | Episodes | Glossary | Sandhi | Samasa | Coherence | Restoration | Session | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 20 | 1.000 | 1.000 | 0.970 | 0.700 | pending | pending | 0.917 |
| `@cf/meta/llama-3.1-70b-instruct` | 5 | 1.000 | 1.000 | 1.000 | 0.700 | pending | pending | 0.925 |
| `@cf/meta/llama-3.1-8b-instruct` | 5 | 1.000 | 1.000 | 1.000 | 0.280 | pending | pending | 0.820 |
| `@cf/meta/llama-3.2-3b-instruct` | 5 | 1.000 | 0.800 | 0.480 | 0.140 | pending | pending | 0.605 |
