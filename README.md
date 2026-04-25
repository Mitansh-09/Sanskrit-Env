---
title: SanskritEnv
emoji: 📜
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
short_description: RL environment for Sanskrit manuscript interpretation
huggingFace_url: https://huggingface.co/spaces/Adityahars/Sanskrit-env
---

# SanskritEnv

> An OpenEnv-compatible RL environment that trains AI agents to act like Sanskrit
> philologists — using deterministic tools to gather evidence and interpret
> ambiguous manuscript passages. Five tasks from simple MCQ to a full tool-use
> POMDP with adaptive difficulty.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue?logo=huggingface)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/Adityahars/Sanskrit-env)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

---

## The Problem

India possesses an estimated **1 crore Sanskrit manuscripts** — the largest manuscript collection of any civilisation on Earth. The **Union Budget 2025-26** allocated ₹60 crore to digitize these under the **Gyan Bharatam Mission**. Digitization is accelerating, but translation is not.

The bottleneck is a collapse in human expertise. Trained Sanskrit scholars capable of reading classical manuscripts are retiring faster than new scholars can replace them. Current AI tools fail because they treat each word independently — they have no world model of how Sanskrit's four linguistic layers interact:

| Layer | Problem | Why AI fails |
|-------|---------|-------------|
| Lexical | A single term (e.g. *agni*) has 4–6 domain-specific meanings | No contextual disambiguation |
| Phonological | Compound words (*sandhi*) have multiple valid splits | Requires grammatical + contextual reasoning |
| Morphological | Compound words (samāsa) must be classified before parsing | Requires grammatical meta-knowledge |
| Discourse | Pronouns span multiple verses with no antecedent markers | Requires cross-sentence coreference tracking |

---

## How SanskritEnv Solves This

SanskritEnv trains AI agents to act like Sanskrit philologists through six escalating tasks:

**Tasks 1–4: Skill Drills** — single-step MCQ tasks that build foundational Sanskrit skills.
**Task 5: Manuscript Restoration** — a full tool-use POMDP where the agent must gather evidence using six philological tools before committing an interpretation.
**Task 6: Full Manuscript Session** — a long-horizon task chaining all 5 challenges on a single passage with cross-phase consistency scoring.

```
Passage → [lexicon_lookup] → [sandhi_parser] → [meter_checker] → COMMIT answer
                ↓                    ↓                  ↓
           glossary data        split options       meter check
```

| Task | ID | Difficulty | Steps | Core challenge |
|------|----|-----------|-------|----------------|
| Glossary Anchoring | `glossary_anchoring` | Easy | 1 | Domain-specific term disambiguation |
| Sandhi Resolution | `sandhi_resolution` | Medium | 1 | Phonological compound splitting |
| Samāsa Classification | `samasa_classification` | Medium | 1 | Grammatical compound type ID |
| Referential Coherence | `referential_coherence` | Hard | 4–7 | Cross-verse pronoun tracking |
| Manuscript Restoration | `manuscript_restoration` | Expert | 4–10 | Tool-use POMDP with evidence |
| Full Manuscript Session | `full_manuscript_session` | Master | 8–15 | 5-phase chain with consistency penalty |

### Adaptive Difficulty Curriculum

Task 5 uses a self-paced curriculum (Bengio et al. 2009):
- **Beginner**: Clean text, full commentary, budget=8
- **Intermediate**: 10% OCR noise, partial commentary, budget=6
- **Hard**: 25% OCR noise, manuscript witnesses, budget=5
- **Expert**: 40% OCR noise, conflicting witnesses, no commentary, budget=4

Difficulty escalates when `mean(last 10 scores) > 0.80` and de-escalates when `< 0.45`.

---

## Reward Function

### Critical Design: Wrong → 0.0 (GRPO-Compatible)

Wrong answers return **exactly 0.0** — no floor. This gives GRPO the variance it needs:

```
GRPO advantage: A_i = (r_i - mean(r_group)) / (std(r_group) + ε)

With old 0.50 floor: std ≈ 0.10–0.15 → weak gradients → no learning
With true zero:      std ≈ 0.35–0.45 → strong gradients → meaningful training
```

### Tasks 1–4 (Single-Step MCQ)

| Outcome | Raw | Shaped |
|---------|-----|--------|
| Correct | 1.00 | 0.95 |
| Partial credit | 0.40 | 0.50 |
| Adjacent sandhi | 0.25 | 0.25 |
| Wrong | 0.00 | **0.00** |

### Task 5 (Manuscript Restoration POMDP)

**Per-step tool reward:**
```
r_tool = relevance_bonus + workflow_bonus - redundancy_penalty
```
- PRIMARY tool: +0.08 | Supporting: +0.04 | Redundant: -0.05

**Terminal commit reward:**
```
r_terminal = r_correctness × M_evidence - P_budget

M_evidence = 0.60 + 0.40 × (distinct_relevant_tools / tools_needed)
P_budget   = 0.10 × max(0, steps_used - ideal_steps) / budget

Evidence never rescues wrong answers: wrong → 0.0 regardless of tools used.
```

This implements potential-based reward shaping (Ng et al. 1999) that preserves the optimal policy while guiding toward evidence-gathering behavior.

### Task 6 (Full Manuscript Session)

**Cross-Phase Consistency:**
The agent completes 5 phases (glossary, sandhi, samāsa, coherence, restoration) on a single passage.
```
r_session = mean(r_phases) - consistency_penalty + consistency_bonus

consistency_penalty = 0.05 × violations
consistency_bonus   = 0.05 (if violations == 0)
```
A violation occurs if the agent's final interpretation in the restoration phase contradicts an answer it selected in an earlier phase. This penalizes "reasoning drift" across long horizons.

---

## Baseline Benchmark Matrix

| Model | Episodes | Glossary | Sandhi | Samāsa | Coherence | Restoration | Overall |
|-------|----------|----------|--------|--------|-----------|-------------|---------|
| @cf/meta/llama-3.3-70b-instruct-fp8-fast | 20 | 1.000 | 1.000 | 0.970 | 0.700 | — | 0.917 |
| @cf/meta/llama-3.1-70b-instruct | 5 | 1.000 | 1.000 | 1.000 | 0.700 | — | 0.925 |
| @cf/meta/llama-3.1-8b-instruct | 5 | 1.000 | 1.000 | 1.000 | 0.280 | — | 0.820 |
| @cf/meta/llama-3.2-3b-instruct | 5 | 1.000 | 0.800 | 0.480 | 0.140 | — | 0.605 |

---

## Six Philological Tools (Task 5)

| Tool | Input | Returns |
|------|-------|---------|
| `lexicon_lookup` | Sanskrit term | Domain-specific meanings from episode glossary |
| `sandhi_parser` | Compound word | All valid phonological splits with rules |
| `meter_checker` | Proposed split | Whether split preserves verse meter |
| `commentary_fetch` | Term or verse ID | Medieval commentary fragment |
| `witness_compare` | Verse ID | Variant readings from manuscript witnesses |
| `referent_tracker` | Pronoun | Possible antecedents with grammatical info |

All tools are **deterministic** — they read from pre-annotated episode data, not computed on the fly.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Local development

```bash
git clone https://huggingface.co/spaces/Adityahars/Sanskrit-env
cd sanskrit-env
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Validate (separate terminal)
openenv validate --url http://localhost:7860
```

### Docker

```bash
docker build -t sanskrit-env:local .
docker run --rm -p 7860:7860 sanskrit-env:local
curl http://localhost:7860/health
```

### Run baseline

```bash
export CLOUDFLARE_API_TOKEN=your_token
export CLOUDFLARE_ACCOUNT_ID=your_account_id

python baseline.py                                    # all tasks
python baseline.py --task referential_coherence        # single task
python baseline.py --task manuscript_restoration --difficulty hard
```

### Run test agent

```bash
python test_agent.py --task manuscript_restoration --episodes 3 --difficulty beginner
python test_agent.py --task all --episodes 5
python test_agent.py --provider cloudflare --model "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
```

### Run inference.py

```bash
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Training (GRPO)

### Recommended setup

- **Model**: Qwen/Qwen2.5-1.5B-Instruct or google/gemma-2-2b-it
- **Framework**: HuggingFace TRL GRPO
- **Key insight**: The evidence_use metric should increase during training

### Expected reward curve

1. **Episodes 1–5**: All rewards near 0.0 (agent guesses randomly)
2. **Episodes 5–20**: Evidence-gathering behavior emerges (tools get used)
3. **Episodes 20+**: Correct+evidence patterns form (rewards 0.75–0.95)

---

## Why RL, Not SFT

| Aspect | SFT | RL (SanskritEnv) |
|--------|-----|-----------------|
| Learning signal | Static dataset | Interactive environment |
| Evidence gathering | Memorized | Learned through exploration |
| Generalization | To training passages only | To unseen passages |
| Key metric | Accuracy | Accuracy + evidence quality |

SFT on a static dataset teaches memorization. RL on `manuscript_restoration` teaches the evidence-gathering **workflow**. The trained agent will generalize to unseen passages; an SFT agent will not. The `evidence_use` metric (distinct tools used / tools needed) is the key indicator.

---

## Grader Design — No LLM, No BLEU

All graders are fully deterministic:
- **No LLM judge calls** — reproducible across runs
- **No BLEU/ROUGE** — unreliable for Sanskrit free word order
- **Exact string match** against pre-annotated answer tables

Two runs with the same seed produce identical scores.

---

## Data Sources

| Text | Domain | Tasks |
|------|--------|-------|
| Sushruta Samhita | Ayurveda | 1, 5 |
| Bhagavad Gita | Vedanta philosophy | 1, 2, 4, 5 |
| Charaka Samhita | Ayurveda | 1, 3 |
| Ramayana | Narrative | 2, 3, 4, 5 |
| Arthashastra | Political philosophy | 1, 3, 4, 5 |
| Surya Siddhanta | Astronomy | 5 |
| Aryabhatiya | Astronomy | 1, 3 |

---

## Contributing

Contributions welcome. Priority areas:

1. **More Task 5 episodes** — manuscript restoration passages with tool data
2. **New domains** — Jyotisha, Natya Shastra, Vedic hymns
3. **Harder sandhi cases** — anusvara, visarga, vowel coalescence
4. **More samāsa episodes** — Dvigu and Avyayibhava patterns
5. **Multi-language targets** — Hindi or regional language translations

---

## Citation

```bibtex
@misc{sanskritenv2026,
  title   = {SanskritEnv: A Reinforcement Learning Environment for Sanskrit Manuscript Interpretation},
  author  = {Meta\_Mesh},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Adityahars/Sanskrit-env},
  note    = {OpenEnv-compatible tool-use POMDP for structured linguistic ambiguity resolution}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Sanskrit texts used are in the public domain (composed before 1928).
Annotations, graders, and environment code are original to this project.

---

## Acknowledgements

- [Meta × HuggingFace OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [Gyan Bharatam Mission](https://indiaculture.gov.in) — the real-world problem this addresses
- [Monier-Williams Sanskrit Dictionary](https://www.sanskrit-lexicon.uni-koeln.de) — lexical reference
- [Sanskrit Sandhi Split Sighum](https://huggingface.co/datasets/chronbmm/sanskrit-sandhi-split-sighum) — annotated corpus reference
- [Itihasa](https://huggingface.co/datasets/rahular/itihasa) — annotated corpus reference
- Ng et al. (1999) "Policy Invariance Under Reward Transformations" — theoretical foundation for reward shaping
