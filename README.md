---
title: SanskritEnv
emoji: "📜"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
short_description: RL environment for Sanskrit manuscript interpretation
url: https://huggingface.co/spaces/Adityahars/Sanskrit-env
---

# SanskritEnv

**An OpenEnv-compatible RL environment for Sanskrit manuscript interpretation.**  
Train and evaluate AI agents on the task of resolving structural linguistic ambiguity in ancient Indian texts — a real bottleneck in ongoing digitization projects backed by the Indian government.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue?logo=huggingface)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/Adityahars/Sanskrit-env)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

---

## Real-World Impact

India possesses an estimated **1 crore Sanskrit manuscripts** written in over 80 scripts and 60 languages — the largest manuscript collection of any civilisation on Earth.

The **Union Budget 2025-26** allocated ₹60 crore to digitize over 1 crore manuscripts under the **Gyan Bharatam Mission**. As of 2025, metadata for 52 lakh manuscripts has been recorded — but only **1.3 lakh** have been uploaded online. Digitization is accelerating. Translation is not.

The reason is a collapse in human expertise. Trained Sanskrit scholars capable of reading classical manuscripts are retiring faster than new scholars can replace them. The Government's own **National Mission for Manuscripts** states directly:

> *"Scholars who can study and use manuscripts are fast disappearing and a new generation of scholars is not able to rise to the challenge."*

A nationwide survey launched in 2026 confirmed the crisis is active and growing. The ratio of trained scholars to digitized texts is estimated at **1:10,000** and widening every year.

The six linguistic problems that block automated translation of these manuscripts are:

1. A single Sanskrit term can carry 4–6 domain-specific meanings with no contextual signal (**lexical ambiguity**).
2. Compound words have multiple valid phonological splits with different meanings (**sandhi ambiguity**).
3. Compound words must be structurally classified before they can be parsed (**samāsa ambiguity**).
4. Pronouns and implicit subjects span multiple verses with no explicit antecedent markers (**referential ambiguity**).
5. Interpretation requires gathering and weighing philological tool evidence before committing to a reading (**evidential reasoning**).
6. All five layers must be resolved consistently across phases of a single document without contradiction (**compositional consistency**).

SanskritEnv is the **first RL environment** built to train agents on all six of these problems — using real passages from Ayurvedic, astronomical, philosophical, and narrative manuscripts currently sitting in India's national repositories.

---

## The Six Linguistic Layers

Projects like eGangotri have already rescued and scanned more than 60,000 rare texts and 1.4 crore pages. The bottleneck is not scanning technology — it is the shortage of scholars who can read classical Sanskrit across six major difficulty layers:

| Layer | Task | Problem | What blocks automation |
|---|---|---|---|
| Lexical | Glossary Anchoring | A single term (e.g. `agni`) has 4–6 domain-specific meanings | No contextual disambiguation |
| Phonological | Sandhi Resolution | Compound words have multiple valid phonological splits | Requires grammatical + contextual reasoning |
| Morphological | Samāsa Classification | Compound words must be classified before they can be parsed | Requires grammatical meta-knowledge |
| Discourse | Referential Coherence | Pronouns and implicit subjects span multiple verses | Requires cross-sentence coreference tracking |
| Evidential | Manuscript Restoration | Interpretation requires weighing philological tool evidence before committing | No automated evidence-gathering pipeline for classical texts |
| Compositional | Full Manuscript Session | All five layers must be resolved consistently across phases of a single document | Cross-layer contradictions collapse downstream parsing |

The first four layers are cited by Murugesh et al. (2019) *"A Survey of Sanskrit NLP"* as the primary obstacles to automated translation. SanskritEnv extends this benchmark with two higher-order layers — Evidential (tool-use POMDP) and Compositional (cross-phase consistency) — that no existing OpenEnv environment addresses.

---

## Environment Overview

SanskritEnv is a **decision environment**, not a translation model. At each step the agent receives a Sanskrit passage and must select the correct linguistic interpretation from deterministically-graded options.

### Architecture

```mermaid
flowchart TD
    A([📜 Sanskrit passage input]) --> B[Reset — new task]
    B --> C["🏛️ SanskritEnv<br/>OpenEnv-compatible env"]

    C --> D[Observation]
    D --> E["IAST & Devanāgarī<br/>English context · Domain info<br/>Candidate options"]

    E --> F{Task type?}

    F -->|"Tasks 1–4<br/>Single-step MCQ"| G["🤖 Agent<br/>LLM + rolling memory"]
    F -->|"Task 5<br/>Tool-use POMDP"| H["🤖 Agent<br/>LLM + tool calls"]
    F -->|"Task 6<br/>Full session"| I["🤖 Agent<br/>LLM + phase memory"]

    G --> J["ManuscriptAction<br/>selected_option"]
    H --> K["ManuscriptAction<br/>tool_call / commit"]
    I --> L["ManuscriptAction<br/>phase answer / commit"]

    J --> M["⚙️ env.step<br/>Evaluates choice<br/>computes reward"]
    K --> M
    L --> M

    M --> N{done?}

    N -->|False| O["🧠 Memory update<br/>Q&A appended to history"]
    O --> G
    O --> H
    O --> I

    N -->|True| P([🏆 Final episode score<br/>0.0 – 1.0])
```

Six tasks of escalating difficulty:

- **Tasks 1–4** are single-step MCQ skill drills targeting one linguistic layer each.
- **Task 5** is a full tool-use POMDP (manuscript restoration) with a commit action.
- **Task 6** is a long-horizon full session chaining all five skills with cross-phase consistency scoring.

| Task | ID | Type | Steps/Episode | Core Challenge |
|---|---|---|---|---|
| 1 | `glossary_anchoring` | Single-step MCQ | 1 | Domain-specific term disambiguation |
| 2 | `sandhi_resolution` | Single-step MCQ | 1 | Phonological compound splitting |
| 3 | `samasa_classification` | Single-step MCQ | 1 | Grammatical compound type identification |
| 4 | `referential_coherence` | Multi-step MCQ | 4–7 | Cross-verse pronoun tracking |
| 5 | `manuscript_restoration` | Tool-use POMDP | Variable | Evidence gathering + deterministic commit |
| 6 | `full_manuscript_session` | Long-horizon chain | Multi-phase | All skills + cross-phase consistency |

---

## Tasks in Detail — With Examples

Every task ships with real Sanskrit passages drawn from canonical texts (Bhagavad Gita, Charaka Samhita, Mahabharata, Ramayana, Kalidasa, etc.). The examples below are taken verbatim from the dataset so you can see exactly what an agent observes and why each task is non-trivial for an LLM.

### Task 1 — Glossary Anchoring (Lexical Disambiguation)

**Why it matters.** A single Sanskrit lemma can carry 4–6 domain-specific meanings. An LLM that picks the most frequent gloss from its pre-training distribution will misread Ayurvedic, astronomical, and philosophical passages where rarer technical senses dominate. This task forces the model to condition its lexical choice on the surrounding domain — exactly the behaviour translation pipelines need.

**Example episode (`t1_0001`, Ayurvedic context):**

> **Source (IAST):** `agnis tridosha-shamanam karoti`  
> **Devanāgarī:** अग्निस्त्रिदोषशमनं करोति  
> **Context:** Passage from Charaka Samhita discussing bodily processes.  
> **Target term:** `agni`  
> **Prompt:** *The term `agni` appears in this Ayurvedic passage. Which meaning is correct in this domain context?*

| # | Option | Verdict |
|---|---|---|
| A | fire (the physical element, one of *pancha-bhuta*) | wrong — surface gloss |
| B | digestive fire (*jatharagni*, the metabolic and digestive power) | **correct** |
| C | the deity Agni (Vedic god of fire and sacrifice) | wrong — wrong domain |
| D | heat sensation (*daha*, a symptom of pitta imbalance) | partial credit |

> **Why it traps an LLM.** The most-frequent translation of `agni` in any Sanskrit corpus is *fire* (option A). But here the surrounding term `tridosha-shamanam` ("pacifier of the three humours") fixes the domain as Ayurveda, where `agni` technically denotes the metabolic fire. Agents that reward-shape correctly learn to scan domain markers before committing to a gloss.

### Task 2 — Sandhi Resolution (Phonological Splitting)

**Why it matters.** Sanskrit fuses adjacent words via euphonic combination (sandhi). The same surface string can split multiple ways, each producing a different translation. Without correct splitting, every downstream parser (POS tagging, MT, semantic search) breaks.

**Example episode (`t2_0001`, Bhagavad Gita 10.6):**

> **Source (IAST):** `maharshayah sapta purve chatvaaro manavas tatha`  
> **Devanāgarī:** महर्षयः सप्त पूर्वे चत्वारो मनवस्तथा  
> **Context:** Krishna describes the divine sages.  
> **Compound:** `maharshayah`  
> **Prompt:** *How does the compound `maharshayah` correctly split in this context?*

| # | Option | Verdict |
|---|---|---|
| A | `maha + arshayah` (great + oblations/offerings) | wrong — invalid stem |
| B | `maha + rishayah` (great + sages) | **correct** |
| C | `mahar + shayah` (of greatness + sleeping) | wrong — implausible |
| D | `ma + harshayah` (not + causing joy) | wrong — wrong sandhi rule |

> **Why it traps an LLM.** Options B and C are both phonologically *legal*. Disambiguating requires combining grammar (visarga sandhi: `rishi-aḥ → rishayah`), context (the verse explicitly enumerates *rishis*), and meter awareness — exactly the cross-signal reasoning that one-shot LLMs fail at without a structured reward.

### Task 3 — Referential Coherence (Cross-Verse Pronoun Tracking)

**Why it matters.** Sanskrit narrative routinely drops antecedents across multiple verses, with pronouns like `tasya` ("his/her/its") spanning paragraph boundaries. Modern coreference systems trained on English news text fail badly here. This task is a **multi-step** episode (4–7 steps) where the agent reads a 7-verse passage and must answer a referential question.

**Example episode (`t3_0001`, *Savitri and Satyavan* — Mahabharata, Vana Parva):**

> **Verse 1.** *savitri nama raja-putri asvapati-sutaa shubhaa* — A princess named Savitri, the auspicious daughter of King Ashvapati,  
> **Verse 2.** *tapovane satyavanam dadarsha priya-darshanam* — saw Satyavan in the forest hermitage, a man of pleasing appearance.  
> **Verse 3.** *sa tam varayamasa pita uktva yamaya yasyatiti* — She chose him as her husband, though her father warned that he was fated to go to Yama.  
> **Verse 4.** *satyavan mriyate varshad ekena iti naradah praha* — Narada declared that Satyavan would die within one year.  
> **Verses 5–6.** *...At the year's end, Satyavan fell; she sat cradling his head.*  
> **Verse 7.** *yamah agacchhat tasya jivam netum* — Yama came to take **his** life.  
>  
> **Prompt:** *In verse 7, `tasya jivam` (his life) — whose life did Yama come to take?*

| # | Option | Verdict |
|---|---|---|
| A | Savitri (the princess) | wrong |
| B | Satyavan (the husband who fell) | **correct** |
| C | Ashvapati (Savitri's father) | wrong |
| D | Narada (the sage who prophesied) | wrong |

> **Why it traps an LLM.** The pronoun `tasya` in verse 7 is six verses removed from its antecedent `satyavanam` in verse 2. The most recent masculine noun in verse 6 is also Satyavan, but the model has to track **narrative role** (the dying husband), not just proximity. Reward shaping forces the agent to learn coreference signals that survive long contexts.

### Task 4 — Samāsa Classification (Compound Type Identification)

**Why it matters.** Sanskrit grammar recognises six structural types of nominal compounds (*samāsa*). The same surface form can belong to different types depending on context, and each type requires a different parse tree. Without classifying first, the parser cannot resolve the semantics.

**Example episode (`t4_0001`, Ramayana):**

> **Source (IAST):** `raja-putrah vane vasati`  
> **Devanāgarī:** राजपुत्रः वने वसति  
> **Context:** Passage from Ramayana describing Rama's lineage; *raja-putrah* means "son of the king".  
> **Compound:** `raja-putrah`  
> **Prompt:** *What type of samāsa (compound) is `raja-putrah` (son of the king)?*

| # | Option | Verdict |
|---|---|---|
| A | **Tatpurusha** (determinative) — second member is head, first qualifies it via case relation | **correct** |
| B | Dvandva (copulative) — "king and son", both members equally prominent | wrong |
| C | Bahuvrihi (possessive) — "he who has a king as son" | wrong |
| D | Avyayibhava (adverbial) — compound functions as indeclinable adverb | wrong |

> **Why it traps an LLM.** The surface form `raja-putrah` is structurally identical to a possessive (Bahuvrihi) compound that *would* mean "one whose son is a king". Disambiguation requires reading the case ending (`-aḥ`, nominative singular) and recognising that the head is `putra` ("son") qualified by the genitive `rajan` ("of the king") — a Tatpurusha pattern. This is exactly the classification step Murugesh et al. (2019) flag as a primary blocker for Sanskrit MT.

### Task 5 — Manuscript Restoration (Tool-Use POMDP)

**Why it matters.** Real manuscript restoration is not single-shot. A scholar gathers evidence from dictionaries, commentaries, meter analysis, and witness collations *before* committing to a reading. This task is a **partially observable MDP** where the agent has a tool budget and must (a) gather relevant evidence, (b) avoid redundant calls, then (c) commit to an answer.

**Example episode (`rest_001`, beginner difficulty, Bhagavad Gita 2.19):**

> **Passage (IAST):** `ya enaṃ vetti hantāraṃ yaś cainaṃ manyate hatam`  
> **Devanāgarī:** य एनं वेत्ति हन्तारं यश्चैनं मन्यते हतम्  
> **Disambiguation type:** glossary (the word `hantāram` is the contested term)  
> **Tool budget:** 8 calls · **Tools needed:** `lexicon_lookup`, `commentary_fetch`

A successful trajectory looks like this:

```text
step 1: action = tool_call("lexicon_lookup", "hantāram")
        observation: [{"meaning": "slayer", "domain": "general", "conf": 0.9},
                      {"meaning": "agent of action (philosophical sense)", "domain": "vedanta", "conf": 0.95}]
        tool_reward: +0.08  (PRIMARY tool for glossary episode)

step 2: action = tool_call("commentary_fetch", "BG 2.19 hantāram")
        observation: Shankara's commentary linking hantāram to "kartṛ" (agent), not to physical killing.
        tool_reward: +0.05  (SECOND tool, PRIMARY already used)
                   + 0.05  (workflow pair bonus: lexicon_lookup → commentary_fetch)

step 3: action = commit("The Self neither slays nor is it slain — 'hantāram' refers to the agent of action,
                        not the true subject")
        terminal_reward = r_correctness × M_evidence − P_budget
                        = 1.00 × (0.60 + 0.40 × 2/2) − 0.10 × max(0, 3−3)/8
                        = 1.00 × 1.00 − 0.00
                        = 1.00
```

> **Why it traps an LLM.** A vanilla LLM will skip evidence-gathering entirely and guess from priors. The reward structure punishes that path: a wrong commit returns `0.0` regardless of how confidently it was produced, and skipping the PRIMARY tool caps the evidence multiplier at 0.60. Agents must learn to **invest steps before committing** — the same protocol a Sanskrit philologist actually follows.

### Task 6 — Full Manuscript Session (Long-Horizon Cross-Phase Consistency)

**Why it matters.** A single Sanskrit verse cluster requires resolving lexical, phonological, morphological, **and** discourse problems simultaneously — and the answers must agree with each other. A model that says `hantāram` means "agent of action" in phase 1 but then treats it as "physical killer" in phase 4 has produced an internally contradictory translation. Task 6 explicitly grades that consistency.

**Example episode (`session_001`, Bhagavad Gita 2.19–2.47):**

| Phase | Skill | Passage | Prompt |
|---|---|---|---|
| 1 | Glossary | `ya enam vetti hantaram` | What does `hantaram` mean here? → *Agent of action (philosophical)* |
| 2 | Sandhi | `karmanyevadhikaras te` | How does `karmanyevadhikaras` split? → *karmani + eva + adhikarah* |
| 3 | Samāsa | `sthitaprajna` | What type of compound? → *Bahuvrihi (one whose prajna is established)* |
| 4 | Coherence | `sa budhhya yukto…` | Pronoun resolution across verses → *the steady-minded yogi* |
| 5 | Restoration | mixed passage | Final committed reading consistent with phases 1–4 |

```text
session_score = mean(phase_rewards)  −  consistency_penalty  +  consistency_bonus

# zero contradictions across phases  →  +0.05 bonus
# each phase contradiction          →  −0.05 penalty
```

> **Why it traps an LLM.** It is easy to score well on each phase in isolation. The penalty fires when, for example, the agent picks "physical killer" priors in phase 5 after committing to "agent of action" in phase 1. This penalises the failure mode that breaks every long-form Sanskrit translation pipeline today: locally plausible, globally inconsistent output.

---

## Dataset Statistics

| Task | Episodes | Domains | Difficulty |
|---|---|---|---|
| Glossary Anchoring | 150 | Ayurveda, Astronomy, Philosophy | Easy |
| Sandhi Resolution | 150 | Philosophy, Ayurveda, Narrative | Medium |
| Samāsa Classification | 150 | Philosophy, Narrative, Ayurveda, Astronomy | Medium |
| Referential Coherence | 150 | Narrative, Philosophy | Hard |
| Manuscript Restoration | 150 | Ayurveda, Philosophy, Narrative, Astronomy | Adaptive (Beginner → Expert) |
| Full Manuscript Session | 150 | All domains | Hard |

Each task has **150 unique hand-annotated episodes** in the data files (900 total). During GRPO training, the trainer dynamically varies seeds over this base pool to generate diverse `(prompt, seed)` pairs without requiring additional annotation.

---

## Grader Design — Why No LLM, No BLEU

All six graders are **fully deterministic**:

- No LLM judge calls
- No BLEU/ROUGE — unreliable for Sanskrit free word order
- Exact string match against pre-annotated answer tables embedded in data JSON

This guarantees **100% reproducible scores** across runs, models, and hardware. Two runs with the same seed will always produce identical scores.

---

## Reward Structure

### Tasks 1–4 (Single-Step MCQ)

Wrong answers always return exactly `0.0`. This is intentional for GRPO compatibility, not a floor.

The advantage normalization formula:

```
A_i = (r_i - mean(group)) / (std(group) + epsilon)
```

With a floor around `0.5`, group standard deviation typically collapses near `0.10–0.15`, producing weak training signal. With true zero for wrong answers, standard deviation typically rises near `0.35–0.45`, yielding stronger group-relative advantages.

| Outcome | Raw | Shaped |
|---|---|---|
| Full credit | 1.00 | 0.95 |
| Partial credit | 0.40 | 0.50 |
| Adjacent sandhi | 0.25 | 0.25 |
| Wrong | 0.00 | 0.00 |

Reward shaping applied in the environment:

- `raw = 0.0` → `0.0` (unchanged)
- `0.0 < raw < 0.40` → linear identity (`shaped = raw`)
- `0.40 ≤ raw ≤ 1.00` → `shaped = 0.50 + (raw − 0.40) × (0.45 / 0.60)`

### Task 5 — Manuscript Restoration (POMDP)

**Per-step tool reward:**

```
tool_reward = relevance_bonus + workflow_bonus − redundancy_penalty
```

| Condition | Reward |
|---|---|
| PRIMARY tool for episode type | +0.08 |
| SECOND tool (PRIMARY already used) | +0.05 |
| SECOND tool (PRIMARY not yet used) | +0.04 |
| Supporting tool | +0.04 |
| Redundant call (same tool + same input) | −0.05 |
| Irrelevant tool (e.g. `meter_checker` on prose) | −0.05 |

**Workflow pair bonuses:**

| Pair | Bonus |
|---|---|
| `sandhi_parser → meter_checker` | +0.05 |
| `lexicon_lookup → commentary_fetch` | +0.05 |
| `witness_compare → referent_tracker` | +0.03 |

**Terminal commit reward:**

```
terminal_reward = r_correctness × M_evidence − P_budget
```

Where:
- `M_evidence = 0.60 + 0.40 × (distinct_relevant_tools_used / tools_needed)` (range 0.60–1.00)
- `P_budget = 0.10 × max(0, steps_used − ideal_steps) / tool_budget` (range 0.00–0.10)

Critical invariants:
- Wrong commit → `0.0` regardless of evidence gathered
- Episode score is the terminal commit reward only
- Tool rewards are dense intermediate signals and are **not** added into the final episode score

### Task 6 — Full Manuscript Session

```
session_score = mean(phase_rewards) − consistency_penalty + consistency_bonus
```

| Rule | Effect |
|---|---|
| Each contradiction between phases | −0.05 per violation |
| Zero violations across all phases | +0.05 bonus |

A violation occurs when the restoration-phase final interpretation contradicts an answer given in an earlier phase.

---

## The Six Philological Tools (Task 5)

| Tool | Input | Returns | PRIMARY for |
|---|---|---|---|
| `lexicon_lookup` | Sanskrit lemma/term | Domain-conditioned meanings and glosses | glossary episodes |
| `sandhi_parser` | Compound form | Candidate splits with rule-level structure | sandhi, samasa episodes |
| `meter_checker` | Candidate split/text span | Meter compatibility signal | SECOND in sandhi/samasa |
| `commentary_fetch` | Term, phrase, or verse reference | Commentary snippets linked to interpretation | SECOND in glossary |
| `witness_compare` | Verse/manuscript locus | Variant witness readings and differences | support (glossary/sandhi/coherence) |
| `referent_tracker` | Pronoun/entity cue | Candidate antecedents and discourse links | coherence episodes |

### Tool Relevance Matrix

| Episode type | `lexicon_lookup` | `sandhi_parser` | `meter_checker` | `commentary_fetch` | `witness_compare` | `referent_tracker` |
|---|---|---|---|---|---|---|
| glossary | PRIMARY | support | none | SECOND | support | none |
| sandhi | support | PRIMARY | SECOND | none | support | none |
| samasa | support | PRIMARY | SECOND | none | none | none |
| coherence | support | support | none | support | support | PRIMARY |

---

## Adaptive Difficulty Curriculum (Task 5)

| Level | OCR Noise | Commentary | Tool Budget |
|---|---|---|---|
| Beginner | None | Full | 8 |
| Intermediate | 10% | Partial | 6 |
| Hard | 25% | Partial | 5 |
| Expert | 40% + conflicting witnesses | None | 4 |

**Promotion threshold:** mean of last 10 scores > 0.80 with at least 5 episodes  
**De-escalation threshold:** mean < 0.45

---

## Project Setup — Local Development

**1. Clone the repository:**

```bash
git clone https://huggingface.co/spaces/Adityahars/Sanskrit-env
cd sanskrit-env
```

**2. Create and activate a virtual environment (Python 3.11+):**

```bash
python -m venv .venv

# PowerShell
.venv\Scripts\Activate.ps1

# bash / zsh
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables:**

```bash
cp .env.example .env
# Fill in HF_TOKEN and CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID
```

**5. Start the server:**

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

**6. Validate:**

```bash
curl http://localhost:7860/health
```

**7. Run the test agent:**

```bash
python test_agent.py --local --task all --episodes 5
```

---

## Project Setup — Docker

```bash
# Build
docker build -t sanskrit-env:local .

# Run
docker run --rm -p 7860:7860 sanskrit-env:local

# Validate
curl http://localhost:7860/health
```

---

## GRPO Training

Train a fine-tuned adapter on SanskritEnv using HuggingFace Jobs (A100 GPU):

```powershell
# Set credentials in .env, then:
$env:HF_TOKEN = "hf_..."
python training/submit_hf_job.py --push-to-hub --flavor a100-large --timeout 12h
```

All training hyperparameters are controlled via `.env` — no hardcoded values in scripts:

| Variable | Description |
|---|---|
| `MODEL_ID` | Base model or checkpoint to fine-tune |
| `EPISODES_PER_TASK` | `(prompt, seed)` pairs sampled per task during training. Default `1500` is generated dynamically over the 150-episode base pool via seed variation. |
| `TRAIN_EPOCHS` | Training epochs (default: 1.0) |
| `GROUP_SIZE` | GRPO group size (default: 8) |
| `LR` | Learning rate (default: 2e-6) |
| `LORA_R` / `LORA_ALPHA` | LoRA rank and scaling |
| `PUSH_TO_HUB` | Set to `1` to push adapter to Hub after training |
| `HUB_MODEL_ID` | Hub repo for the trained adapter |

See `.env.example` for the full list.

---

## Training Results

Latest GRPO run on HuggingFace Jobs (A100-large, ~6h). The run started from the v1 checkpoint (`Adityahars/sanskrit-qwen-grpo`), trained for **3 epochs × 225 steps = 675 global steps** with cosine LR decay (`4.0e-6 → 5.9e-9`).

### Run configuration

| Setting | Value |
|---|---|
| Base model | `Adityahars/sanskrit-qwen-grpo` (v1 checkpoint, Qwen2.5-1.5B-Instruct + LoRA) |
| Algorithm | GRPO (group size = 8) |
| Episodes per task (dynamic) | 1500 over 150 unique base episodes |
| Tasks trained | 6 (all linguistic layers) |
| Optimizer | AdamW, cosine schedule, peak LR 4e-6 |
| Total global steps | 675 |
| Per-device batch / grad accum | 4 / 4 |

### Reward curve

![Group-relative reward over training](assets/reward_curve.png)

- **Start (step 5):** `reward_mean = 0.475` — baseline behaviour from the v1 checkpoint.
- **First 50 steps mean:** `0.452` (model briefly explores around the prior policy).
- **Peak step 545:** `reward_mean = 0.733` — strongest sustained single-batch performance.
- **Final 10 batches mean:** `~0.576` — a stable **+27% relative lift** over the early-training average without entropy collapse.

Reward stays bounded between roughly `0.31` and `0.73` throughout, with the trajectory drifting upward across all three epochs. There are no spikes, no divergence, and no "reward hacking" plateaus.

### Reward variance & GRPO health

![Per-group reward standard deviation](assets/reward_std_curve.png)

Group-relative reward std stays in the healthy range `0.21 – 0.39` across the entire run. Crucially:

![Fraction of groups with zero variance](assets/zero_variance_fraction.png)

`frac_reward_zero_std == 0.0` for every step. **No GRPO group ever collapsed to a single reward value**, which means the advantage signal `A_i = (r_i − μ) / (σ + ε)` is well-defined throughout. This is the clearest possible health signal for a GRPO run — the de-shaped reward design (zero floor on wrong answers) is doing its job.

### Policy entropy

![Policy entropy over training](assets/entropy_curve.png)

Entropy stays in the band `0.84 – 1.07` across all 675 steps, with no monotone collapse. The model is improving its rewards **without** the typical GRPO failure mode of greedy-mode collapse, where entropy plummets and the policy stops exploring. This is the expected behaviour of a low LR (2e-6) and small KL footprint.

### Clipping behaviour

![Importance-ratio clipping](assets/clipped_ratio_curve.png)

`clipped_ratio` sits at `1.00` for ~95% of steps with occasional dips to `0.99375` (one in eight tokens hitting a clip). This indicates that the GRPO ratio bounds are loose enough not to throttle gradient flow but tight enough to prevent runaway updates.

### Per-task improvement

![Per-task score improvement vs v1 baseline](assets/per_task_improvement.png)
![Success-rate comparison: pre-train vs post-train](assets/success_rate_comparison.png)

Across all six tasks the post-training success rate dominates the v1 baseline. The largest absolute lift is on the **single-step MCQ tasks** (Glossary, Sandhi, Samāsa) where the GRPO advantage signal is strongest. The **multi-step tasks** (Coherence, Restoration, Full Session) move more slowly because their reward density is lower per token — but they also do not regress.

### Training summary table

| Phase | Steps | Mean reward | Std | Entropy | Notes |
|---|---|---:|---:|---:|---|
| Early (5–50) | 10 | 0.452 | 0.34 | 0.91 | Warm-up on top of v1 checkpoint |
| Mid-epoch 1 (50–225) | 35 | 0.541 | 0.32 | 0.94 | First clear lift above baseline |
| Epoch 2 (225–450) | 45 | 0.561 | 0.30 | 0.95 | Steady gains, healthy variance |
| Epoch 3 (450–675) | 45 | 0.572 | 0.30 | 0.96 | Peak at step 545 (0.733); final mean 0.576 |

> **Headline result.** The v3 run improved mean episode reward from `0.45 → 0.58` (+0.13 absolute, +27% relative) while keeping entropy, variance, and gradient norms in the healthy band — and without ever collapsing a single GRPO group. The trained adapter is published at [`archijaiswal07/sanskrit-qwen-grpo-v3`](https://huggingface.co/archijaiswal07/sanskrit-qwen-grpo-v3).

---

## Running `inference.py` (Submission Script)

`inference.py` is the OpenEnv submission artifact. It follows output constraints strictly:

- Stdout contains only `[START]`, `[STEP]`, and `[END]` lines
- Debug/error details go to stderr
- All settings are pulled from environment variables

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Running `test_agent.py` (Development Evaluation)

`test_agent.py` is the development evaluation runner. It supports both remote and local environments.

```bash
# All tasks against the HF Space (default)
python test_agent.py --task all --episodes 5

# All tasks against localhost
python test_agent.py --local --task all --episodes 5

# Single task with verbose output
python test_agent.py --task referential_coherence --episodes 1 --verbose

# Task 5 at hard difficulty
python test_agent.py --task manuscript_restoration --difficulty hard --episodes 10

# Task 6 full session
python test_agent.py --task full_manuscript_session --episodes 3
```

---

## Test Results

Model: `@cf/meta/llama-3.2-3b-instruct` · Provider: `cloudflare`

### Run 1 — Seed `42` · 3 episodes per task · Overall mean `0.465`, std `0.352`

| Task | Episodes | Score Mean | Score Std |
|---|---:|---:|---:|
| Glossary Anchoring | 3 | 0.333 | 0.236 |
| Sandhi Resolution | 3 | 0.400 | 0.402 |
| Samāsa Classification | 3 | 0.483 | 0.388 |
| Referential Coherence | 3 | 0.067 | 0.047 |
| Manuscript Restoration | 3 | 0.650 | 0.000 |
| Full Manuscript Session | 3 | 0.857 | 0.066 |

### Run 2 — 25 episodes (Tasks 1–4) · 10 episodes (Tasks 5–6) · Overall mean `0.478`, std `0.399`

| Task | Episodes | Score Mean | Score Std | Notes |
|---|---:|---:|---:|---|
| Glossary Anchoring | 25 | 0.444 | 0.402 | |
| Sandhi Resolution | 25 | 0.630 | 0.399 | |
| Samāsa Classification | 25 | 0.386 | 0.405 | |
| Referential Coherence | 25 | 0.278 | 0.345 | |
| Manuscript Restoration | 10 | 0.573 | 0.304 | mean tools used: 1.4 · mean steps: 2.4 |
| Full Manuscript Session | 10 | 0.824 | 0.042 | |

---

## Citation

```bibtex
@misc{sanskritenv2026,
  title   = {SanskritEnv: A Reinforcement Learning Environment for Sanskrit Manuscript Interpretation},
  author  = {Meta\_Mesh},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Adityahars/Sanskrit-env},
  note    = {OpenEnv-compatible environment for structured linguistic ambiguity resolution}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).  
Sanskrit texts used are in the public domain (composed before 1928). Annotations, graders, and environment code are original to this project.

---

## Acknowledgements

- [Meta × HuggingFace OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework
- [Gyan Bharatam Mission](https://indiaculture.gov.in/) — the real-world problem this addresses
- [Monier-Williams Sanskrit Dictionary](https://www.sanskrit-lexicon.uni-koeln.de/) — lexical reference
- [Sanskrit Sandhi Split Sighum](https://github.com/DorenaBudajeva/sighum) — annotated corpus reference
- [Itihasa](https://github.com/goru001/nlp-for-sanskrit) — annotated corpus reference
