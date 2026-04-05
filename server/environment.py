"""
SanskritEnv Environment — core logic.

Implements the OpenEnv Environment interface:
    reset() -> ManuscriptObservation
    step(action: ManuscriptAction) -> ManuscriptObservation
    state -> ManuscriptState

Four task modes:
    task_id = "glossary_anchoring"       (Easy)
    task_id = "sandhi_resolution"        (Medium)
    task_id = "samasa_classification"    (Medium)
    task_id = "referential_coherence"    (Hard)

All graders are deterministic. No external API calls inside this file.
"""

import json
import uuid
import random
from pathlib import Path
from typing import Optional
from openenv.core.env_server import Environment
from models import ManuscriptAction, ManuscriptObservation, ManuscriptState
from graders import GlossaryGrader, SandhiGrader, CoherenceGrader, SamasaGrader

DATA_DIR = Path(__file__).parent.parent / "data"


class SanskritEnvironment(Environment):
    """
    Sanskrit Manuscript Interpretation Environment.

    An RL environment where agents resolve structural ambiguity
    in Sanskrit manuscript passages across three difficulty levels.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = ManuscriptState()
        self._current_episode = None
        self._task_id = "glossary_anchoring"
        self._glossary_grader = GlossaryGrader()
        self._sandhi_grader = SandhiGrader()
        self._coherence_grader = CoherenceGrader()
        self._samasa_grader = SamasaGrader()

        # Load all data at startup — never at step time
        self._task1_data = self._load_json("task1_glossary.json")
        self._task2_data = self._load_json("task2_sandhi.json")
        self._task3_data = self._load_json("task3_coherence.json")
        self._task4_data = self._load_json("task4_samasa.json")

        # Task 3 state
        self._t3_verse_index = 0
        self._t3_checkpoint_index = 0
        self._t3_checkpoint_rewards = []
        self._t3_phase = "checkpoint"  # "checkpoint" or "final"

    def _load_json(self, filename: str) -> dict:
        path = DATA_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> ManuscriptObservation:
        """
        Start a new episode.

        Args:
            seed: Random seed for episode selection. None = random.
            episode_id: Optional specific episode ID to load.
            task_id: Which task to run. One of:
                     "glossary_anchoring" | "sandhi_resolution" | "samasa_classification" | "referential_coherence"
                     Defaults to "glossary_anchoring".
        """
        if seed is not None:
            random.seed(seed)

        # Set task
        self._task_id = task_id or "glossary_anchoring"
        if self._task_id not in ("glossary_anchoring", "sandhi_resolution", "samasa_classification", "referential_coherence"):
            self._task_id = "glossary_anchoring"

        # Select episode
        episodes = self._get_episodes_for_task(self._task_id)
        if episode_id:
            ep = next((e for e in episodes if e["id"] == episode_id), None)
            if ep is None:
                ep = random.choice(episodes)
        else:
            ep = random.choice(episodes)

        self._current_episode = ep

        # Reset state
        new_episode_id = episode_id or str(uuid.uuid4())
        self._state = ManuscriptState(
            episode_id=new_episode_id,
            step_count=0,
            task_id=self._task_id,
            passage_id=ep["id"],
            total_decisions=self._count_total_decisions(ep),
            correct_decisions=0,
            partial_decisions=0,
            decision_history=[],
            consistency_map={},
            is_complete=False,
        )

        # Task 3 specific reset
        if self._task_id == "referential_coherence":
            self._t3_verse_index = 0
            self._t3_checkpoint_index = 0
            self._t3_checkpoint_rewards = []
            self._t3_phase = "checkpoint" if ep.get("consistency_checkpoints") else "final"

        return self._build_initial_observation(ep)

    def step(self, action: ManuscriptAction, **kwargs) -> ManuscriptObservation:
        """
        Process one decision from the agent.

        The agent's selected_option must exactly match one entry
        in the candidate_options provided in the previous observation.
        """
        self._state.step_count += 1

        if self._current_episode is None:
            return ManuscriptObservation(
                task_id="none",
                episode_id="none",
                source_text_iast="",
                source_text_devanagari="",
                english_context="",
                domain="none",
                decision_prompt="Call reset() before step().",
                candidate_options=["reset", "reset", "reset", "reset"],
                done=True,
                reward=0.0,
                feedback_message="Environment not initialized. Call reset() first.",
            )

        ep = self._current_episode

        if self._task_id == "glossary_anchoring":
            return self._step_task1(action, ep)
        elif self._task_id == "sandhi_resolution":
            return self._step_task2(action, ep)
        elif self._task_id == "referential_coherence":
            return self._step_task3(action, ep)
        elif self._task_id == "samasa_classification":
            return self._step_task4(action, ep)
        else:
            return self._step_task1(action, ep)

    @property
    def state(self) -> ManuscriptState:
        return self._state

    # ─────────────────────────────────────────────────────────────
    # Task 1 — Glossary Anchoring
    # ─────────────────────────────────────────────────────────────

    def _step_task1(self, action: ManuscriptAction, ep: dict) -> ManuscriptObservation:
        reward, feedback = self._glossary_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )

        # Update state
        is_correct = reward == 1.0
        is_partial = 0.0 < reward < 1.0
        self._state.correct_decisions += int(is_correct)
        self._state.partial_decisions += int(is_partial)
        self._state.decision_history.append({
            "step": self._state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "reward": reward,
        })
        self._state.is_complete = True

        final_score = self._normalize_score(
            self._state.correct_decisions,
            self._state.partial_decisions,
            self._state.total_decisions,
            task_id="glossary_anchoring",
        )

        return ManuscriptObservation(
            task_id=self._task_id,
            episode_id=self._state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            target_term_iast=ep["target_term_iast"],
            active_glossary={ep["target_term_iast"]: "See candidate options"},
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )

    # ─────────────────────────────────────────────────────────────
    # Task 2 — Sandhi Resolution
    # ─────────────────────────────────────────────────────────────

    def _step_task2(self, action: ManuscriptAction, ep: dict) -> ManuscriptObservation:
        reward, feedback = self._sandhi_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )

        is_correct = reward == 1.0
        is_partial = 0.0 < reward < 1.0
        self._state.correct_decisions += int(is_correct)
        self._state.partial_decisions += int(is_partial)
        self._state.decision_history.append({
            "step": self._state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "reward": reward,
        })
        self._state.is_complete = True

        final_score = self._normalize_score(
            self._state.correct_decisions,
            self._state.partial_decisions,
            self._state.total_decisions,
            task_id="sandhi_resolution",
        )

        return ManuscriptObservation(
            task_id=self._task_id,
            episode_id=self._state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            compound_iast=ep["compound_iast"],
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )

    # ─────────────────────────────────────────────────────────────
    # Task 3 — Referential Coherence (multi-step episode)
    # ─────────────────────────────────────────────────────────────

    def _step_task3(self, action: ManuscriptAction, ep: dict) -> ManuscriptObservation:
        checkpoints = ep.get("consistency_checkpoints", [])

        # Are we in checkpoint phase?
        if self._t3_phase == "checkpoint" and self._t3_checkpoint_index < len(checkpoints):
            cp = checkpoints[self._t3_checkpoint_index]
            cp_candidates = self._get_checkpoint_candidates(cp["answer"], ep)

            cp_reward, cp_feedback = self._coherence_grader.grade_checkpoint(
                selected_option=action.selected_option,
                correct_answer=cp["answer"],
                candidate_options=cp_candidates,
            )
            self._t3_checkpoint_rewards.append(cp_reward)
            self._t3_checkpoint_index += 1

            # Update consistency map
            self._state.consistency_map[cp["question"]] = action.selected_option

            # Advance verse index
            self._t3_verse_index = cp["after_verse"]

            # Check if all checkpoints done
            if self._t3_checkpoint_index >= len(checkpoints):
                self._t3_phase = "final"

            # Verses seen so far
            verses_so_far = ep["verses"][: self._t3_verse_index]

            # Next checkpoint or final question
            if self._t3_phase == "checkpoint":
                next_cp = checkpoints[self._t3_checkpoint_index]
                next_prompt = next_cp["question"]
                next_candidates = self._get_checkpoint_candidates(next_cp["answer"], ep)
            else:
                next_prompt = ep["referential_question"]
                next_candidates = ep["candidate_options"]

            return ManuscriptObservation(
                task_id=self._task_id,
                episode_id=self._state.episode_id,
                source_text_iast=ep["verses"][self._t3_verse_index - 1]["iast"] if verses_so_far else "",
                source_text_devanagari=ep["verses"][self._t3_verse_index - 1].get("devanagari", "") if verses_so_far else "",
                english_context=ep.get("title", ""),
                domain=ep.get("domain", "narrative"),
                verses_so_far=[
                    {"verse_num": v["verse_num"], "iast": v["iast"], "english": v["english"]}
                    for v in verses_so_far
                ],
                current_verse_num=self._t3_verse_index,
                decision_prompt=next_prompt,
                candidate_options=next_candidates,
                step_reward=cp_reward,
                cumulative_score=self._compute_t3_partial_score(),
                feedback_message=cp_feedback,
                consistency_history=[
                    {"question": q, "answer": a}
                    for q, a in self._state.consistency_map.items()
                ],
                done=False,
                reward=None,
            )

        else:
            # Final referential question
            final_reward, final_feedback = self._coherence_grader.grade_final(
                selected_option=action.selected_option,
                correct_answer=ep["correct_answer"],
                candidate_options=ep["candidate_options"],
            )

            episode_score = self._coherence_grader.compute_episode_score(
                final_reward=final_reward,
                checkpoint_rewards=self._t3_checkpoint_rewards,
            )

            self._state.correct_decisions += int(final_reward > 0)
            self._state.is_complete = True
            self._state.decision_history.append({
                "step": self._state.step_count,
                "selected": action.selected_option,
                "correct": ep["correct_answer"],
                "reward": final_reward,
                "episode_score": episode_score,
            })

            all_verses = ep["verses"]
            return ManuscriptObservation(
                task_id=self._task_id,
                episode_id=self._state.episode_id,
                source_text_iast=all_verses[-1]["iast"],
                source_text_devanagari=all_verses[-1].get("devanagari", ""),
                english_context=ep.get("title", ""),
                domain=ep.get("domain", "narrative"),
                verses_so_far=[
                    {"verse_num": v["verse_num"], "iast": v["iast"], "english": v["english"]}
                    for v in all_verses
                ],
                current_verse_num=len(all_verses),
                decision_prompt=ep["referential_question"],
                candidate_options=ep["candidate_options"],
                step_reward=final_reward,
                cumulative_score=episode_score,
                feedback_message=final_feedback,
                consistency_history=[
                    {"question": q, "answer": a}
                    for q, a in self._state.consistency_map.items()
                ],
                done=True,
                reward=episode_score,
            )


    # ─────────────────────────────────────────────────────────────
    # Task 4 — Samasa Classification
    # ─────────────────────────────────────────────────────────────

    def _step_task4(self, action: ManuscriptAction, ep: dict) -> ManuscriptObservation:
        reward, feedback = self._samasa_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )

        is_correct = reward == 1.0
        is_partial = 0.0 < reward < 1.0
        self._state.correct_decisions += int(is_correct)
        self._state.partial_decisions += int(is_partial)
        self._state.decision_history.append({
            "step": self._state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "reward": reward,
        })
        self._state.is_complete = True

        final_score = self._normalize_score(
            self._state.correct_decisions,
            self._state.partial_decisions,
            self._state.total_decisions,
            task_id="samasa_classification",
        )

        return ManuscriptObservation(
            task_id=self._task_id,
            episode_id=self._state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            compound_iast=ep.get("compound_iast"),
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )
    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _get_episodes_for_task(self, task_id: str) -> list:
        if task_id == "glossary_anchoring":
            return self._task1_data["episodes"]
        elif task_id == "sandhi_resolution":
            return self._task2_data["episodes"]
        elif task_id == "referential_coherence":
            return self._task3_data["episodes"]
        elif task_id == "samasa_classification":
            return self._task4_data["episodes"]
        return self._task1_data["episodes"]

    def _count_total_decisions(self, ep: dict) -> int:
        if self._task_id == "referential_coherence":
            return len(ep.get("consistency_checkpoints", [])) + 1
        return 1

    def _build_initial_observation(self, ep: dict) -> ManuscriptObservation:
        if self._task_id == "referential_coherence":
            checkpoints = ep.get("consistency_checkpoints", [])
            if checkpoints:
                first_cp = checkpoints[0]
                prompt = first_cp["question"]
                candidates = self._get_checkpoint_candidates(first_cp["answer"], ep)
                verse_index = first_cp["after_verse"]
            else:
                prompt = ep["referential_question"]
                candidates = ep["candidate_options"]
                verse_index = len(ep["verses"])
                self._t3_phase = "final"

            verses_so_far = ep["verses"][:verse_index]
            return ManuscriptObservation(
                task_id=self._task_id,
                episode_id=self._state.episode_id,
                source_text_iast=ep["verses"][verse_index - 1]["iast"] if verses_so_far else "",
                source_text_devanagari=ep["verses"][verse_index - 1].get("devanagari", "") if verses_so_far else "",
                english_context=ep.get("title", ""),
                domain=ep.get("domain", "narrative"),
                verses_so_far=[
                    {"verse_num": v["verse_num"], "iast": v["iast"], "english": v["english"]}
                    for v in verses_so_far
                ],
                current_verse_num=verse_index,
                decision_prompt=prompt,
                candidate_options=candidates,
                step_reward=0.0,
                cumulative_score=0.0,
                feedback_message="New episode started. Read the verses and answer the question.",
                consistency_history=[],
                done=False,
                reward=None,
            )

        # Tasks 1 and 2 — single decision episodes
        return ManuscriptObservation(
            task_id=self._task_id,
            episode_id=self._state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            target_term_iast=ep.get("target_term_iast"),
            compound_iast=ep.get("compound_iast"),
            active_glossary={ep.get("target_term_iast", ""): "See candidate options"} if ep.get("target_term_iast") else None,
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=0.0,
            cumulative_score=0.0,
            feedback_message="New episode started. Read the passage and select the correct interpretation.",
            done=False,
            reward=None,
        )

    def _get_checkpoint_candidates(self, correct_answer: str, ep: dict) -> list:
        """
        Build 4 candidates for a checkpoint question by reusing the episode's
        candidate_options pool (which already has well-written descriptions).

        The correct option is whichever candidate_option starts with correct_answer.
        Distractors are the remaining candidate_options, shuffled.
        This guarantees:
          - The correct option string is always present verbatim in the list
          - grade_checkpoint() exact-match against cp["answer"] will succeed
          - Distractors are meaningful character names, not verse first-words
        """
        episode_options = ep.get("candidate_options", [])

        # Find the full option string that matches the short checkpoint answer
        correct_full = next(
            (opt for opt in episode_options if opt.startswith(correct_answer)),
            correct_answer,  # fallback: use short name as-is if no match found
        )

        # Build distractor list from remaining episode options
        distractors = [opt for opt in episode_options if opt != correct_full]

        candidates = [correct_full] + distractors
        random.shuffle(candidates)
        return candidates[:4]

    def _normalize_score(
        self,
        correct: int,
        partial: int,
        total: int,
        task_id: str,
    ) -> float:
        if total == 0:
            return 0.0
        if task_id == "glossary_anchoring":
            raw = correct * 1.0 + partial * 0.4
            return round(min(raw / total, 1.0), 4)
        elif task_id == "sandhi_resolution":
            raw = correct * 1.0 + partial * 0.25
            return round(min(raw / total, 1.0), 4)
        elif task_id == "samasa_classification":
            raw = correct * 1.0 + partial * 0.4
            return round(min(raw / total, 1.0), 4)
        return round(min(correct / total, 1.0), 4)

    def _compute_t3_partial_score(self) -> float:
        cp_score = sum(self._t3_checkpoint_rewards)
        max_cp = self._coherence_grader.CHECKPOINT_CORRECT * len(self._t3_checkpoint_rewards)
        if max_cp == 0:
            return 0.0
        return round(min(cp_score / (max_cp + self._coherence_grader.MAIN_CORRECT), 1.0), 4)
