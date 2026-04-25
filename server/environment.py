"""
SanskritEnv Environment — core logic.

Implements the OpenEnv Environment interface:
    reset() -> ManuscriptObservation
    step(action: ManuscriptAction) -> ManuscriptObservation
    state -> ManuscriptState

Six task modes:
    task_id = "glossary_anchoring"       (Easy)
    task_id = "sandhi_resolution"        (Medium)
    task_id = "samasa_classification"    (Medium)
    task_id = "referential_coherence"    (Hard)
    task_id = "manuscript_restoration"   (Expert, POMDP)
    task_id = "full_manuscript_session"  (Expert, Long-Horizon)

All graders are deterministic. No external API calls inside this file.
"""

import json
import uuid
import random
from pathlib import Path
from typing import Optional, List
from openenv.core.env_server import Environment
from models import ManuscriptAction, ManuscriptObservation, ManuscriptState
from graders import (
    GlossaryGrader, SandhiGrader, CoherenceGrader, SamasaGrader,
    RestorationGrader, ConsistencyGrader,
)
from server.tools import ManuscriptToolkit

DATA_DIR = Path(__file__).parent.parent / "data"


class SanskritEnvironment(Environment):
    """
    Sanskrit Manuscript Interpretation Environment.

    An RL environment where agents resolve structural ambiguity
    in Sanskrit manuscript passages across three difficulty levels.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_EPISODE_SCORE = 0.95

    DIFFICULTY_LADDER = ["beginner", "intermediate", "hard", "expert"]

    def __init__(self):
        self._sessions = {} # {session_id: session_dict}
        self._active_session_id: Optional[str] = None
        self._episode_rng = random.Random()
        self._task_episode_cycles = {}  # {task_id: {order: [indices], cursor: int}}
        self._glossary_grader = GlossaryGrader()
        self._sandhi_grader = SandhiGrader()
        self._coherence_grader = CoherenceGrader()
        self._samasa_grader = SamasaGrader()
        self._restoration_grader = RestorationGrader()
        self._consistency_grader = ConsistencyGrader()
        self._toolkit = ManuscriptToolkit()

        # Load all data at startup
        self._task1_data = self._load_json("task1_glossary.json")
        self._task2_data = self._load_json("task2_sandhi.json")
        self._task3_data = self._load_json("task3_coherence.json")
        self._task4_data = self._load_json("task4_samasa.json")
        self._task5_data = self._load_json("task5_restoration.json")
        self._task6_data = self._load_json("task6_full_session.json")

    def _shape_reward_signal(self, raw_score: float) -> float:
        """Shape raw reward for GRPO-compatible training signal.

        Critical for RL: wrong answers must return true 0.0 so that
        GRPO computes meaningful group-relative advantages.
        """
        raw = min(max(float(raw_score), 0.0), 1.0)
        if raw == 0.0:
            return 0.0  # wrong answer → true zero (CRITICAL for GRPO)
        if raw < 0.40:
            return round(raw, 4)  # scale small partial rewards linearly
        # Map [0.40, 1.0] → [0.50, 0.95]
        return round(0.50 + (raw - 0.40) * (0.45 / 0.60), 4)

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
        selected_task = task_id or "glossary_anchoring"
        difficulty = kwargs.get("difficulty", None)

        # For manuscript_restoration, handle adaptive difficulty
        if selected_task == "manuscript_restoration":
            if difficulty is None or difficulty == "auto":
                difficulty = self._auto_select_difficulty([])
            episodes = [
                e for e in self._get_episodes_for_task(selected_task)
                if e.get("difficulty") == difficulty
            ]
            if not episodes:
                episodes = self._get_episodes_for_task(selected_task)
        else:
            episodes = self._get_episodes_for_task(selected_task)

        ep = self._select_episode(
            task_id=selected_task,
            episodes=episodes,
            seed=seed,
            episode_id=episode_id,
        )

        session_id = episode_id or str(uuid.uuid4())
        
        # Initialize session state
        session = {
            "task_id": selected_task,
            "current_episode": ep,
            "state": ManuscriptState(
                episode_id=session_id,
                step_count=0,
                task_id=selected_task,
                passage_id=ep["id"],
                total_decisions=self._count_total_decisions(ep, selected_task),
                correct_decisions=0,
                partial_decisions=0,
                decision_history=[],
                consistency_map={},
                is_complete=False,
            )
        }

        # Task 3 specific reset
        if session["task_id"] == "referential_coherence":
            session["t3_verse_index"] = 0
            session["t3_checkpoint_index"] = 0
            session["t3_checkpoint_rewards"] = []
            session["t3_phase"] = "checkpoint" if ep.get("consistency_checkpoints") else "final"

        # Task 5 specific reset
        if selected_task == "manuscript_restoration":
            session["t5_tool_history"] = []
            session["t5_steps_remaining"] = ep.get("tool_budget", 8)
            session["t5_relevant_tools_used"] = set()
            session["t5_workflow_pairs_awarded"] = set()
            session["t5_difficulty"] = ep.get("difficulty", "beginner")
            session["t5_tool_rewards"] = []

        # Task 6 specific reset
        if selected_task == "full_manuscript_session":
            session["t6_phase_index"] = 0
            session["t6_phase_rewards"] = []
            session["t6_phase_answers"] = []
            session["t6_consistency_violations"] = 0
            # Restoration sub-phase state
            session["t5_tool_history"] = []
            rest_phase = next((p for p in ep.get("phases", []) if p["phase"] == "restoration"), None)
            session["t5_steps_remaining"] = rest_phase.get("tool_budget", 4) if rest_phase else 4
            session["t5_relevant_tools_used"] = set()
            session["t5_workflow_pairs_awarded"] = set()
            session["t5_difficulty"] = "expert"
            session["t5_tool_rewards"] = []

        self._sessions[session_id] = session
        self._active_session_id = session_id
        return self._build_initial_observation(ep, session)

    def _select_episode(
        self,
        task_id: str,
        episodes: list,
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> dict:
        if not episodes:
            raise ValueError(f"No episodes available for task '{task_id}'.")

        if episode_id:
            exact = next((episode for episode in episodes if episode["id"] == episode_id), None)
            if exact is not None:
                return exact

        # Deterministic, duplicate-free traversal for sequential seeds.
        # For a task with N episodes, seeds that differ by < N map to unique indices.
        if seed is not None:
            index = seed % len(episodes)
            return episodes[index]

        cycle = self._task_episode_cycles.get(task_id)
        if cycle is None or cycle["cursor"] >= len(cycle["order"]):
            order = list(range(len(episodes)))
            self._episode_rng.shuffle(order)
            cycle = {"order": order, "cursor": 0}
            self._task_episode_cycles[task_id] = cycle

        episode_index = cycle["order"][cycle["cursor"]]
        cycle["cursor"] += 1
        return episodes[episode_index]

    def _resolve_session(self, request_id: Optional[str]) -> Optional[dict]:
        """
        Resolve session for both HTTP and WebSocket flows.

        HTTP UI passes request_id explicitly.
        OpenEnv WebSocket flow does not pass request_id to env.step(), so we
        fall back to the most recently reset session (or the only session).
        """
        if request_id and request_id in self._sessions:
            return self._sessions[request_id]

        if self._active_session_id and self._active_session_id in self._sessions:
            return self._sessions[self._active_session_id]

        if len(self._sessions) == 1:
            only_id = next(iter(self._sessions))
            self._active_session_id = only_id
            return self._sessions[only_id]

        return None

    def step(self, action: ManuscriptAction, request_id: Optional[str] = None, **kwargs) -> ManuscriptObservation:
        """
        Process one decision from the agent.
        """
        session = self._resolve_session(request_id)
        if request_id and request_id in self._sessions:
            self._active_session_id = request_id
        if not session:
            return ManuscriptObservation(
                task_id="none",
                episode_id="none",
                source_text_iast="",
                source_text_devanagari="",
                english_context="",
                domain="none",
                decision_prompt="Environment not initialized for this session. Call reset() first.",
                candidate_options=["reset", "reset", "reset", "reset"],
                done=True,
                reward=0.0,
                feedback_message="Session not found. Please click 'New Episode' to initialize.",
            )

        state = session["state"]
        ep = session["current_episode"]
        task_id = session["task_id"]
        
        state.step_count += 1

        if task_id == "glossary_anchoring":
            return self._step_task1(action, ep, session)
        elif task_id == "sandhi_resolution":
            return self._step_task2(action, ep, session)
        elif task_id == "referential_coherence":
            return self._step_task3(action, ep, session)
        elif task_id == "samasa_classification":
            return self._step_task4(action, ep, session)
        elif task_id == "manuscript_restoration":
            return self._step_task5_restoration(action, ep, session)
        elif task_id == "full_manuscript_session":
            return self._step_full_manuscript_session(action, ep, session)
        else:
            return self._step_task1(action, ep, session)

    @property
    def state(self) -> ManuscriptState:
        session = self._resolve_session(self._active_session_id)
        if session:
            return session["state"]
        raise RuntimeError("No active session. Call reset() first.")

    # ─────────────────────────────────────────────────────────────
    # Task 1 — Glossary Anchoring
    # ─────────────────────────────────────────────────────────────

    def _step_task1(self, action: ManuscriptAction, ep: dict, session: dict) -> ManuscriptObservation:
        raw_reward, feedback = self._glossary_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )
        step_reward = self._shape_reward_signal(raw_reward)

        state = session["state"]
        task_id = session["task_id"]

        is_correct = raw_reward == 1.0
        is_partial = 0.0 < raw_reward < 1.0
        state.correct_decisions += int(is_correct)
        state.partial_decisions += int(is_partial)
        state.decision_history.append({
            "step": state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "raw_reward": raw_reward,
            "reward": step_reward,
        })
        state.is_complete = True

        final_score = step_reward

        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            target_term_iast=ep["target_term_iast"],
            active_glossary={ep["target_term_iast"]: "See candidate options"},
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=step_reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )

    # ─────────────────────────────────────────────────────────────
    # Task 2 — Sandhi Resolution
    # ─────────────────────────────────────────────────────────────

    def _step_task2(self, action: ManuscriptAction, ep: dict, session: dict) -> ManuscriptObservation:
        raw_reward, feedback = self._sandhi_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )
        step_reward = self._shape_reward_signal(raw_reward)

        state = session["state"]
        task_id = session["task_id"]

        is_correct = raw_reward == 1.0
        is_partial = 0.0 < raw_reward < 1.0
        state.correct_decisions += int(is_correct)
        state.partial_decisions += int(is_partial)
        state.decision_history.append({
            "step": state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "raw_reward": raw_reward,
            "reward": step_reward,
        })
        state.is_complete = True

        final_score = step_reward

        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            compound_iast=ep["compound_iast"],
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=step_reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )

    # ─────────────────────────────────────────────────────────────
    # Task 3 — Referential Coherence (multi-step episode)
    # ─────────────────────────────────────────────────────────────

    def _step_task3(self, action: ManuscriptAction, ep: dict, session: dict) -> ManuscriptObservation:
        checkpoints = ep.get("consistency_checkpoints", [])
        state = session["state"]
        task_id = session["task_id"]

        # Are we in checkpoint phase?
        if session["t3_phase"] == "checkpoint" and session["t3_checkpoint_index"] < len(checkpoints):
            cp = checkpoints[session["t3_checkpoint_index"]]

            # Build the correct full option text for grading (same as _get_checkpoint_candidates)
            episode_options = ep.get("candidate_options", [])
            correct_full = next(
                (opt for opt in episode_options if opt.startswith(cp["answer"])),
                cp["answer"],
            )

            # Grade against the full option text so exact-match works
            raw_checkpoint_reward, cp_feedback = self._coherence_grader.grade_checkpoint(
                selected_option=action.selected_option,
                correct_answer=correct_full,
                candidate_options=episode_options,
            )
            session["t3_checkpoint_rewards"].append(raw_checkpoint_reward)
            session["t3_checkpoint_index"] += 1
            checkpoint_reward = self._shape_reward_signal(raw_checkpoint_reward)

            # Update consistency map
            state.consistency_map[cp["question"]] = action.selected_option

            # Advance verse index
            session["t3_verse_index"] = cp["after_verse"]

            # Check if all checkpoints done
            if session["t3_checkpoint_index"] >= len(checkpoints):
                session["t3_phase"] = "final"

            # Verses seen so far
            verses_so_far = ep["verses"][: session["t3_verse_index"]]

            # Next checkpoint or final question
            if session["t3_phase"] == "checkpoint":
                next_cp = checkpoints[session["t3_checkpoint_index"]]
                next_prompt = next_cp["question"]
                next_candidates = self._get_checkpoint_candidates(next_cp["answer"], ep)
            else:
                next_prompt = ep["referential_question"]
                next_candidates = ep["candidate_options"]

            # Shuffle next candidates
            next_candidates = self._shuffle_options(next_candidates, ep, session)

            return ManuscriptObservation(
                task_id=task_id,
                episode_id=state.episode_id,
                source_text_iast=ep["verses"][session["t3_verse_index"] - 1]["iast"] if verses_so_far else "",
                source_text_devanagari=ep["verses"][session["t3_verse_index"] - 1].get("devanagari", "") if verses_so_far else "",
                english_context=ep.get("title", ""),
                domain=ep.get("domain", "narrative"),
                verses_so_far=[
                    {"verse_num": v["verse_num"], "iast": v["iast"], "english": v["english"]}
                    for v in verses_so_far
                ],
                current_verse_num=session["t3_verse_index"],
                decision_prompt=next_prompt,
                candidate_options=next_candidates,
                step_reward=checkpoint_reward,
                cumulative_score=self._compute_t3_cumulative_score(session),
                feedback_message=cp_feedback,
                consistency_history=[
                    {"question": q, "answer": a}
                    for q, a in state.consistency_map.items()
                ],
                done=False,
                reward=None,
            )

        else:
            # Final referential question
            raw_final_reward, final_feedback = self._coherence_grader.grade_final(
                selected_option=action.selected_option,
                correct_answer=ep["correct_answer"],
                candidate_options=ep["candidate_options"],
            )
            step_reward = self._shape_reward_signal(raw_final_reward)

            raw_episode_score = self._coherence_grader.compute_episode_score(
                final_reward=raw_final_reward,
                checkpoint_rewards=session["t3_checkpoint_rewards"],
            )
            episode_score = self._shape_reward_signal(raw_episode_score)

            state.correct_decisions += int(raw_final_reward > 0)
            state.is_complete = True
            state.decision_history.append({
                "step": state.step_count,
                "selected": action.selected_option,
                "correct": ep["correct_answer"],
                "raw_reward": raw_final_reward,
                "reward": step_reward,
                "episode_score": episode_score,
            })

            all_verses = ep["verses"]
            return ManuscriptObservation(
                task_id=task_id,
                episode_id=state.episode_id,
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
                step_reward=step_reward,
                cumulative_score=episode_score,
                feedback_message=final_feedback,
                consistency_history=[
                    {"question": q, "answer": a}
                    for q, a in state.consistency_map.items()
                ],
                done=True,
                reward=episode_score,
            )


    # ─────────────────────────────────────────────────────────────
    # Task 4 — Samasa Classification
    # ─────────────────────────────────────────────────────────────

    def _step_task4(self, action: ManuscriptAction, ep: dict, session: dict) -> ManuscriptObservation:
        raw_reward, feedback = self._samasa_grader.grade(
            selected_option=action.selected_option,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep["partial_credit_indices"],
        )
        step_reward = self._shape_reward_signal(raw_reward)

        state = session["state"]
        task_id = session["task_id"]

        is_correct = raw_reward == 1.0
        is_partial = 0.0 < raw_reward < 1.0
        state.correct_decisions += int(is_correct)
        state.partial_decisions += int(is_partial)
        state.decision_history.append({
            "step": state.step_count,
            "selected": action.selected_option,
            "correct": ep["correct_answer"],
            "raw_reward": raw_reward,
            "reward": step_reward,
        })
        state.is_complete = True

        final_score = step_reward

        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            compound_iast=ep.get("compound_iast"),
            decision_prompt=ep["decision_prompt"],
            candidate_options=ep["candidate_options"],
            step_reward=step_reward,
            cumulative_score=final_score,
            feedback_message=feedback,
            done=True,
            reward=final_score,
        )
    # ─────────────────────────────────────────────────────────────
    # Task 5 — Manuscript Restoration (POMDP)
    # ─────────────────────────────────────────────────────────────

    def _step_task5_restoration(
        self, action: ManuscriptAction, ep: dict, session: dict
    ) -> ManuscriptObservation:
        """Handle tool_call or commit action for manuscript restoration."""
        state = session["state"]
        task_id = session["task_id"]
        action_type = action.action_type or "commit"

        if action_type == "tool_call":
            return self._step_t5_tool_call(action, ep, session)
        else:
            return self._step_t5_commit(action, ep, session)

    def _step_t5_tool_call(
        self, action: ManuscriptAction, ep: dict, session: dict
    ) -> ManuscriptObservation:
        """Process a tool_call action for Task 5."""
        state = session["state"]
        task_id = session["task_id"]
        tool_name = action.tool_name or ""
        tool_input = action.tool_input or ""

        # Dispatch tool
        tool_output = self._toolkit.dispatch(tool_name, tool_input, ep)

        # Record in history
        step_entry = {
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
            "step_num": state.step_count,
        }
        session["t5_tool_history"].append(step_entry)
        session["t5_steps_remaining"] -= 1

        # Track relevant tools used
        tools_needed = set(ep.get("tools_needed", []))
        if tool_name in tools_needed:
            session["t5_relevant_tools_used"].add(tool_name)

        # Grade tool call
        ep_with_workflow = dict(ep)
        ep_with_workflow["_workflow_pairs_awarded"] = session["t5_workflow_pairs_awarded"]
        tool_reward, tool_feedback = self._restoration_grader.grade_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            episode=ep_with_workflow,
            history=session["t5_tool_history"][:-1],  # exclude current
        )
        session["t5_tool_rewards"].append(tool_reward)

        state.decision_history.append({
            "step": state.step_count,
            "action_type": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_reward": tool_reward,
        })

        # If budget exhausted, force commit with empty answer
        if session["t5_steps_remaining"] <= 0:
            return self._step_t5_commit(
                ManuscriptAction(action_type="commit", final_answer=""),
                ep, session,
            )

        # Get passage text (noisy at higher difficulty)
        passage = ep.get("passage_iast_noisy", ep.get("passage_iast", ""))

        shuffled_options = self._shuffle_options(ep["candidate_options"], ep, session)

        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=passage,
            source_text_devanagari=ep.get("passage_devanagari", ""),
            english_context=ep.get("passage_english_gloss", ""),
            domain=ep.get("domain", ""),
            decision_prompt=f"You have {session['t5_steps_remaining']} tool calls remaining. Use a tool or commit your answer.",
            candidate_options=shuffled_options,
            step_reward=self._shape_reward_signal(tool_reward),
            cumulative_score=0.0,
            feedback_message=tool_feedback,
            done=False,
            reward=None,
            tool_call_history=session["t5_tool_history"],
            steps_remaining=session["t5_steps_remaining"],
            available_tools=ManuscriptToolkit.TOOL_NAMES,
            last_tool_output=tool_output,
            ocr_noise_level=ep.get("ocr_noise_level", 0.0),
            difficulty=session["t5_difficulty"],
        )

    def _step_t5_commit(
        self, action: ManuscriptAction, ep: dict, session: dict
    ) -> ManuscriptObservation:
        """Process a commit action for Task 5."""
        state = session["state"]
        task_id = session["task_id"]
        final_answer = action.final_answer or action.selected_option or ""

        # Grade commit
        raw_reward, feedback = self._restoration_grader.grade_commit(
            final_answer=final_answer,
            correct_answer=ep["correct_answer"],
            candidate_options=ep["candidate_options"],
            partial_credit_indices=ep.get("partial_credit_indices", []),
            tool_history=session["t5_tool_history"],
            tool_budget=ep.get("tool_budget", 8),
            tools_needed=ep.get("tools_needed", []),
        )

        # Shape the terminal reward
        episode_score = self._shape_reward_signal(raw_reward)

        state.is_complete = True
        state.correct_decisions += int(raw_reward >= 0.9)
        state.decision_history.append({
            "step": state.step_count,
            "action_type": "commit",
            "final_answer": final_answer,
            "raw_reward": raw_reward,
            "episode_score": episode_score,
        })

        passage = ep.get("passage_iast_noisy", ep.get("passage_iast", ""))
        shuffled_options = self._shuffle_options(ep["candidate_options"], ep, session)

        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=passage,
            source_text_devanagari=ep.get("passage_devanagari", ""),
            english_context=ep.get("passage_english_gloss", ""),
            domain=ep.get("domain", ""),
            decision_prompt="Episode complete.",
            candidate_options=shuffled_options,
            step_reward=episode_score,
            cumulative_score=episode_score,
            feedback_message=feedback,
            done=True,
            reward=episode_score,
            tool_call_history=session["t5_tool_history"],
            steps_remaining=session["t5_steps_remaining"],
            available_tools=ManuscriptToolkit.TOOL_NAMES,
            ocr_noise_level=ep.get("ocr_noise_level", 0.0),
            difficulty=session["t5_difficulty"],
        )

    def _build_restoration_observation(
        self, ep: dict, session: dict
    ) -> ManuscriptObservation:
        """Build the initial observation for a manuscript_restoration episode."""
        state = session["state"]
        passage = ep.get("passage_iast_noisy", ep.get("passage_iast", ""))
        shuffled_options = self._shuffle_options(ep["candidate_options"], ep, session)

        return ManuscriptObservation(
            task_id="manuscript_restoration",
            episode_id=state.episode_id,
            source_text_iast=passage,
            source_text_devanagari=ep.get("passage_devanagari", ""),
            english_context=ep.get("passage_english_gloss", ""),
            domain=ep.get("domain", ""),
            decision_prompt=(
                f"Interpret this Sanskrit passage. You have {session['t5_steps_remaining']} "
                f"tool calls available. Use tools to gather evidence, then commit your answer."
            ),
            candidate_options=shuffled_options,
            step_reward=0.0,
            cumulative_score=0.0,
            feedback_message="New manuscript restoration episode. Use tools to gather evidence before committing.",
            done=False,
            reward=None,
            tool_call_history=[],
            steps_remaining=session["t5_steps_remaining"],
            available_tools=ManuscriptToolkit.TOOL_NAMES,
            ocr_noise_level=ep.get("ocr_noise_level", 0.0),
            difficulty=session["t5_difficulty"],
        )

    def _auto_select_difficulty(self, recent_scores: list) -> str:
        """Select difficulty based on mastery threshold curriculum.

        escalate if:   mean(last 10) > 0.80 AND len >= 5
        deescalate if: mean(last 10) < 0.45 AND len >= 3
        """
        if not recent_scores or len(recent_scores) < 3:
            return "beginner"

        avg = sum(recent_scores[-10:]) / len(recent_scores[-10:])
        current_idx = 0  # default beginner

        if avg > 0.80 and len(recent_scores) >= 5:
            current_idx = min(current_idx + 1, len(self.DIFFICULTY_LADDER) - 1)
        elif avg < 0.45:
            current_idx = max(current_idx - 1, 0)

        return self.DIFFICULTY_LADDER[current_idx]

    # ─────────────────────────────────────────────────────────────
    # Task 6 — Full Manuscript Session (Long-Horizon)
    # ─────────────────────────────────────────────────────────────

    def _step_full_manuscript_session(
        self, action: ManuscriptAction, ep: dict, session: dict
    ) -> ManuscriptObservation:
        """Handle one step of a full manuscript session (chains all 5 task types)."""
        state = session["state"]
        phases = ep.get("phases", [])
        phase_idx = session["t6_phase_index"]

        if phase_idx >= len(phases):
            # Session already complete
            return self._build_t6_done_observation(ep, session)

        current_phase = phases[phase_idx]
        phase_type = current_phase["phase"]

        # Restoration phase uses multi-step tool loop
        if phase_type == "restoration":
            return self._step_t6_restoration_subphase(action, ep, session, current_phase)

        # All other phases are single-step MCQ
        return self._step_t6_mcq_subphase(action, ep, session, current_phase)

    def _step_t6_mcq_subphase(
        self, action: ManuscriptAction, ep: dict, session: dict, phase: dict
    ) -> ManuscriptObservation:
        """Handle a single MCQ sub-phase (glossary, sandhi, samasa, coherence)."""
        state = session["state"]
        selected = action.selected_option or ""
        correct = phase["correct"]
        options = phase.get("options", [])
        partial_idx = phase.get("partial_idx", [])

        # Grade
        if selected == correct:
            raw_reward = 1.0
            feedback = "Correct."
        elif options.index(selected) in partial_idx if selected in options else False:
            raw_reward = 0.4
            feedback = "Partial credit."
        else:
            raw_reward = 0.0
            feedback = f"Wrong. Correct: {correct}"

        step_reward = self._shape_reward_signal(raw_reward)
        session["t6_phase_rewards"].append(step_reward)
        session["t6_phase_answers"].append({"phase": phase["phase"], "answer": selected, "correct": correct, "reward": step_reward})

        # Check consistency violations
        contradiction_pairs = ep.get("contradiction_pairs", [])
        for pair in contradiction_pairs:
            if len(pair) == 2:
                a_text, b_text = pair
                prev_answers = [a["answer"] for a in session["t6_phase_answers"][:-1]]
                for prev in prev_answers:
                    if (a_text.lower() in prev.lower() and b_text.lower() in selected.lower()) or \
                       (b_text.lower() in prev.lower() and a_text.lower() in selected.lower()):
                        session["t6_consistency_violations"] += 1

        state.correct_decisions += int(raw_reward == 1.0)
        state.partial_decisions += int(0 < raw_reward < 1.0)
        state.decision_history.append({
            "step": state.step_count,
            "phase": phase["phase"],
            "selected": selected,
            "correct": correct,
            "raw_reward": raw_reward,
        })

        # Advance to next phase
        session["t6_phase_index"] += 1

        # Check if session is done
        if session["t6_phase_index"] >= len(ep.get("phases", [])):
            return self._build_t6_done_observation(ep, session)

        # Build observation for next phase
        return self._build_t6_phase_observation(ep, session)

    def _step_t6_restoration_subphase(
        self, action: ManuscriptAction, ep: dict, session: dict, phase: dict
    ) -> ManuscriptObservation:
        """Handle the restoration sub-phase within a full session."""
        state = session["state"]
        action_type = action.action_type or "commit"

        if action_type == "tool_call":
            tool_name = action.tool_name or ""
            tool_input = action.tool_input or ""
            tool_output = self._toolkit.dispatch(tool_name, tool_input, phase)

            session["t5_tool_history"].append({
                "tool": tool_name, "input": tool_input,
                "output": tool_output, "step_num": state.step_count,
            })
            session["t5_steps_remaining"] -= 1

            tools_needed = set(phase.get("tools_needed", []))
            if tool_name in tools_needed:
                session["t5_relevant_tools_used"].add(tool_name)

            ep_ctx = dict(phase)
            ep_ctx["_workflow_pairs_awarded"] = session["t5_workflow_pairs_awarded"]
            tool_reward, tool_feedback = self._restoration_grader.grade_tool_call(
                tool_name, tool_input, tool_output, ep_ctx,
                session["t5_tool_history"][:-1],
            )
            session["t5_tool_rewards"].append(tool_reward)

            state.decision_history.append({
                "step": state.step_count, "action_type": "tool_call",
                "tool_name": tool_name, "tool_reward": tool_reward,
            })

            if session["t5_steps_remaining"] <= 0:
                return self._step_t6_restoration_subphase(
                    ManuscriptAction(action_type="commit", final_answer=""),
                    ep, session, phase,
                )

            # Return observation for continued tool use
            options = self._shuffle_options(phase.get("options", []), ep, session)
            return ManuscriptObservation(
                task_id="full_manuscript_session",
                episode_id=state.episode_id,
                source_text_iast=phase.get("passage_iast", ""),
                source_text_devanagari=phase.get("passage_dev", ""),
                english_context=phase.get("gloss", ""),
                domain=ep.get("domain", ""),
                decision_prompt=f"Restoration phase. {session['t5_steps_remaining']} tool calls remaining.",
                candidate_options=options,
                step_reward=self._shape_reward_signal(tool_reward),
                cumulative_score=0.0,
                feedback_message=tool_feedback,
                done=False, reward=None,
                tool_call_history=session["t5_tool_history"],
                steps_remaining=session["t5_steps_remaining"],
                available_tools=ManuscriptToolkit.TOOL_NAMES,
                last_tool_output=tool_output,
                difficulty="expert",
            )

        # COMMIT
        final_answer = action.final_answer or action.selected_option or ""
        raw_reward, feedback = self._restoration_grader.grade_commit(
            final_answer=final_answer,
            correct_answer=phase["correct"],
            candidate_options=phase.get("options", []),
            partial_credit_indices=phase.get("partial_idx", []),
            tool_history=session["t5_tool_history"],
            tool_budget=phase.get("tool_budget", 4),
            tools_needed=phase.get("tools_needed", []),
        )
        step_reward = self._shape_reward_signal(raw_reward)
        session["t6_phase_rewards"].append(step_reward)
        session["t6_phase_answers"].append({
            "phase": "restoration", "answer": final_answer,
            "correct": phase["correct"], "reward": step_reward,
        })

        state.correct_decisions += int(raw_reward >= 0.9)
        state.decision_history.append({
            "step": state.step_count, "action_type": "commit",
            "final_answer": final_answer, "raw_reward": raw_reward,
        })

        # Advance past restoration phase
        session["t6_phase_index"] += 1
        return self._build_t6_done_observation(ep, session)

    def _build_t6_done_observation(self, ep: dict, session: dict) -> ManuscriptObservation:
        """Build the final observation for a completed full session."""
        state = session["state"]
        state.is_complete = True

        # Compute final score: mean of phase rewards with consistency penalty
        phase_rewards = session["t6_phase_rewards"]
        if phase_rewards:
            base_score = sum(phase_rewards) / len(phase_rewards)
        else:
            base_score = 0.0

        # Consistency penalty: -0.05 per violation
        consistency_penalty = 0.05 * session["t6_consistency_violations"]
        final_score = max(0.0, base_score - consistency_penalty)
        # Apply consistency bonus from grader if no violations
        if session["t6_consistency_violations"] == 0 and len(phase_rewards) >= 3:
            final_score = min(final_score + 0.05, 0.95)

        return ManuscriptObservation(
            task_id="full_manuscript_session",
            episode_id=state.episode_id,
            source_text_iast=ep.get("source", ""),
            source_text_devanagari="",
            english_context=f"Full session on {ep.get('source', '')}",
            domain=ep.get("domain", ""),
            decision_prompt="Session complete.",
            candidate_options=[],
            step_reward=final_score,
            cumulative_score=final_score,
            feedback_message=f"Session complete. {len(phase_rewards)} phases. {session['t6_consistency_violations']} consistency violations.",
            done=True, reward=final_score,
        )

    def _build_t6_phase_observation(self, ep: dict, session: dict) -> ManuscriptObservation:
        """Build observation for the next phase in a full session."""
        state = session["state"]
        phases = ep.get("phases", [])
        phase_idx = session["t6_phase_index"]
        phase = phases[phase_idx]

        options = self._shuffle_options(phase.get("options", []), ep, session)

        # For restoration phase, add tool-use context
        if phase["phase"] == "restoration":
            return ManuscriptObservation(
                task_id="full_manuscript_session",
                episode_id=state.episode_id,
                source_text_iast=phase.get("passage_iast", ""),
                source_text_devanagari=phase.get("passage_dev", ""),
                english_context=phase.get("gloss", ""),
                domain=ep.get("domain", ""),
                decision_prompt=f"Phase {phase_idx+1}/{len(phases)}: Restoration. Use tools then commit. Budget: {session['t5_steps_remaining']}",
                candidate_options=options,
                step_reward=0.0, cumulative_score=0.0,
                feedback_message=f"Entering restoration phase.",
                done=False, reward=None,
                tool_call_history=session["t5_tool_history"],
                steps_remaining=session["t5_steps_remaining"],
                available_tools=ManuscriptToolkit.TOOL_NAMES,
                difficulty="expert",
            )

        # MCQ phase
        return ManuscriptObservation(
            task_id="full_manuscript_session",
            episode_id=state.episode_id,
            source_text_iast=phase.get("passage_iast", ""),
            source_text_devanagari=phase.get("passage_dev", ""),
            english_context="",
            domain=ep.get("domain", ""),
            decision_prompt=f"Phase {phase_idx+1}/{len(phases)} ({phase['phase']}): {phase.get('prompt', '')}",
            candidate_options=options,
            step_reward=0.0, cumulative_score=0.0,
            feedback_message=f"Phase {phase_idx+1}: {phase['phase']}",
            done=False, reward=None,
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
        elif task_id == "manuscript_restoration":
            return self._task5_data["episodes"]
        elif task_id == "full_manuscript_session":
            return self._task6_data["episodes"]
        return self._task1_data["episodes"]

    def _count_total_decisions(self, ep: dict, task_id: str) -> int:
        if task_id == "referential_coherence":
            return len(ep.get("consistency_checkpoints", [])) + 1
        if task_id == "manuscript_restoration":
            return ep.get("tool_budget", 8) + 1  # tools + commit
        if task_id == "full_manuscript_session":
            phases = ep.get("phases", [])
            rest = next((p for p in phases if p["phase"] == "restoration"), None)
            return len(phases) - 1 + (rest.get("tool_budget", 4) + 1 if rest else 1)
        return 1

    def _shuffle_options(self, options: list, ep: dict, session: dict) -> list:
        """Shuffle candidate_options using a deterministic seed from episode_id + session_id.

        This prevents option-position memorization: the same episode gives
        different option orders in different sessions, but the same order
        within a session (reproducible).
        """
        episode_id = ep.get("id", "")
        session_id = session["state"].episode_id or ""
        seed_str = f"{episode_id}:{session_id}"
        seed_int = int.from_bytes(seed_str.encode("utf-8")[:8], "big")
        rng = random.Random(seed_int)
        shuffled = list(options)
        rng.shuffle(shuffled)
        return shuffled

    def _build_initial_observation(self, ep: dict, session: dict) -> ManuscriptObservation:
        state = session["state"]
        task_id = session["task_id"]

        if task_id == "manuscript_restoration":
            return self._build_restoration_observation(ep, session)

        if task_id == "full_manuscript_session":
            return self._build_t6_phase_observation(ep, session)

        if task_id == "referential_coherence":
            checkpoints = ep.get("consistency_checkpoints", [])
            if checkpoints:
                first_cp = checkpoints[0]
                prompt = first_cp["question"]
                candidates = self._get_checkpoint_candidates(first_cp["answer"], ep)
                verse_index = first_cp["after_verse"]
            else:
                prompt = ep["referential_question"]
                candidates = self._shuffle_options(ep["candidate_options"], ep, session)
                verse_index = len(ep["verses"])
                session["t3_phase"] = "final"

            # Shuffle checkpoint candidates
            candidates = self._shuffle_options(candidates, ep, session)

            verses_so_far = ep["verses"][:verse_index]
            return ManuscriptObservation(
                task_id=task_id,
                episode_id=state.episode_id,
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

        # Tasks 1, 2, 4 — single decision episodes
        shuffled_options = self._shuffle_options(ep["candidate_options"], ep, session)
        return ManuscriptObservation(
            task_id=task_id,
            episode_id=state.episode_id,
            source_text_iast=ep["source_text_iast"],
            source_text_devanagari=ep["source_text_devanagari"],
            english_context=ep["english_context"],
            domain=ep["domain"],
            target_term_iast=ep.get("target_term_iast"),
            compound_iast=ep.get("compound_iast"),
            active_glossary={ep.get("target_term_iast", ""): "See candidate options"} if ep.get("target_term_iast") else None,
            decision_prompt=ep["decision_prompt"],
            candidate_options=shuffled_options,
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

    def _compute_t3_cumulative_score(self, session: dict) -> float:
        raw_checkpoint_credit = sum(session["t3_checkpoint_rewards"])
        max_possible_credit = (
            self._coherence_grader.MAIN_CORRECT
            + self._coherence_grader.CHECKPOINT_CORRECT * len(session["t3_checkpoint_rewards"])
        )
        if max_possible_credit == 0:
            return 0.0
        return self._shape_reward_signal(min(raw_checkpoint_credit / max_possible_credit, 1.0))
