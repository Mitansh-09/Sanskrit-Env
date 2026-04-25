"""
Task 5 Grader — Manuscript Restoration.

Deterministic. No LLM. No semantic similarity.
Implements the full multi-component reward function for the POMDP:
  R1: Per-step tool reward (relevance + workflow - redundancy)
  R2: Terminal commit reward (correctness × evidence - budget waste)
  R3: Final shaped episode reward

All methods are pure functions. See Changes.md Section 2.3 for the
mathematical specification.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


# ── Tool Relevance Matrix ────────────────────────────────────────────────────
# Appendix B from Changes.md. Defines which tools are relevant for which
# primary_disambiguation_type of the episode.
#
# Values: "PRIMARY", "SECOND", "support", "none"
TOOL_RELEVANCE = {
    "glossary": {
        "lexicon_lookup":    "PRIMARY",
        "sandhi_parser":     "support",
        "meter_checker":     "none",
        "commentary_fetch":  "SECOND",
        "witness_compare":   "support",
        "referent_tracker":  "none",
    },
    "sandhi": {
        "lexicon_lookup":    "support",
        "sandhi_parser":     "PRIMARY",
        "meter_checker":     "SECOND",
        "commentary_fetch":  "none",
        "witness_compare":   "support",
        "referent_tracker":  "none",
    },
    "samasa": {
        "lexicon_lookup":    "support",
        "sandhi_parser":     "PRIMARY",
        "meter_checker":     "SECOND",
        "commentary_fetch":  "none",
        "witness_compare":   "none",
        "referent_tracker":  "none",
    },
    "coherence": {
        "lexicon_lookup":    "support",
        "sandhi_parser":     "support",
        "meter_checker":     "none",
        "commentary_fetch":  "support",
        "witness_compare":   "support",
        "referent_tracker":  "PRIMARY",
    },
}

# Workflow pairs that earn a bonus when used in sequence
WORKFLOW_PAIRS = [
    ("sandhi_parser", "meter_checker", 0.05),
    ("lexicon_lookup", "commentary_fetch", 0.05),
    ("witness_compare", "referent_tracker", 0.03),
]

# Tools needed per episode type (for evidence multiplier calculation)
TOOLS_NEEDED = {
    "glossary":  {"lexicon_lookup", "commentary_fetch"},
    "sandhi":    {"sandhi_parser", "meter_checker"},
    "coherence": {"referent_tracker", "witness_compare"},
    "samasa":    {"sandhi_parser", "lexicon_lookup"},
}

ALL_TOOL_NAMES = frozenset([
    "lexicon_lookup",
    "sandhi_parser",
    "meter_checker",
    "commentary_fetch",
    "witness_compare",
    "referent_tracker",
])


class RestorationGrader:
    """
    Grades ManuscriptAction for Task 5 (Manuscript Restoration).

    All methods are deterministic and pure — no LLM calls, no external API.
    """

    def grade_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: Dict[str, Any],
        episode: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Compute per-step tool reward (R1 from Section 2.3).

        Returns:
            (reward: float, feedback: str)
        """
        episode_type = episode.get("primary_disambiguation_type", "glossary")
        relevance_map = TOOL_RELEVANCE.get(episode_type, TOOL_RELEVANCE["glossary"])

        reward = 0.0
        feedback_parts: List[str] = []

        # ── Relevance bonus ──────────────────────────────────────────────
        relevance = relevance_map.get(tool_name, "none")
        if relevance == "PRIMARY":
            reward += 0.08
            feedback_parts.append(f"Primary tool for {episode_type} episode (+0.08)")
        elif relevance == "SECOND":
            # SECOND bonus only if PRIMARY was already used for same item
            primary_tools_used = {
                h["tool"] for h in history
                if relevance_map.get(h["tool"]) == "PRIMARY"
            }
            if primary_tools_used:
                reward += 0.05
                feedback_parts.append("Follow-up evidence tool (+0.05)")
            else:
                reward += 0.04
                feedback_parts.append("Supporting tool (+0.04)")
        elif relevance == "support":
            reward += 0.04
            feedback_parts.append("Supporting evidence tool (+0.04)")
        else:
            feedback_parts.append("Tool not relevant for this episode type")

        # ── Workflow bonus ───────────────────────────────────────────────
        workflow_bonus = self._compute_workflow_bonus_for_step(
            tool_name, history, episode.get("_workflow_pairs_awarded", set())
        )
        if workflow_bonus > 0:
            reward += workflow_bonus
            feedback_parts.append(f"Workflow sequence bonus (+{workflow_bonus:.2f})")

        # ── Redundancy penalty ───────────────────────────────────────────
        for prev in history:
            if prev["tool"] == tool_name and prev["input"] == tool_input:
                reward -= 0.05
                feedback_parts.append("Redundant call penalty (-0.05)")
                break

        # ── Irrelevance penalty ──────────────────────────────────────────
        irrelevance_penalty = self._check_irrelevance_penalty(
            tool_name, tool_output, episode
        )
        if irrelevance_penalty < 0:
            reward += irrelevance_penalty
            feedback_parts.append(
                f"Irrelevance penalty ({irrelevance_penalty:.2f})"
            )

        return round(reward, 4), " | ".join(feedback_parts)

    def grade_commit(
        self,
        final_answer: str,
        correct_answer: str,
        candidate_options: List[str],
        partial_credit_indices: List[int],
        tool_history: List[Dict[str, Any]],
        tool_budget: int,
        tools_needed: List[str],
    ) -> Tuple[float, str]:
        """
        Compute terminal commit reward (R2 from Section 2.3).

        Returns:
            (reward: float, feedback: str)
        """
        # ── r_correctness ────────────────────────────────────────────────
        r_base = 0.0
        feedback = ""

        if final_answer.strip() == correct_answer.strip():
            r_base = 1.0
            feedback = f"Correct interpretation: '{final_answer[:60]}...'"
        else:
            # Check partial credit
            matched_idx = None
            for i, opt in enumerate(candidate_options):
                if final_answer.strip() == opt.strip():
                    matched_idx = i
                    break

            if matched_idx is not None and matched_idx in partial_credit_indices:
                r_base = 0.40
                feedback = (
                    f"Partially correct. '{final_answer[:40]}...' is related "
                    f"but not the best interpretation."
                )
            else:
                r_base = 0.0
                feedback = (
                    f"Incorrect. The correct answer was: '{correct_answer[:60]}...'"
                )

        # For wrong answers, return 0.0 immediately — evidence never rescues wrong
        if r_base == 0.0:
            return 0.0, feedback

        # ── r_evidence_multiplier ────────────────────────────────────────
        evidence_mult = self.compute_evidence_multiplier(tool_history, tools_needed)

        # ── r_budget_waste ───────────────────────────────────────────────
        steps_used = len(tool_history)
        ideal_steps = len(tools_needed) + 1  # tools + commit
        budget_penalty = self.compute_budget_penalty(steps_used, ideal_steps, tool_budget)

        # ── r_terminal ───────────────────────────────────────────────────
        raw_terminal = r_base * evidence_mult - budget_penalty

        feedback += (
            f" | Evidence: {evidence_mult:.2f}"
            f" | Budget penalty: {budget_penalty:.3f}"
            f" | Raw terminal: {raw_terminal:.3f}"
        )

        return round(max(0.0, raw_terminal), 4), feedback

    def compute_evidence_multiplier(
        self,
        tool_history: List[Dict[str, Any]],
        tools_needed: List[str],
    ) -> float:
        """
        Compute the evidence multiplier M_evidence.

        M_evidence = 0.60 + 0.40 × (|relevant_tools_used| / |tools_needed|)
        Range: [0.60, 1.00]
        """
        needed_set = set(tools_needed)
        if not needed_set:
            return 1.0

        distinct_relevant = {
            h["tool"] for h in tool_history if h["tool"] in needed_set
        }
        evidence_score = len(distinct_relevant) / len(needed_set)
        return round(0.60 + 0.40 * evidence_score, 4)

    def compute_budget_penalty(
        self,
        steps_used: int,
        ideal_steps: int,
        tool_budget: int,
    ) -> float:
        """
        Compute the budget waste penalty P_budget.

        P_budget = 0.10 × max(0, steps_used - ideal_steps) / tool_budget
        Range: [0.00, 0.10]
        """
        if tool_budget <= 0:
            return 0.0
        waste_ratio = max(0, steps_used - ideal_steps) / tool_budget
        return round(0.10 * waste_ratio, 4)

    def compute_workflow_bonus(
        self,
        tool_history: List[Dict[str, Any]],
    ) -> float:
        """
        Compute total workflow bonus for the episode.
        Each workflow pair is awarded at most once.
        """
        awarded = set()
        bonus = 0.0

        tool_sequence = [h["tool"] for h in tool_history]
        for first, second, pair_bonus in WORKFLOW_PAIRS:
            pair_key = (first, second)
            if pair_key in awarded:
                continue
            # Check if first appears before second in the sequence
            try:
                first_idx = tool_sequence.index(first)
                second_idx = tool_sequence.index(second, first_idx + 1)
                if second_idx > first_idx:
                    bonus += pair_bonus
                    awarded.add(pair_key)
            except ValueError:
                continue

        return round(bonus, 4)

    def compute_episode_score(
        self,
        tool_rewards: List[float],
        commit_reward: float,
    ) -> float:
        """
        Compute final episode score.

        Per the spec: the episode total reward = R_T (terminal only).
        Per-step tool rewards are dense training signals but the final
        episode score comes from the terminal commit reward.
        """
        return commit_reward

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_workflow_bonus_for_step(
        self,
        tool_name: str,
        history: List[Dict[str, Any]],
        pairs_awarded: Set[tuple],
    ) -> float:
        """Check if this tool call completes a workflow pair."""
        bonus = 0.0
        tool_sequence = [h["tool"] for h in history]

        for first, second, pair_bonus in WORKFLOW_PAIRS:
            pair_key = (first, second)
            if pair_key in pairs_awarded:
                continue
            # This tool is the second in the pair, and first already used
            if tool_name == second and first in tool_sequence:
                bonus += pair_bonus
                pairs_awarded.add(pair_key)

        return bonus

    def _check_irrelevance_penalty(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        episode: Dict[str, Any],
    ) -> float:
        """
        Check for specific irrelevance penalties:
        - meter_checker on a prose passage (no meter data) → -0.05
        - witness_compare when witnesses agree (same reading) → -0.03
        """
        if tool_name == "meter_checker":
            meter_data = episode.get("meter_data", {})
            if not meter_data:
                return -0.05

        if tool_name == "witness_compare":
            witness_data = episode.get("witness_data", {})
            if not witness_data:
                return -0.03

        return 0.0
