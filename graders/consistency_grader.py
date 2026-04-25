"""
Consistency Grader — Cross-Phase Consistency Checking.

Used in Task 6 (full_manuscript_session) to reward consistent
decision-making across phases. Deterministic — checks pre-annotated
contradiction pairs in the episode data.
"""

from typing import Dict, List, Tuple


class ConsistencyGrader:
    """
    Checks cross-phase consistency in a full manuscript session.

    Consistency bonus ranges from 0.0 to 0.10.
    Each contradiction detected reduces the bonus by 0.03.
    """

    MAX_BONUS = 0.10
    PENALTY_PER_VIOLATION = 0.03

    def check_cross_phase_consistency(
        self,
        decision_history: List[Dict[str, str]],
        contradiction_pairs: List[Dict[str, str]],
    ) -> Tuple[float, str]:
        """
        Check cross-phase consistency against pre-annotated contradiction pairs.

        Args:
            decision_history: List of {"decision_id": str, "answer": str}
                              representing the agent's answers across phases.
            contradiction_pairs: List of dicts, each with:
                {
                    "decision_a": str,  # decision_id of first decision
                    "answer_a": str,    # the specific answer that forms
                    "decision_b": str,  # decision_id of second decision
                    "answer_b": str,    # the specific answer that contradicts
                    "contradicts": bool
                }

        Returns:
            (bonus: float, explanation: str)
        """
        if not contradiction_pairs:
            return self.MAX_BONUS, "No contradiction pairs to check; full bonus."

        # Build lookup: decision_id → answer
        chosen = {}
        for entry in decision_history:
            chosen[entry.get("decision_id", "")] = entry.get("answer", "")

        violations = 0
        violation_details: List[str] = []

        for pair in contradiction_pairs:
            if not pair.get("contradicts", False):
                continue

            decision_a = pair.get("decision_a", "")
            answer_a = pair.get("answer_a", "")
            decision_b = pair.get("decision_b", "")
            answer_b = pair.get("answer_b", "")

            # Check if the agent chose the exact answers that form a contradiction
            agent_a = chosen.get(decision_a, "")
            agent_b = chosen.get(decision_b, "")

            if agent_a == answer_a and agent_b == answer_b:
                violations += 1
                violation_details.append(
                    f"Contradiction: {decision_a}='{answer_a[:30]}' ⟷ "
                    f"{decision_b}='{answer_b[:30]}'"
                )

        bonus = max(0.0, self.MAX_BONUS - self.PENALTY_PER_VIOLATION * violations)

        if violations == 0:
            explanation = (
                f"Fully consistent across {len(decision_history)} decisions. "
                f"Bonus: +{bonus:.2f}"
            )
        else:
            explanation = (
                f"{violations} contradiction(s) detected. "
                f"Bonus: +{bonus:.2f}. "
                + " | ".join(violation_details)
            )

        return round(bonus, 4), explanation
