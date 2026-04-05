"""
Task 4 Grader — Samasa (Compound) Classification.

Deterministic. No LLM. No semantic similarity.
Exact string match against pre-annotated answer keys.

Samasa types tested:
  - Tatpurusha     (determinative compound)
  - Karmadharaya   (descriptive tatpurusha — same-referent adjective+noun)
  - Dvigu          (numerical tatpurusha — numeral + noun collective)
  - Dvandva        (copulative compound — A and B)
  - Bahuvrihi      (possessive/exocentric — describes external referent)
  - Avyayibhava    (adverbial indeclinable compound)
"""

from typing import Tuple


class SamasaGrader:
    """
    Grades ManuscriptAction for Task 4 (Samasa Classification).

    Scoring:
    - Correct samasa type:           1.00
    - Partial credit option:         0.40  (adjacent type, e.g. karmadharaya vs tatpurusha)
    - Wrong type:                    0.00
    - Invalid option (not in list):  0.00
    """

    FULL_CREDIT = 1.00
    PARTIAL_CREDIT = 0.40
    NO_CREDIT = 0.00

    def grade(
        self,
        selected_option: str,
        correct_answer: str,
        candidate_options: list,
        partial_credit_indices: list,
    ) -> Tuple[float, str]:
        """
        Grade a single samasa classification decision.

        Returns:
            (reward: float, feedback: str)
        """
        # Validate option is in candidate list
        if selected_option not in candidate_options:
            return (
                self.NO_CREDIT,
                "Invalid selection. Must choose from the provided options exactly as written."
            )

        # Full credit
        if selected_option.strip() == correct_answer.strip():
            return (
                self.FULL_CREDIT,
                f"Correct. '{selected_option}' is the right samasa classification for this compound."
            )

        # Partial credit
        selected_index = candidate_options.index(selected_option)
        if selected_index in partial_credit_indices:
            return (
                self.PARTIAL_CREDIT,
                f"Partially correct. '{selected_option}' is a related compound type, "
                f"but not the most precise classification. "
                f"The best answer was: '{correct_answer}'."
            )

        # No credit
        return (
            self.NO_CREDIT,
            f"Incorrect. '{selected_option}' does not correctly classify this compound. "
            f"The correct samasa type was: '{correct_answer}'."
        )
