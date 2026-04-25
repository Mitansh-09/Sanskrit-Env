"""Unit tests for RestorationGrader."""

import unittest
from graders.restoration_grader import RestorationGrader


class TestRestorationGrader(unittest.TestCase):
    def setUp(self):
        self.grader = RestorationGrader()
        self.episode = {
            "primary_disambiguation_type": "glossary",
            "tools_needed": ["lexicon_lookup", "commentary_fetch"],
            "tool_budget": 8,
            "correct_answer": "The correct answer",
            "candidate_options": [
                "The correct answer",
                "Partial answer",
                "Wrong answer 1",
                "Wrong answer 2",
            ],
            "partial_credit_indices": [1],
            "glossary_data": {"term": [{"meaning": "test", "domain": "general", "confidence": 0.9}]},
            "sandhi_data": {},
            "meter_data": {},
            "commentary_data": {"term": "Test commentary"},
            "witness_data": {},
            "entity_map": {},
            "_workflow_pairs_awarded": set(),
        }

    # ── Evidence multiplier tests ────────────────────────────────────────

    def test_evidence_multiplier_zero_tools(self):
        mult = self.grader.compute_evidence_multiplier([], ["lexicon_lookup", "commentary_fetch"])
        self.assertAlmostEqual(mult, 0.60)

    def test_evidence_multiplier_one_tool(self):
        history = [{"tool": "lexicon_lookup", "input": "x", "output": {}}]
        mult = self.grader.compute_evidence_multiplier(history, ["lexicon_lookup", "commentary_fetch"])
        self.assertAlmostEqual(mult, 0.80)

    def test_evidence_multiplier_all_tools(self):
        history = [
            {"tool": "lexicon_lookup", "input": "x", "output": {}},
            {"tool": "commentary_fetch", "input": "y", "output": {}},
        ]
        mult = self.grader.compute_evidence_multiplier(history, ["lexicon_lookup", "commentary_fetch"])
        self.assertAlmostEqual(mult, 1.00)

    def test_evidence_multiplier_empty_needed(self):
        mult = self.grader.compute_evidence_multiplier([], [])
        self.assertAlmostEqual(mult, 1.0)

    # ── Budget penalty tests ─────────────────────────────────────────────

    def test_budget_penalty_ideal_steps(self):
        # ideal_steps = 2 tools + 1 commit = 3, used 3 steps
        penalty = self.grader.compute_budget_penalty(steps_used=3, ideal_steps=3, tool_budget=8)
        self.assertAlmostEqual(penalty, 0.0)

    def test_budget_penalty_wasted_steps(self):
        penalty = self.grader.compute_budget_penalty(steps_used=8, ideal_steps=3, tool_budget=8)
        expected = 0.10 * (8 - 3) / 8  # 0.0625
        self.assertAlmostEqual(penalty, round(expected, 4))

    def test_budget_penalty_zero_budget(self):
        penalty = self.grader.compute_budget_penalty(steps_used=5, ideal_steps=3, tool_budget=0)
        self.assertAlmostEqual(penalty, 0.0)

    # ── Workflow bonus tests ─────────────────────────────────────────────

    def test_workflow_bonus_complete_pair(self):
        history = [
            {"tool": "sandhi_parser", "input": "x", "output": {}},
            {"tool": "meter_checker", "input": "y", "output": {}},
        ]
        bonus = self.grader.compute_workflow_bonus(history)
        self.assertAlmostEqual(bonus, 0.05)

    def test_workflow_bonus_deduplication(self):
        history = [
            {"tool": "sandhi_parser", "input": "x", "output": {}},
            {"tool": "meter_checker", "input": "y", "output": {}},
            {"tool": "sandhi_parser", "input": "z", "output": {}},
            {"tool": "meter_checker", "input": "w", "output": {}},
        ]
        bonus = self.grader.compute_workflow_bonus(history)
        # Only awarded once per pair
        self.assertAlmostEqual(bonus, 0.05)

    def test_workflow_bonus_wrong_order(self):
        history = [
            {"tool": "meter_checker", "input": "y", "output": {}},
            {"tool": "sandhi_parser", "input": "x", "output": {}},
        ]
        bonus = self.grader.compute_workflow_bonus(history)
        self.assertAlmostEqual(bonus, 0.0)

    # ── Commit grading tests ─────────────────────────────────────────────

    def test_commit_correct_with_evidence(self):
        history = [
            {"tool": "lexicon_lookup", "input": "term", "output": {}},
            {"tool": "commentary_fetch", "input": "term", "output": {}},
        ]
        reward, _ = self.grader.grade_commit(
            final_answer="The correct answer",
            correct_answer="The correct answer",
            candidate_options=self.episode["candidate_options"],
            partial_credit_indices=[1],
            tool_history=history,
            tool_budget=8,
            tools_needed=["lexicon_lookup", "commentary_fetch"],
        )
        self.assertGreater(reward, 0.8)

    def test_commit_correct_without_evidence(self):
        reward, _ = self.grader.grade_commit(
            final_answer="The correct answer",
            correct_answer="The correct answer",
            candidate_options=self.episode["candidate_options"],
            partial_credit_indices=[1],
            tool_history=[],
            tool_budget=8,
            tools_needed=["lexicon_lookup", "commentary_fetch"],
        )
        # Correct but no evidence: should be lower
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.7)

    def test_commit_correct_with_evidence_beats_without(self):
        history = [
            {"tool": "lexicon_lookup", "input": "x", "output": {}},
            {"tool": "commentary_fetch", "input": "y", "output": {}},
        ]
        with_ev, _ = self.grader.grade_commit(
            "The correct answer", "The correct answer",
            self.episode["candidate_options"], [1], history, 8,
            ["lexicon_lookup", "commentary_fetch"],
        )
        without_ev, _ = self.grader.grade_commit(
            "The correct answer", "The correct answer",
            self.episode["candidate_options"], [1], [], 8,
            ["lexicon_lookup", "commentary_fetch"],
        )
        self.assertGreater(with_ev, without_ev)

    def test_commit_wrong_returns_zero(self):
        history = [
            {"tool": "lexicon_lookup", "input": "x", "output": {}},
            {"tool": "commentary_fetch", "input": "y", "output": {}},
        ]
        reward, _ = self.grader.grade_commit(
            final_answer="Wrong answer 1",
            correct_answer="The correct answer",
            candidate_options=self.episode["candidate_options"],
            partial_credit_indices=[1],
            tool_history=history,
            tool_budget=8,
            tools_needed=["lexicon_lookup", "commentary_fetch"],
        )
        self.assertEqual(reward, 0.0)

    def test_commit_partial_credit(self):
        reward, _ = self.grader.grade_commit(
            final_answer="Partial answer",
            correct_answer="The correct answer",
            candidate_options=self.episode["candidate_options"],
            partial_credit_indices=[1],
            tool_history=[],
            tool_budget=8,
            tools_needed=["lexicon_lookup", "commentary_fetch"],
        )
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.5)

    # ── Episode score tests ──────────────────────────────────────────────

    def test_episode_score_equals_commit_reward(self):
        score = self.grader.compute_episode_score(
            tool_rewards=[0.08, 0.04],
            commit_reward=0.85,
        )
        self.assertEqual(score, 0.85)

    # ── Tool call grading tests ──────────────────────────────────────────

    def test_tool_call_primary_relevance(self):
        reward, feedback = self.grader.grade_tool_call(
            tool_name="lexicon_lookup",
            tool_input="term",
            tool_output={"found": True},
            episode=self.episode,
            history=[],
        )
        self.assertGreaterEqual(reward, 0.08)
        self.assertIn("Primary", feedback)

    def test_tool_call_redundancy_penalty(self):
        history = [{"tool": "lexicon_lookup", "input": "term", "output": {}}]
        reward, feedback = self.grader.grade_tool_call(
            tool_name="lexicon_lookup",
            tool_input="term",
            tool_output={"found": True},
            episode=self.episode,
            history=history,
        )
        self.assertIn("Redundant", feedback)


if __name__ == "__main__":
    unittest.main()
