"""Unit tests for Task 6 — Full Manuscript Session."""

import unittest
from server.environment import SanskritEnvironment
from models import ManuscriptAction


class TestFullManuscriptSession(unittest.TestCase):
    def setUp(self):
        self.env = SanskritEnvironment()

    def test_reset_returns_first_phase(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        self.assertEqual(obs.task_id, "full_manuscript_session")
        self.assertFalse(obs.done)
        self.assertIn("Phase 1/", obs.decision_prompt)

    def test_mcq_correct_advances_phase(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        # Find correct answer for phase 1
        session = self.env._sessions[obs.episode_id]
        ep = session["current_episode"]
        phase1 = ep["phases"][0]
        correct = phase1["correct"]

        obs2 = self.env.step(ManuscriptAction(selected_option=correct))
        self.assertFalse(obs2.done)  # More phases remain
        self.assertIn("Phase 2/", obs2.decision_prompt)

    def test_all_correct_completes_session(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        session = self.env._sessions[obs.episode_id]
        ep = session["current_episode"]
        phases = ep["phases"]

        for i, phase in enumerate(phases):
            if phase["phase"] == "restoration":
                # Commit directly
                obs = self.env.step(ManuscriptAction(
                    action_type="commit",
                    final_answer=phase["correct"],
                ))
            else:
                obs = self.env.step(ManuscriptAction(
                    selected_option=phase["correct"],
                ))

        self.assertTrue(obs.done)
        self.assertGreater(obs.reward, 0.0)

    def test_wrong_answer_returns_lower_score(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        session = self.env._sessions[obs.episode_id]
        ep = session["current_episode"]
        phases = ep["phases"]

        for i, phase in enumerate(phases):
            if phase["phase"] == "restoration":
                obs = self.env.step(ManuscriptAction(
                    action_type="commit", final_answer="wrong answer",
                ))
            else:
                # Wrong answer on first, correct on rest
                answer = "wrong" if i == 0 else phase["correct"]
                obs = self.env.step(ManuscriptAction(selected_option=answer))

        self.assertTrue(obs.done)
        # Score should be lower than all-correct
        self.assertLess(obs.reward, 0.95)

    def test_restoration_tool_call_in_session(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        session = self.env._sessions[obs.episode_id]
        ep = session["current_episode"]
        phases = ep["phases"]

        # Answer all MCQ phases correctly
        for phase in phases:
            if phase["phase"] == "restoration":
                break
            obs = self.env.step(ManuscriptAction(selected_option=phase["correct"]))

        # Now in restoration phase — use a tool
        self.assertFalse(obs.done)
        obs2 = self.env.step(ManuscriptAction(
            action_type="tool_call",
            tool_name="lexicon_lookup",
            tool_input="test",
        ))
        self.assertFalse(obs2.done)
        self.assertIsNotNone(obs2.tool_call_history)
        self.assertEqual(len(obs2.tool_call_history), 1)

        # Now commit
        rest_phase = next(p for p in phases if p["phase"] == "restoration")
        obs3 = self.env.step(ManuscriptAction(
            action_type="commit",
            final_answer=rest_phase["correct"],
        ))
        self.assertTrue(obs3.done)
        self.assertGreater(obs3.reward, 0.0)

    def test_consistency_violation_penalizes_score(self):
        """When agent contradicts itself across phases, score should be penalized."""
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        session = self.env._sessions[obs.episode_id]
        # Force a consistency violation by manipulating state
        session["t6_consistency_violations"] = 3
        session["t6_phase_rewards"] = [0.95, 0.95, 0.95]
        session["t6_phase_answers"] = []

        done_obs = self.env._build_t6_done_observation(
            session["current_episode"], session
        )
        # Should be penalized: 0.95 - 0.15 + 0.05 bonus not applied
        self.assertLess(done_obs.reward, 0.95)

    def test_session_count_matches_episode_data(self):
        obs = self.env.reset(task_id="full_manuscript_session", seed=42)
        session = self.env._sessions[obs.episode_id]
        ep = session["current_episode"]
        self.assertEqual(len(ep["phases"]), 5)
        self.assertTrue(any(p["phase"] == "restoration" for p in ep["phases"]))


if __name__ == "__main__":
    unittest.main()
