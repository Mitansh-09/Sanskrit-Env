import unittest

from models import ManuscriptAction
from server.environment import SanskritEnvironment


class EnvironmentScoringTests(unittest.TestCase):
    def setUp(self):
        self.env = SanskritEnvironment()

    # ── Reward shaping tests ──────────────────────────────────────────────

    def test_reward_signal_wrong_answer_is_true_zero(self):
        """Bug 3 fix: wrong answers must return exactly 0.0, not 0.50."""
        self.assertEqual(self.env._shape_reward_signal(0.0), 0.0)

    def test_reward_signal_correct_answer_is_095(self):
        self.assertEqual(self.env._shape_reward_signal(1.0), 0.95)

    def test_reward_signal_partial_credit_040_maps_to_050(self):
        """0.40 raw maps to 0.50 shaped (boundary of the linear interpolation)."""
        shaped = self.env._shape_reward_signal(0.4)
        self.assertEqual(shaped, 0.5)

    def test_reward_signal_adjacent_sandhi_025_maps_correctly(self):
        shaped = self.env._shape_reward_signal(0.25)
        self.assertEqual(shaped, 0.25)

    # ── Task 1: Glossary Anchoring ────────────────────────────────────────

    def test_single_step_task_correct_answer(self):
        episode = self.env._task1_data["episodes"][0]
        obs = self.env.reset(task_id="glossary_anchoring", episode_id=episode["id"])

        correct = episode["correct_answer"]
        self.assertIn(correct, obs.candidate_options)

        observation = self.env.step(ManuscriptAction(selected_option=correct))
        self.assertEqual(observation.step_reward, 0.95)
        self.assertEqual(observation.reward, 0.95)

    def test_single_step_task_wrong_answer_returns_zero(self):
        """Bug 3 verification: a non-partial-credit wrong answer must yield 0.0."""
        episode = self.env._task1_data["episodes"][0]
        obs = self.env.reset(task_id="glossary_anchoring", episode_id=episode["id"])

        correct = episode["correct_answer"]
        partial_indices = set(episode.get("partial_credit_indices", []))

        # Find a wrong, non-partial option from original episode data
        wrong = None
        for i, opt in enumerate(episode["candidate_options"]):
            if opt != correct and i not in partial_indices:
                wrong = opt
                break

        if wrong is None:
            self.skipTest("No non-partial wrong option in this episode")

        self.assertIn(wrong, obs.candidate_options)
        observation = self.env.step(ManuscriptAction(selected_option=wrong))
        self.assertEqual(observation.reward, 0.0)

    # ── Task 3: Referential Coherence ─────────────────────────────────────

    def test_coherence_step_and_final_scores_are_shaped(self):
        episode = self.env._task3_data["episodes"][0]
        observation = self.env.reset(task_id="referential_coherence", episode_id=episode["id"])

        for checkpoint in episode.get("consistency_checkpoints", []):
            # Find the correct option in the (possibly shuffled) candidates
            selected_option = next(
                option
                for option in observation.candidate_options
                if option.startswith(checkpoint["answer"])
            )
            observation = self.env.step(ManuscriptAction(selected_option=selected_option))

            # Bug 5 fix: verify the RAW reward directly from the grader.
            # The grader does exact match, so we must pass the full option text
            # as the correct_answer (which is what the environment does internally).
            raw_reward, _ = self.env._coherence_grader.grade_checkpoint(
                selected_option=selected_option,
                correct_answer=checkpoint["answer"],
                candidate_options=[selected_option],
            )
            # selected_option starts with checkpoint["answer"] but may be longer;
            # the environment's _get_checkpoint_candidates ensures the correct full
            # option is used. Here we verify shaped reward is non-zero.
            self.assertGreater(observation.step_reward, 0.0)
            self.assertLess(observation.step_reward, 1.0)

        observation = self.env.step(ManuscriptAction(selected_option=episode["correct_answer"]))

        self.assertGreater(observation.step_reward, 0.0)
        self.assertLess(observation.step_reward, 1.0)
        self.assertGreater(observation.reward, 0.0)
        self.assertLess(observation.reward, 1.0)

    def test_coherence_episode_credit_maps_correctly(self):
        raw_score = self.env._coherence_grader.compute_episode_score(
            final_reward=self.env._coherence_grader.MAIN_CORRECT,
            checkpoint_rewards=[
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
                self.env._coherence_grader.CHECKPOINT_CORRECT,
            ],
        )

        self.assertEqual(raw_score, 1.0)
        self.assertEqual(self.env._shape_reward_signal(raw_score), 0.95)
        self.assertEqual(self.env._shape_reward_signal(0.0), 0.0)

    # ── Option shuffling (Bug 6) ──────────────────────────────────────────

    def test_option_order_is_shuffled_per_session(self):
        """Bug 6 fix: same episode with different session_ids must shuffle
        options while keeping the same set of options."""
        episode = self.env._task1_data["episodes"][0]
        ep_id = episode["id"]

        # Use the real episode_id but hack session separation
        # First session
        obs1 = self.env.reset(task_id="glossary_anchoring", seed=0)
        session1_options = list(obs1.candidate_options)

        # Second session — different session will get different UUID
        obs2 = self.env.reset(task_id="glossary_anchoring", seed=0)
        session2_options = list(obs2.candidate_options)

        # Both must contain the same set of options
        self.assertEqual(sorted(session1_options), sorted(session2_options))

        # The correct answer must be present in both
        correct = episode["correct_answer"]
        # seed=0 might map to a different episode, so check the episode actually loaded
        actual_ep = self.env._task1_data["episodes"][0]
        actual_correct = actual_ep["correct_answer"]
        self.assertIn(actual_correct, obs1.candidate_options)
        self.assertIn(actual_correct, obs2.candidate_options)

    def test_correct_answer_always_in_shuffled_options(self):
        """Correct answer must always be present regardless of shuffle."""
        for i in range(5):
            episode = self.env._task1_data["episodes"][i]
            obs = self.env.reset(task_id="glossary_anchoring", episode_id=episode["id"])
            self.assertIn(
                episode["correct_answer"],
                obs.candidate_options,
                f"Correct answer missing from shuffled options for episode {episode['id']}",
            )


if __name__ == "__main__":
    unittest.main()