from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from egocentric_dataset_test.competition import EgocentricFactoryAction, EgocentricFactoryCompetitionEnv, list_task_specs
from egocentric_dataset_test.competition.server import create_app


class CompetitionSubmissionTests(unittest.TestCase):
    def test_env_exposes_three_tasks(self) -> None:
        tasks = list_task_specs()
        self.assertGreaterEqual(len(tasks), 3)
        self.assertEqual(tasks[0].task_id, "easy_bin_pick")

    def test_all_tasks_grade_in_normalized_range(self) -> None:
        for spec in list_task_specs():
            env = EgocentricFactoryCompetitionEnv(task_id=spec.task_id, seed=7)
            observation = env.reset(task_id=spec.task_id, seed=7)
            while not observation.done:
                observation = env.step(EgocentricFactoryAction(joint_targets=observation.action_hint))
            self.assertGreaterEqual(env.state.grader_score, 0.0)
            self.assertLessEqual(env.state.grader_score, 1.0)
            self.assertGreaterEqual(env.state.cumulative_reward, 0.0)
            env.close()

    def test_env_reset_and_step(self) -> None:
        env = EgocentricFactoryCompetitionEnv(task_id="easy_bin_pick", seed=7)
        observation = env.reset(task_id="easy_bin_pick", seed=7)
        self.assertEqual(observation.task_id, "easy_bin_pick")
        self.assertEqual(observation.backend_mode, "surrogate_myo")
        self.assertTrue(observation.shard_id)
        self.assertFalse(observation.done)

        result = env.step(EgocentricFactoryAction(joint_targets=observation.action_hint))
        self.assertGreaterEqual(float(result.reward or 0.0), 0.0)
        self.assertLessEqual(float(result.reward or 0.0), 1.0)
        self.assertGreaterEqual(env.state.grader_score, 0.0)
        self.assertLessEqual(env.state.grader_score, 1.0)
        self.assertEqual(env.state.backend_mode, "surrogate_myo")
        self.assertTrue(env.state.shard_id)
        env.close()

    def test_server_reset_works_without_payload(self) -> None:
        client = TestClient(create_app())
        response = client.post("/reset", json={})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("observation", payload)
        self.assertEqual(payload["observation"]["task_id"], "easy_bin_pick")
        self.assertEqual(payload["observation"]["backend_mode"], "surrogate_myo")
        self.assertIn("reward", payload)
        self.assertIn("done", payload)

        step_response = client.post(
            "/step",
            json={"action": {"joint_targets": payload["observation"]["action_hint"]}},
        )
        self.assertEqual(step_response.status_code, 200)
        step_payload = step_response.json()
        self.assertIn("observation", step_payload)
        self.assertGreaterEqual(float(step_payload["reward"]), 0.0)
        self.assertLessEqual(float(step_payload["reward"]), 1.0)

        state_response = client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        state_payload = state_response.json()
        self.assertEqual(state_payload["task_id"], "easy_bin_pick")
        self.assertEqual(state_payload["backend_mode"], "surrogate_myo")
        self.assertIn("grader_breakdown", state_payload)


if __name__ == "__main__":
    unittest.main()
