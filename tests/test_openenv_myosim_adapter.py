from __future__ import annotations

import unittest

import numpy as np

from egocentric_dataset_test.competition.environment import EgocentricFactoryAction, EgocentricFactoryCompetitionEnv
from egocentric_dataset_test.competition.mujoco_sim import create_mujoco_task
from egocentric_dataset_test.competition.openenv_myosim_adapter import OpenEnvToMyoSimAdapter


class OpenEnvMyoSimAdapterTests(unittest.TestCase):
    def test_adapter_builds_target_and_action_for_easy_task(self) -> None:
        env = EgocentricFactoryCompetitionEnv(task_id="easy_bin_pick", seed=7)
        try:
            observation = env.reset(task_id="easy_bin_pick", seed=7)
            observation = env.step(EgocentricFactoryAction(joint_targets=observation.action_hint))
            state = env.state
        finally:
            env.close()

        adapter = OpenEnvToMyoSimAdapter.from_openenv(observation, state, [])
        task = create_mujoco_task("easy_finger_reach", seed=7)
        try:
            adapter.configure_task(task)
            obs = task.reset(seed=7)
            action = adapter.action(obs, task.sim.n_actuators)
        finally:
            task.close()

        self.assertEqual(action.shape, (5,))
        self.assertTrue(np.all(action >= 0.0))
        self.assertTrue(np.all(action <= 1.0))
        self.assertEqual(len(adapter.target_position), 3)
        self.assertGreaterEqual(adapter.grip_prior, 0.0)
        self.assertLessEqual(adapter.grip_prior, 1.0)

    def test_adapter_retargets_hand_task_and_exposes_conditioning(self) -> None:
        env = EgocentricFactoryCompetitionEnv(task_id="hard_precision_assembly", seed=7)
        try:
            observation = env.reset(task_id="hard_precision_assembly", seed=7)
            state = env.state
        finally:
            env.close()

        adapter = OpenEnvToMyoSimAdapter.from_openenv(observation, state, [])
        task = create_mujoco_task("hard_precision_assembly", seed=7)
        try:
            adapter.configure_task(task)
            obs = task.reset(seed=7)
            adapted_obs = adapter.adapt_myosim_observation(obs)
            actuator_count = task.sim.n_actuators
            action = adapter.action(obs, actuator_count)
        finally:
            task.close()

        self.assertEqual(len(adapted_obs["target_position"]), 3)
        self.assertGreater(len(adapted_obs["policy_observation"]), 10)
        self.assertEqual(action.shape, (actuator_count,))
        self.assertTrue(np.all(action >= 0.0))
        self.assertTrue(np.all(action <= 1.0))


if __name__ == "__main__":
    unittest.main()
