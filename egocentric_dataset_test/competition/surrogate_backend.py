from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egocentric_dataset_test.competition.shards import (
    DEFAULT_MAX_SHARD_SECONDS,
    SurrogateMyoManifest,
    SurrogateMyoShard,
    load_surrogate_manifests,
)
from egocentric_dataset_test.competition.tasks import CompetitionTaskSpec, GradeBreakdown, grade_task_run


BACKEND_MODE_SURROGATE_MYO = "surrogate_myo"


@dataclass(slots=True)
class SurrogateMyoEpisode:
    spec: CompetitionTaskSpec
    manifest: SurrogateMyoManifest
    episode_id: str
    max_steps: int
    tool_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    previous_action: np.ndarray | None = None
    action_history: list[np.ndarray] = field(default_factory=list)
    stage_distances: list[float] = field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0
    current_stage_index: int = 0
    grader: GradeBreakdown | None = None
    last_action_error: str | None = None
    contact_confidence: float = 0.0
    insertion_depth: float = 0.0

    @property
    def done(self) -> bool:
        return self.current_stage_index >= len(self.spec.stages) or self.step_count >= self.max_steps

    @property
    def progress(self) -> float:
        return min(self.current_stage_index / max(len(self.spec.stages), 1), 1.0)

    @property
    def active_stage_name(self) -> str:
        if self.current_stage_index >= len(self.spec.stages):
            return "completed"
        return self.spec.stages[self.current_stage_index].name

    @property
    def active_target(self) -> np.ndarray:
        if self.current_stage_index >= len(self.spec.stages):
            return np.asarray(self.spec.stages[-1].target, dtype=np.float32)
        return np.asarray(self.spec.stages[self.current_stage_index].target, dtype=np.float32)

    @property
    def active_shard(self) -> SurrogateMyoShard:
        return self.manifest.shard_for_stage(self.current_stage_index)

    @property
    def shard_progress(self) -> float:
        shard = self.active_shard
        span = max(shard.stage_end - shard.stage_start + 1, 1)
        stage_offset = min(max(self.current_stage_index - shard.stage_start, 0), span)
        return min(stage_offset / span, 1.0)


class SurrogateMyoBackend:
    def __init__(self, manifest_path: str | None = None, max_shard_seconds: int = DEFAULT_MAX_SHARD_SECONDS) -> None:
        self._manifests = load_surrogate_manifests(manifest_path, max_shard_seconds=max_shard_seconds)

    def reset(self, spec: CompetitionTaskSpec, *, episode_id: str, max_steps: int) -> SurrogateMyoEpisode:
        episode = SurrogateMyoEpisode(
            spec=spec,
            manifest=self._manifests[spec.task_id],
            episode_id=episode_id,
            max_steps=max_steps,
        )
        episode.grader = grade_task_run(
            spec,
            completed_stages=0,
            stage_distances=[],
            actions=[],
            steps_taken=0,
        )
        return episode

    def step(
        self,
        episode: SurrogateMyoEpisode,
        *,
        action_values: np.ndarray,
        action_low: float,
        action_high: float,
    ) -> float:
        stage = episode.spec.stages[min(episode.current_stage_index, len(episode.spec.stages) - 1)]
        active_target = episode.active_target
        shard = episode.active_shard

        previous_distance = float(np.linalg.norm(active_target - episode.tool_state))
        biased_target = np.clip(active_target + np.asarray(shard.pose_bias, dtype=np.float32), action_low, action_high)
        desired_state = np.clip(0.84 * action_values + 0.16 * biased_target, action_low, action_high)
        updated_state = episode.tool_state + shard.response_gain * (desired_state - episode.tool_state)
        current_distance = float(np.linalg.norm(active_target - updated_state))

        progress_gain = max(previous_distance - current_distance, 0.0)
        regress_penalty = max(current_distance - previous_distance, 0.0)
        oscillation_penalty = 0.0
        if episode.previous_action is not None:
            oscillation_penalty = max(float(np.linalg.norm(action_values - episode.previous_action)) - 1.2, 0.0) * 0.08

        proximity_signal = max(0.0, 1.0 - current_distance / max(stage.tolerance * 3.0, 1e-6))
        episode.contact_confidence = float(
            np.clip(0.55 * episode.contact_confidence + 0.45 * proximity_signal * shard.contact_scale, 0.0, 1.0)
        )

        stage_name = stage.name.lower()
        if "insert" in stage_name or "seat" in stage_name or "align" in stage_name:
            episode.insertion_depth = float(
                np.clip(max(episode.insertion_depth, episode.contact_confidence * 0.92), 0.0, 1.0)
            )

        reward = 0.0
        reward += progress_gain * (0.72 + stage.weight + 0.12 * shard.contact_scale)
        reward += 0.05 * episode.contact_confidence
        reward -= regress_penalty * 0.25
        reward -= oscillation_penalty

        completion_tolerance = stage.tolerance * (1.0 + 0.04 * shard.contact_scale)
        stage_completed = current_distance <= completion_tolerance
        if stage_completed:
            reward += 0.24 + stage.weight + 0.05 * episode.contact_confidence
            episode.current_stage_index += 1

        episode.tool_state = updated_state.astype(np.float32)
        episode.previous_action = action_values.astype(np.float32)
        episode.action_history.append(action_values.astype(np.float32))
        episode.stage_distances.append(current_distance)
        episode.step_count += 1

        episode.grader = grade_task_run(
            episode.spec,
            completed_stages=episode.current_stage_index,
            stage_distances=episode.stage_distances,
            actions=episode.action_history,
            steps_taken=episode.step_count,
        )

        if episode.current_stage_index >= len(episode.spec.stages):
            reward += 0.10 + 0.20 * episode.grader.precision + 0.05 * episode.insertion_depth
        elif episode.step_count >= episode.max_steps:
            reward -= 0.08

        reward = float(np.clip(reward, 0.0, 1.0))
        episode.cumulative_reward += reward
        return reward