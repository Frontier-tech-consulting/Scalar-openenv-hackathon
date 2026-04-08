from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
from pydantic import Field

from egocentric_dataset_test.competition.tasks import (
    CompetitionTaskSpec,
    GradeBreakdown,
    grade_task_run,
    list_task_specs,
    resolve_task_spec,
    stage_hint,
)
from egocentric_dataset_test.competition.surrogate_backend import (
    BACKEND_MODE_SURROGATE_MYO,
    SurrogateMyoBackend,
    SurrogateMyoEpisode,
)

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
except Exception:  # pragma: no cover
    Environment = object  # type: ignore[misc,assignment]

    class Action:  # type: ignore[no-redef]
        pass

    class Observation:  # type: ignore[no-redef]
        pass

    class State:  # type: ignore[no-redef]
        pass

    @dataclass
    class EnvironmentMetadata:  # type: ignore[no-redef]
        name: str
        description: str
        version: str
        author: str


ACTION_DIMENSION = 4
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
STATE_DIMENSION = 8
DEFAULT_BACKEND_MODE = BACKEND_MODE_SURROGATE_MYO


class EgocentricFactoryAction(Action):
    joint_targets: list[float] = Field(
        default_factory=lambda: [0.0] * ACTION_DIMENSION,
        description=(
            "Normalized continuous control vector [reach_x, reach_y, grip_force, wrist_roll] "
            "bounded to [-1, 1]."
        ),
    )


class EgocentricFactoryObservation(Observation):
    step_idx: int = Field(default=0, description="Current step index within the task episode.")
    task_id: str = Field(default="easy_bin_pick", description="Deterministic task identifier.")
    task_title: str = Field(default="", description="Human-readable title for the active task.")
    difficulty: str = Field(default="easy", description="Task difficulty bucket.")
    task_prompt: str = Field(default="", description="Natural-language objective for the current task.")
    current_stage: str = Field(default="", description="Name of the active subgoal stage.")
    stage_index: int = Field(default=0, description="Zero-based active stage index.")
    remaining_stages: int = Field(default=0, description="Number of unfinished stages.")
    progress: float = Field(default=0.0, description="Fraction of stages completed in [0, 1].")
    backend_mode: str = Field(default=DEFAULT_BACKEND_MODE, description="Active competition runtime backend.")
    shard_id: str = Field(default="", description="Identifier of the active surrogate shard.")
    shard_progress: float = Field(default=0.0, description="Normalized progress through the active shard.")
    state_values: list[float] = Field(default_factory=list, description="Compact numeric state summary.")
    action_hint: list[float] = Field(
        default_factory=lambda: [0.0] * ACTION_DIMENSION,
        description="Recommended normalized control vector for the current stage.",
    )
    stage_guidance: str = Field(default="", description="Human-readable instruction for the current stage.")
    last_action_error: str | None = Field(default=None, description="Most recent action validation error.")
    reward: float | None = Field(default=None, description="Most recent reward in [0, 1].")
    done: bool = Field(default=False, description="Whether the episode has terminated.")


class EgocentricFactoryState(State):
    episode_id: str = Field(default="", description="Unique episode identifier.")
    step_count: int = Field(default=0, description="Number of steps taken.")
    cumulative_reward: float = Field(default=0.0, description="Cumulative reward over the episode.")
    total_steps: int = Field(default=0, description="Maximum step budget for the task.")
    task_id: str = Field(default="easy_bin_pick", description="Current task identifier.")
    task_title: str = Field(default="", description="Human-readable title for the active task.")
    difficulty: str = Field(default="easy", description="Difficulty bucket for the active task.")
    backend_mode: str = Field(default=DEFAULT_BACKEND_MODE, description="Active competition runtime backend.")
    shard_id: str = Field(default="", description="Identifier of the active surrogate shard.")
    shard_progress: float = Field(default=0.0, description="Normalized progress through the active shard.")
    simulation_origin: str = Field(default="", description="Summary of the backend's real-world simulation lineage.")
    current_stage: str = Field(default="", description="Active stage name.")
    completed_stages: int = Field(default=0, description="Number of completed sub-stages.")
    total_stages: int = Field(default=0, description="Total number of sub-stages in the task.")
    progress: float = Field(default=0.0, description="Fraction of completed stages in [0, 1].")
    grader_score: float = Field(default=0.0, description="Current deterministic grader score in [0, 1].")
    success: bool = Field(default=False, description="Whether the grader considers the run successful.")
    grader_breakdown: dict[str, float | bool] = Field(
        default_factory=dict,
        description="Deterministic grader breakdown for stage completion, precision, efficiency, and smoothness.",
    )
    last_action_error: str | None = Field(default=None, description="Last action validation error.")


@dataclass(slots=True)
class RuntimeEpisode:
    spec: CompetitionTaskSpec
    episode_id: str
    max_steps: int
    tool_state: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIMENSION, dtype=np.float32))
    previous_action: np.ndarray | None = None
    action_history: list[np.ndarray] = field(default_factory=list)
    stage_distances: list[float] = field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0
    current_stage_index: int = 0
    grader: GradeBreakdown | None = None
    last_action_error: str | None = None

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


class EgocentricFactoryCompetitionEnv(
    Environment[EgocentricFactoryAction, EgocentricFactoryObservation, EgocentricFactoryState]
):
    def __init__(self, task_id: str | None = None, seed: int = 7, **kwargs: Any) -> None:
        backend_mode = kwargs.pop("backend_mode", None)
        manifest_path = kwargs.pop("surrogate_manifest_path", None)
        super().__init__(**kwargs)
        self._seed = seed
        self._episode_counter = 0
        self._episode: RuntimeEpisode | None = None
        self._requested_task = task_id
        self._rng = np.random.default_rng(seed)
        self._backend_mode = str(backend_mode or os.getenv("COMPETITION_BACKEND", DEFAULT_BACKEND_MODE))
        manifest_path = manifest_path or os.getenv("SURROGATE_MYO_MANIFEST")
        max_shard_seconds = int(os.getenv("SURROGATE_MYO_MAX_SHARD_SECONDS", "110"))
        self._surrogate_backend = SurrogateMyoBackend(
            manifest_path=str(manifest_path) if manifest_path else None,
            max_shard_seconds=max_shard_seconds,
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> EgocentricFactoryObservation:
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        requested_task = kwargs.get("task_id") or kwargs.get("task") or self._requested_task
        spec = resolve_task_spec(requested_task)
        self._episode_counter += 1
        resolved_episode_id = episode_id or f"{spec.task_id}-{self._episode_counter:04d}"
        self._episode = self._surrogate_backend.reset(spec, episode_id=resolved_episode_id, max_steps=spec.max_steps)
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: EgocentricFactoryAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> EgocentricFactoryObservation:
        episode = self._require_episode()
        reward = 0.0
        raw_targets = np.asarray(action.joint_targets, dtype=np.float32).reshape(-1)
        last_action_error: str | None = None

        if raw_targets.size != ACTION_DIMENSION:
            last_action_error = (
                f"expected {ACTION_DIMENSION} action values, received {int(raw_targets.size)}; "
                "action was resized"
            )
            raw_targets = np.resize(raw_targets, (ACTION_DIMENSION,))

        clipped_targets = np.clip(raw_targets, ACTION_LOW, ACTION_HIGH)
        if not np.allclose(raw_targets, clipped_targets):
            last_action_error = "action values were clipped to [-1, 1]"

        episode.last_action_error = last_action_error
        reward = self._surrogate_backend.step(
            episode,
            action_values=clipped_targets,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
        )
        done = episode.done
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> EgocentricFactoryState:
        episode = self._require_episode()
        grader = episode.grader or grade_task_run(
            episode.spec,
            completed_stages=episode.current_stage_index,
            stage_distances=episode.stage_distances,
            actions=episode.action_history,
            steps_taken=episode.step_count,
        )
        return EgocentricFactoryState(
            episode_id=episode.episode_id,
            step_count=episode.step_count,
            cumulative_reward=float(episode.cumulative_reward),
            total_steps=episode.max_steps,
            task_id=episode.spec.task_id,
            task_title=episode.spec.title,
            difficulty=episode.spec.difficulty,
            backend_mode=self._backend_mode,
            shard_id=episode.active_shard.shard_id,
            shard_progress=float(episode.shard_progress),
            simulation_origin=episode.manifest.simulation_origin,
            current_stage=episode.active_stage_name,
            completed_stages=min(episode.current_stage_index, len(episode.spec.stages)),
            total_stages=len(episode.spec.stages),
            progress=float(episode.progress),
            grader_score=float(grader.score),
            success=bool(grader.success),
            grader_breakdown=grader.to_dict(),
            last_action_error=episode.last_action_error,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="egocentric_factory_competition_env",
            description=(
                "Egocentric manipulation environment for real-world factory tasks: component pick, "
                "sorting, and precision assembly with deterministic graders and a shard-based "
                "surrogate_myo backend derived from the earlier MyoSuite/MuJoCo pipeline."
            ),
            version="0.2.0",
            author="openenv-course",
        )

    def close(self) -> None:
        self._episode = None

    def available_tasks(self) -> list[dict[str, object]]:
        return [spec.to_dict() for spec in list_task_specs()]

    def _require_episode(self) -> RuntimeEpisode:
        if self._episode is None:
            raise RuntimeError("Call reset() before step() or state().")
        return self._episode

    def _build_observation(self, *, reward: float, done: bool) -> EgocentricFactoryObservation:
        episode = self._require_episode()
        stage_name = episode.active_stage_name
        active_target = episode.active_target
        stage_text = stage_hint(episode.spec, episode.current_stage_index)
        active_shard = episode.active_shard
        grader = episode.grader or grade_task_run(
            episode.spec,
            completed_stages=episode.current_stage_index,
            stage_distances=episode.stage_distances,
            actions=episode.action_history,
            steps_taken=episode.step_count,
        )
        state_values = [
            float(value) for value in np.concatenate(
                [
                    episode.tool_state,
                    active_target - episode.tool_state,
                ]
            )
        ]
        return EgocentricFactoryObservation(
            step_idx=episode.step_count,
            task_id=episode.spec.task_id,
            task_title=episode.spec.title,
            difficulty=episode.spec.difficulty,
            task_prompt=episode.spec.objective,
            current_stage=stage_name,
            stage_index=min(episode.current_stage_index, len(episode.spec.stages) - 1),
            remaining_stages=max(len(episode.spec.stages) - episode.current_stage_index, 0),
            progress=float(grader.stage_completion),
            backend_mode=self._backend_mode,
            shard_id=active_shard.shard_id,
            shard_progress=float(episode.shard_progress),
            state_values=state_values,
            action_hint=[float(value) for value in active_target.tolist()],
            stage_guidance=stage_text,
            last_action_error=episode.last_action_error,
            reward=float(reward),
            done=done,
            metadata={
                "task_title": episode.spec.title,
                "backend_mode": self._backend_mode,
                "grader": grader.to_dict(),
                "active_stage_target": [float(value) for value in active_target.tolist()],
                "active_shard": active_shard.to_dict(),
                "simulation_origin": episode.manifest.simulation_origin,
                "contact_confidence": round(float(episode.contact_confidence), 4),
                "insertion_depth": round(float(episode.insertion_depth), 4),
                "completed_stages": min(episode.current_stage_index, len(episode.spec.stages)),
                "total_stages": len(episode.spec.stages),
            },
        )
