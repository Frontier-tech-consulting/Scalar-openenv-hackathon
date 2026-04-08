from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class TaskStage:
    name: str
    target: tuple[float, float, float, float]
    tolerance: float
    weight: float
    guidance: str


@dataclass(frozen=True, slots=True)
class CompetitionTaskSpec:
    task_id: str
    title: str
    difficulty: str
    objective: str
    grader_description: str
    max_steps: int
    stages: tuple[TaskStage, ...]

    @property
    def ideal_steps(self) -> int:
        return len(self.stages) * 2

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "difficulty": self.difficulty,
            "objective": self.objective,
            "grader_description": self.grader_description,
            "max_steps": self.max_steps,
            "stages": [
                {
                    "name": stage.name,
                    "target": list(stage.target),
                    "tolerance": stage.tolerance,
                    "weight": stage.weight,
                    "guidance": stage.guidance,
                }
                for stage in self.stages
            ],
        }


COMPETITION_TASKS: dict[str, CompetitionTaskSpec] = {
    "easy_bin_pick": CompetitionTaskSpec(
        task_id="easy_bin_pick",
        title="Pick PCB component from bin",
        difficulty="easy",
        objective=(
            "Reach into the left component bin, close the gripper around a PCB component, "
            "and lift it to the inspection pose without excessive motion."
        ),
        grader_description=(
            "Scores stage completion, closeness to each subgoal, action smoothness, and step efficiency."
        ),
        max_steps=8,
        stages=(
            TaskStage(
                name="approach_bin",
                target=(0.78, -0.52, 0.10, 0.00),
                tolerance=0.16,
                weight=0.25,
                guidance="Move the wrist over the left bin with the gripper open.",
            ),
            TaskStage(
                name="secure_component",
                target=(0.28, -0.10, 0.86, 0.04),
                tolerance=0.14,
                weight=0.35,
                guidance="Close the gripper while centering on the component.",
            ),
            TaskStage(
                name="lift_for_inspection",
                target=(0.58, 0.34, 0.94, 0.10),
                tolerance=0.16,
                weight=0.40,
                guidance="Lift upward and stabilize the component at inspection height.",
            ),
        ),
    ),
    "medium_sort_and_place": CompetitionTaskSpec(
        task_id="medium_sort_and_place",
        title="Sort and place component",
        difficulty="medium",
        objective=(
            "Pick a component, rotate it toward the sorting tray, translate to the placement area, "
            "and release it cleanly into the tray."
        ),
        grader_description=(
            "Scores completion of reach, grasp, transfer, orientation, and release stages with penalties for oscillation."
        ),
        max_steps=12,
        stages=(
            TaskStage(
                name="reach_component",
                target=(0.76, -0.48, 0.12, -0.10),
                tolerance=0.17,
                weight=0.18,
                guidance="Approach the loose component with an open gripper.",
            ),
            TaskStage(
                name="grasp_component",
                target=(0.34, -0.06, 0.88, -0.02),
                tolerance=0.14,
                weight=0.22,
                guidance="Close the gripper and center the part before moving.",
            ),
            TaskStage(
                name="rotate_for_sorting",
                target=(0.40, 0.02, 0.92, 0.48),
                tolerance=0.15,
                weight=0.18,
                guidance="Rotate the wrist to align the part with the tray orientation.",
            ),
            TaskStage(
                name="move_to_tray",
                target=(-0.18, 0.42, 0.92, 0.42),
                tolerance=0.16,
                weight=0.20,
                guidance="Translate the part to the sorting tray while keeping grip stable.",
            ),
            TaskStage(
                name="release_into_tray",
                target=(-0.26, 0.38, 0.14, 0.36),
                tolerance=0.17,
                weight=0.22,
                guidance="Release gently into the tray without a sudden wrist change.",
            ),
        ),
    ),
    "hard_precision_assembly": CompetitionTaskSpec(
        task_id="hard_precision_assembly",
        title="Precision assembly insertion",
        difficulty="hard",
        objective=(
            "Pick a connector, align it with the assembly slot, execute a careful insertion motion, "
            "and finish with a stable seated pose."
        ),
        grader_description=(
            "Scores all stages, with strong weighting on alignment and insertion precision plus efficiency and smoothness."
        ),
        max_steps=14,
        stages=(
            TaskStage(
                name="reach_connector",
                target=(0.72, -0.50, 0.08, -0.12),
                tolerance=0.16,
                weight=0.14,
                guidance="Move over the connector pickup area with the gripper open.",
            ),
            TaskStage(
                name="pinch_connector",
                target=(0.26, -0.12, 0.90, -0.08),
                tolerance=0.13,
                weight=0.18,
                guidance="Pinch the connector securely without twisting it yet.",
            ),
            TaskStage(
                name="pre_align",
                target=(0.18, 0.02, 0.94, 0.32),
                tolerance=0.14,
                weight=0.16,
                guidance="Start wrist rotation and move toward the slot entrance.",
            ),
            TaskStage(
                name="fine_align",
                target=(0.04, 0.26, 0.96, 0.74),
                tolerance=0.11,
                weight=0.22,
                guidance="Perform the precision alignment directly in front of the slot.",
            ),
            TaskStage(
                name="insert_connector",
                target=(-0.08, 0.54, 0.98, 0.80),
                tolerance=0.10,
                weight=0.20,
                guidance="Advance forward with minimal wrist disturbance to insert the connector.",
            ),
            TaskStage(
                name="seat_and_hold",
                target=(-0.10, 0.58, 0.72, 0.82),
                tolerance=0.12,
                weight=0.10,
                guidance="Hold a stable seated pose and slightly relax grip to finish the assembly.",
            ),
        ),
    ),
}


DEFAULT_TASK_ID = "easy_bin_pick"


def list_task_specs() -> list[CompetitionTaskSpec]:
    return list(COMPETITION_TASKS.values())


def get_task_spec(task_id: str) -> CompetitionTaskSpec:
    normalized = (task_id or DEFAULT_TASK_ID).strip().lower()
    if normalized not in COMPETITION_TASKS:
        available = ", ".join(sorted(COMPETITION_TASKS))
        raise KeyError(f"Unknown task_id {task_id!r}. Available tasks: {available}")
    return COMPETITION_TASKS[normalized]


def resolve_task_spec(task_name_or_id: str | None) -> CompetitionTaskSpec:
    if not task_name_or_id:
        return get_task_spec(DEFAULT_TASK_ID)

    normalized = task_name_or_id.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in COMPETITION_TASKS:
        return COMPETITION_TASKS[normalized]

    if "assembly" in normalized or "insert" in normalized or "precision" in normalized:
        return COMPETITION_TASKS["hard_precision_assembly"]
    if "sort" in normalized or "tray" in normalized or "place" in normalized:
        return COMPETITION_TASKS["medium_sort_and_place"]
    return COMPETITION_TASKS[DEFAULT_TASK_ID]


def stage_hint(spec: CompetitionTaskSpec, stage_index: int) -> str:
    if stage_index >= len(spec.stages):
        return "All stages completed. Hold a stable finishing pose."
    stage = spec.stages[stage_index]
    target_str = ", ".join(f"{value:.2f}" for value in stage.target)
    return f"{stage.guidance} Target action: [{target_str}]"


@dataclass(slots=True)
class GradeBreakdown:
    score: float
    success: bool
    stage_completion: float
    precision: float
    efficiency: float
    smoothness: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "score": round(self.score, 4),
            "success": self.success,
            "stage_completion": round(self.stage_completion, 4),
            "precision": round(self.precision, 4),
            "efficiency": round(self.efficiency, 4),
            "smoothness": round(self.smoothness, 4),
        }


def grade_task_run(
    spec: CompetitionTaskSpec,
    *,
    completed_stages: int,
    stage_distances: Iterable[float],
    actions: Iterable[np.ndarray],
    steps_taken: int,
) -> GradeBreakdown:
    stage_completion = completed_stages / max(len(spec.stages), 1)

    distance_array = np.asarray(list(stage_distances), dtype=np.float32)
    if distance_array.size == 0:
        precision = 0.0
    else:
        normalized = np.clip(distance_array / 1.75, 0.0, 1.0)
        precision = float(1.0 - normalized.mean())

    steps_over_ideal = max(steps_taken - spec.ideal_steps, 0)
    denominator = max(spec.max_steps - spec.ideal_steps, 1)
    efficiency = float(1.0 - min(steps_over_ideal / denominator, 1.0))

    action_list = [np.asarray(action, dtype=np.float32) for action in actions]
    if len(action_list) < 2:
        smoothness = 1.0
    else:
        deltas = [float(np.linalg.norm(current - previous)) for previous, current in zip(action_list[:-1], action_list[1:])]
        smoothness = float(1.0 - min(np.mean(deltas) / 2.5, 1.0))

    score = (
        0.55 * stage_completion
        + 0.20 * precision
        + 0.15 * efficiency
        + 0.10 * smoothness
    )
    score = float(np.clip(score, 0.0, 1.0))
    success = completed_stages == len(spec.stages) and score >= 0.70
    return GradeBreakdown(
        score=score,
        success=success,
        stage_completion=stage_completion,
        precision=precision,
        efficiency=efficiency,
        smoothness=smoothness,
    )
