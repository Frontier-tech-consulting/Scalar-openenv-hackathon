from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from egocentric_dataset_test.competition.tasks import CompetitionTaskSpec, list_task_specs


DEFAULT_MAX_SHARD_SECONDS = 110


@dataclass(frozen=True, slots=True)
class SurrogateMyoShard:
    shard_id: str
    label: str
    stage_start: int
    stage_end: int
    duration_seconds: int
    pose_bias: tuple[float, float, float, float]
    response_gain: float
    contact_scale: float

    def contains_stage(self, stage_index: int) -> bool:
        return self.stage_start <= stage_index <= self.stage_end

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "label": self.label,
            "stage_start": self.stage_start,
            "stage_end": self.stage_end,
            "duration_seconds": self.duration_seconds,
            "pose_bias": list(self.pose_bias),
            "response_gain": self.response_gain,
            "contact_scale": self.contact_scale,
        }


@dataclass(frozen=True, slots=True)
class SurrogateMyoManifest:
    task_id: str
    task_title: str
    max_shard_seconds: int
    simulation_origin: str
    shards: tuple[SurrogateMyoShard, ...]

    def shard_for_stage(self, stage_index: int) -> SurrogateMyoShard:
        normalized = max(stage_index, 0)
        for shard in self.shards:
            if shard.contains_stage(normalized):
                return shard
        return self.shards[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "max_shard_seconds": self.max_shard_seconds,
            "simulation_origin": self.simulation_origin,
            "shards": [shard.to_dict() for shard in self.shards],
        }


def load_surrogate_manifests(
    manifest_path: str | None = None,
    *,
    max_shard_seconds: int = DEFAULT_MAX_SHARD_SECONDS,
) -> dict[str, SurrogateMyoManifest]:
    if manifest_path:
        candidate = Path(manifest_path)
        if candidate.exists():
            return _load_manifests_from_json(candidate, max_shard_seconds=max_shard_seconds)
    return {
        spec.task_id: build_default_manifest(spec, max_shard_seconds=max_shard_seconds)
        for spec in list_task_specs()
    }


def build_default_manifest(
    spec: CompetitionTaskSpec,
    *,
    max_shard_seconds: int = DEFAULT_MAX_SHARD_SECONDS,
) -> SurrogateMyoManifest:
    stage_count = len(spec.stages)
    if stage_count <= 3:
        stage_groups = [(0, stage_count - 1)]
    elif stage_count <= 5:
        stage_groups = [(0, 1), (2, stage_count - 1)]
    else:
        stage_groups = [(0, 1), (2, 3), (4, stage_count - 1)]

    profile_biases = [
        (0.04, -0.03, 0.02, -0.02),
        (0.00, 0.02, 0.04, 0.05),
        (-0.03, 0.05, -0.01, 0.03),
    ]
    response_gains = [0.74, 0.78, 0.72]
    contact_scales = [0.95, 1.00, 1.08]

    shards: list[SurrogateMyoShard] = []
    for index, (stage_start, stage_end) in enumerate(stage_groups):
        shard_length = stage_end - stage_start + 1
        base_duration = min(max_shard_seconds, 45 + shard_length * 20)
        shards.append(
            SurrogateMyoShard(
                shard_id=f"{spec.task_id}_shard_{index + 1:02d}",
                label=f"phase_{index + 1}",
                stage_start=stage_start,
                stage_end=stage_end,
                duration_seconds=base_duration,
                pose_bias=profile_biases[min(index, len(profile_biases) - 1)],
                response_gain=response_gains[min(index, len(response_gains) - 1)],
                contact_scale=contact_scales[min(index, len(contact_scales) - 1)],
            )
        )

    return SurrogateMyoManifest(
        task_id=spec.task_id,
        task_title=spec.title,
        max_shard_seconds=max_shard_seconds,
        simulation_origin=(
            "surrogate_myo: shard-based approximation of the earlier Egocentric-100K + "
            "MyoSuite/MuJoCo assembly pipeline, tuned for deterministic CPU-safe OpenEnv judging"
        ),
        shards=tuple(shards),
    )


def _load_manifests_from_json(path: Path, *, max_shard_seconds: int) -> dict[str, SurrogateMyoManifest]:
    payload = json.loads(path.read_text())
    task_payload = payload.get("tasks", payload)
    manifests: dict[str, SurrogateMyoManifest] = {}
    specs = {spec.task_id: spec for spec in list_task_specs()}

    for task_id, config in task_payload.items():
        spec = specs.get(task_id)
        if spec is None:
            continue

        shard_items = []
        for index, shard_config in enumerate(config.get("shards", [])):
            pose_bias = tuple(float(value) for value in shard_config.get("pose_bias", [0.0, 0.0, 0.0, 0.0]))
            if len(pose_bias) != 4:
                pose_bias = (0.0, 0.0, 0.0, 0.0)
            shard_items.append(
                SurrogateMyoShard(
                    shard_id=str(shard_config.get("shard_id", f"{task_id}_shard_{index + 1:02d}")),
                    label=str(shard_config.get("label", f"phase_{index + 1}")),
                    stage_start=int(shard_config.get("stage_start", 0)),
                    stage_end=int(shard_config.get("stage_end", max(len(spec.stages) - 1, 0))),
                    duration_seconds=int(shard_config.get("duration_seconds", max_shard_seconds)),
                    pose_bias=pose_bias,
                    response_gain=float(shard_config.get("response_gain", 0.76)),
                    contact_scale=float(shard_config.get("contact_scale", 1.0)),
                )
            )

        if not shard_items:
            manifests[task_id] = build_default_manifest(spec, max_shard_seconds=max_shard_seconds)
            continue

        manifests[task_id] = SurrogateMyoManifest(
            task_id=task_id,
            task_title=spec.title,
            max_shard_seconds=int(config.get("max_shard_seconds", max_shard_seconds)),
            simulation_origin=str(
                config.get(
                    "simulation_origin",
                    "surrogate_myo: external manifest-backed shard approximation for competition runtime",
                )
            ),
            shards=tuple(shard_items),
        )

    for spec in list_task_specs():
        manifests.setdefault(spec.task_id, build_default_manifest(spec, max_shard_seconds=max_shard_seconds))
    return manifests