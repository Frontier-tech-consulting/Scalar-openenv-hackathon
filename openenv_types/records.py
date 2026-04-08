from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from openenv.core.tools.local_python_executor import LocalPythonExecutor

class StepTrace(BaseModel):
    step_index: int
    action_id: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    observation: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)


class EpisodeRecording(BaseModel):
    recording_id: str
    env_id: str
    task_id: Optional[str] = None
    play_mode: Optional[str] = None
    total_reward: float = 0.0
    solved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepTrace] = Field(default_factory=list)

    def append(self, trace: StepTrace) -> None:
        self.steps.append(trace)
        self.total_reward += trace.reward
        self.solved = self.solved or trace.info.get("solved", False)
