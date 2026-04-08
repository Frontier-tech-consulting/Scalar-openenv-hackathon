from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
class PlayMode(str, Enum):
    TURN_BASED = "turn_based"
    GRID_EDIT = "grid_edit"
    RECORDING_REPLAY = "recording_replay"
    CONTINUOUS_CONTROL = "continuous_control"
    SWARM = "swarm"
    TOOL_USE = "tool_use"


class ActionKind(str, Enum):
    ENUM = "enum"
    COORDINATE = "coordinate"
    CONTINUOUS = "continuous"
    TOOL = "tool"
    COMPOSITE = "composite"


class ActionDescriptor(BaseModel):
    id: str = Field(description="Stable identifier used by policies and servers")
    label: str = Field(description="Human-readable action label")
    description: str = Field(description="What the action does")
    kind: ActionKind = Field(default=ActionKind.ENUM)
    requires_coordinates: bool = Field(default=False)
    requires_value: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionCatalog(BaseModel):
    name: str
    description: str
    play_modes: list[PlayMode] = Field(default_factory=list)
    actions: list[ActionDescriptor] = Field(default_factory=list)
    submit_action_id: Optional[str] = None
    reset_action_id: Optional[str] = None

    def by_id(self, action_id: str) -> ActionDescriptor:
        for action in self.actions:
            if action.id == action_id:
                return action
        raise KeyError(f"Unknown action id: {action_id}")

    def discrete_count(self, coordinate_bins: Optional[int] = None) -> int:
        total = 0
        for action in self.actions:
            if action.kind == ActionKind.COORDINATE:
                if coordinate_bins is None:
                    raise ValueError("coordinate_bins is required for coordinate actions")
                total += coordinate_bins
            else:
                total += 1
        return total
