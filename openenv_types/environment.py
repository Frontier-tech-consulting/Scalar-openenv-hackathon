from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .actions import ActionCatalog, PlayMode


class ObservationFieldDescriptor(BaseModel):
    name: str
    dtype: str
    shape: list[int] = Field(default_factory=list)
    description: str
    exposed_to_policy: bool = True


class RLProjectionSpec(BaseModel):
    observation_dim: int
    action_count: int
    observation_encoder: str
    action_encoder: str


class EnvironmentDescriptor(BaseModel):
    env_id: str
    title: str
    description: str
    play_modes: list[PlayMode]
    action_catalog: ActionCatalog
    observation_fields: list[ObservationFieldDescriptor]
    supports_recordings: bool = True
    supports_swarms: bool = False
    supports_server: bool = True
    max_episode_steps: int = 100
    max_grid_size: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    rl_projection: Optional[RLProjectionSpec] = None
