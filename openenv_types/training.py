from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RLTechnique(str, Enum):
    PPO = "ppo"
    REINFORCE = "reinforce"
    A2C = "a2c"
    DQN = "dqn"
    BEHAVIOR_CLONING = "behavior_cloning"


class AlgorithmCapabilities(BaseModel):
    """
    Capabilities of a given training algorithm/technique.
    supports_discrete_actions: Whether the algorithm can handle discrete action spaces.
    supports_continuous_actions: Whether the algorithm can handle continuous action spaces.
    supports_multi_agent: Whether the algorithm can be applied in multi-agent settings.
    on_policy: Whether the algorithm is on-policy (e.g., PPO, A2C, REINFORCE) or off-policy (e.g., DQN).
    """
    supports_discrete_actions: bool = True
    supports_continuous_actions: bool = False
    supports_multi_agent: bool = False
    on_policy: bool = True


class TrainingTemplate(BaseModel):
    name: str
    technique: RLTechnique
    description: str
    recommended_for: list[str] = Field(default_factory=list)
    capabilities: AlgorithmCapabilities
    notes: list[str] = Field(default_factory=list)
