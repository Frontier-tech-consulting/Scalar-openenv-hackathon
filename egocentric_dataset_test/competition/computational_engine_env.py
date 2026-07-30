"""
Computational Engineering Environment for OpenEnv
Integrates rocket/IC engine simulation with OpenEnv framework
"""

from __future__ import annotations

import abc
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
except ImportError:
    # Fallback for development
    Environment = object
    Action = object
    Observation = object
    State = object

    @dataclass
    class EnvironmentMetadata:
        name: str
        description: str
        version: str
        author: str

from egocentric_dataset_test.competition.computational_engine_geometry import (
    ComputationalEngine, VoxelField, EngineSimulation, create_engine_demo
)


class BaseComputationalEngineEnvironment(abc.ABC):
    """Abstract base class defining the interface for computational engine environments."""

    @abc.abstractmethod
    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata."""

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, dict]:
        """Reset the environment to an initial state and return the first observation."""

    @abc.abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Apply *action* and return ``(observation, reward, terminated, truncated, info)``."""

    @abc.abstractmethod
    def get_state(self) -> State:
        """Return the current internal state."""

    @abc.abstractmethod
    def render(self) -> None:
        """Render the environment (e.g. for debugging or visualisation)."""


class ComputationalEngineAction(Action):
    """Action space for computational engine environment
    
    Attributes:
        engine_type: Type of engine to simulate: 'rocket' or 'ic'
        parameter_set: Parameter set for engine generation
        simulation_steps: Number of assembly simulation steps
        export_format: Export format for results: 'json', 'voxel', 'mesh'
    """
    engine_type: str = "rocket"
    parameter_set: str = "default"
    simulation_steps: int = 10
    export_format: str = "json"
    
    def __init__(self, **kwargs):
        super().__init__()
        # Set default values
        self.engine_type = "rocket"
        self.parameter_set = "default"
        self.simulation_steps = 10
        self.export_format = "json"
        # Override with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ComputationalEngineObservation(Observation):
    """Observation space for computational engine environment"""
    engine_generated: bool = False
    engine_type: str = ""
    voxel_dimensions: List[int] = []
    surface_voxel_count: int = 0
    simulation_progress: float = 0.0
    assembly_complete: bool = False
    engine_metrics: Dict[str, Any] = {}
    available_engines: List[str] = ["rocket", "ic"]
    last_action: Optional[str] = None
    
    def __init__(self, **kwargs):
        # Initialize parent Observation class
        super().__init__()
        # Set our custom fields with defaults
        self.engine_generated = False
        self.engine_type = ""
        self.voxel_dimensions = []
        self.surface_voxel_count = 0
        self.simulation_progress = 0.0
        self.assembly_complete = False
        self.engine_metrics = {}
        self.available_engines = ["rocket", "ic"]
        self.last_action = None
        # Override with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ComputationalEngineState(State):
    """State for computational engine environment"""
    engine_generated: bool = False
    engine_type: str = ""
    voxel_field: Optional[VoxelField] = None
    simulation: Optional[EngineSimulation] = None
    current_step: int = 0
    max_steps: int = 100
    total_reward: float = 0.0
    engine_metrics: Dict[str, Any] = {}
    assembly_complete: bool = False
    
    def __init__(self, **kwargs):
        # Initialize parent State class
        super().__init__()
        # Set our custom fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ComputationalEngineEnvironment(BaseComputationalEngineEnvironment, Environment):
    """
    Concrete OpenEnv environment for computational engine simulation.
    Integrates LEAP 71 PicoGK-inspired geometry generation with physics simulation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config: Dict[str, Any] = config or {}
        self._state: ComputationalEngineState = ComputationalEngineState()
        self.engine_designer: ComputationalEngine = ComputationalEngine(
            voxel_size=self.config.get("voxel_size", 2.0)
        )

    # ------------------------------------------------------------------
    # BaseComputationalEngineEnvironment implementation
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="computational_engine_sim",
            description="Computational engineering environment for rocket/IC engine simulation using PicoGK principles",
            version="1.0.0",
            author="LEAP 71 Inspired",
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self._state = ComputationalEngineState()
        self._state.max_steps = self.config.get("max_steps", 100)

        observation = ComputationalEngineObservation()
        info: Dict[str, Any] = {
            "reset_reason": "environment_reset",
            "available_engines": ["rocket", "ic"],
            "voxel_size": self.engine_designer.voxel_size,
        }

        return observation, info

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Take a step in the environment."""
        if not isinstance(action, ComputationalEngineAction):
            action = ComputationalEngineAction(
                engine_type=getattr(action, "engine_type", "rocket"),
                parameter_set=getattr(action, "parameter_set", "default"),
                simulation_steps=getattr(action, "simulation_steps", 10),
                export_format=getattr(action, "export_format", "json"),
            )

        self._state.engine_type = action.engine_type
        self._state.current_step += 1

        reward = 0.0
        done = False

        if not self._state.engine_generated:
            engine_field = self._generate_engine(action.engine_type)
            self._state.voxel_field = engine_field
            self._state.engine_generated = True

            surface_voxels = engine_field.extract_surface(threshold=1.0)
            self._state.engine_metrics = {
                "voxel_dimensions": list(engine_field.size),
                "surface_voxel_count": len(surface_voxels),
                "voxel_size": self.engine_designer.voxel_size,
                "estimated_volume_mm3": len(surface_voxels) * (self.engine_designer.voxel_size ** 3),
            }
            reward += 1.0

        if self._state.simulation is None:
            self._state.simulation = EngineSimulation()
            self._state.simulation.add_engine(
                self._state.voxel_field,
                engine_type=self._state.engine_type,
            )

        progress = min(self._state.current_step / self._state.max_steps, 1.0)
        self._state.simulation.simulate_assembly(steps=1)

        if progress >= 1.0:
            done = True
            reward += 2.0
            self._state.assembly_complete = True

        observation = ComputationalEngineObservation(
            engine_generated=self._state.engine_generated,
            engine_type=self._state.engine_type,
            voxel_dimensions=list(self._state.voxel_field.size) if self._state.voxel_field else [],
            surface_voxel_count=self._state.engine_metrics.get("surface_voxel_count", 0),
            simulation_progress=progress,
            assembly_complete=self._state.assembly_complete,
            engine_metrics=self._state.engine_metrics,
            last_action=f"generated_{action.engine_type}_engine",
            reward=reward,
            done=done,
        )

        self._state.total_reward += reward

        info: Dict[str, Any] = {
            "step": self._state.current_step,
            "engine_type": self._state.engine_type,
            "simulation_progress": progress,
            "total_reward": self._state.total_reward,
        }

        return observation, reward, done, False, info

    def get_state(self) -> ComputationalEngineState:
        """Return the current internal environment state."""
        return self._state

    def render(self) -> None:
        """Render the environment (for debugging)."""
        if self._state.voxel_field:
            print(f"Engine Voxel Field: {self._state.voxel_field.size}")
            if self._state.engine_metrics:
                print(f"Surface voxels: {self._state.engine_metrics.get('surface_voxel_count', 0)}")
                print(f"Estimated volume: {self._state.engine_metrics.get('estimated_volume_mm3', 0):.2f} mm³")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_engine(self, engine_type: str) -> VoxelField:
        """Generate engine geometry based on type."""
        if engine_type == "rocket":
            return self.engine_designer.create_rocket_engine(
                chamber_diameter=self.config.get("chamber_diameter", 40.0),
                chamber_length=self.config.get("chamber_length", 80.0),
                nozzle_exit_diameter=self.config.get("nozzle_exit_diameter", 25.0),
                nozzle_length=self.config.get("nozzle_length", 40.0),
                wall_thickness=self.config.get("wall_thickness", 3.0),
            )
        elif engine_type == "ic":
            return self.engine_designer.create_internal_combustion_engine(
                cylinder_bore=self.config.get("cylinder_bore", 60.0),
                stroke=self.config.get("stroke", 70.0),
                num_cylinders=self.config.get("num_cylinders", 4),
            )
        return self.engine_designer.create_rocket_engine()


def create_computational_engine_env(**kwargs) -> ComputationalEngineEnvironment:
    """Factory function to create computational engine environment"""
    return ComputationalEngineEnvironment(config=kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = create_computational_engine_env(
        voxel_size=2.0,
        max_steps=50
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print("Environment reset:")
    print(f"  Available engines: {info['available_engines']}")
    print(f"  Voxel size: {info['voxel_size']}")
    
    # Test rocket engine generation
    rocket_action = ComputationalEngineAction(
        engine_type="rocket",
        simulation_steps=20
    )
    
    obs, reward, done, truncated, info = env.step(rocket_action)
    print(f"\nStep 1 - Rocket Engine:")
    print(f"  Engine generated: {obs.engine_generated}")
    print(f"  Voxel dimensions: {obs.voxel_dimensions}")
    print(f"  Surface voxels: {obs.surface_voxel_count}")
    print(f"  Reward: {reward}")
    
    # Continue simulation
    for i in range(5):
        obs, reward, done, truncated, info = env.step(rocket_action)
        if done:
            break
        print(f"  Step {i+2}: Progress={obs.simulation_progress:.2f}, Reward={reward}")
    
    print(f"\nFinal state:")
    print(f"  Assembly complete: {obs.assembly_complete}")
    print(f"  Total reward: {info['total_reward']}")
    
    # Test IC engine
    print("\n" + "="*50)
    print("Testing Internal Combustion Engine...")
    
    obs, info = env.reset()
    ic_action = ComputationalEngineAction(
        engine_type="ic",
        simulation_steps=15
    )
    
    obs, reward, done, truncated, info = env.step(ic_action)
    print(f"IC Engine generated: {obs.engine_generated}")
    print(f"Voxel dimensions: {obs.voxel_dimensions}")
    print(f"Surface voxels: {obs.surface_voxel_count}")
    
    # Run to completion
    total_reward = reward
    for i in range(10):
        obs, reward, done, truncated, info = env.step(ic_action)
        total_reward += reward
        if done:
            break
        print(f"  Step {i+2}: Progress={obs.simulation_progress:.2f}")
    
    print(f"IC Engine simulation complete: {obs.assembly_complete}")
    print(f"Total reward: {total_reward}")