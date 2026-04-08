"""
Egocentric-100K-specific typed models and catalog builders.

Extends the base openenv_types (Action, Observation, State) with
Egocentric-100K-specific fields for joint targets, camera intrinsics,
runtime selection, and RL projection.

Usage::

    from openenv_types.egocentric import (
        EgocentricAction,
        EgocentricObservation,
        EgocentricState,
        EgocentricEnvironmentDescriptor,
        EgocentricActionCatalog,
        build_egocentric_descriptor,
    )
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from openenv_types.actions import (
    ActionCatalog,
    ActionDescriptor,
    ActionKind,
    PlayMode,
)
from openenv_types.environment import (
    EnvironmentDescriptor,
    ObservationFieldDescriptor,
    RLProjectionSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Egocentric-100K Camera Intrinsics
# Sourced from builddotai/Egocentric-100K intrinsics.json on HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

EGOCENTRIC_INTRINSICS = {
    "model": "fisheye",
    "image_width": 456,
    "image_height": 256,
    "fx": 137.98,
    "fy": 138.23,
    "cx": 232.17,
    "cy": 125.37,
    "k1": 0.3948,
    "k2": 0.1798,
    "k3": -0.2753,
    "k4": 0.0793,
}

EGOCENTRIC_DATASET_NAME = "Egocentric-100K"
EGOCENTRIC_HF_REPO_ID = "builddotai/Egocentric-100K"
EGOCENTRIC_HF_DATASET_URL = "https://huggingface.co/datasets/builddotai/Egocentric-100K"
EGOCENTRIC_S3_MIRROR_URI = "s3://buildai-egocentric/raw"


class EgocentricIntrinsics(BaseModel):
    """OpenCV Kannala-Brandt fisheye camera intrinsics for Egocentric-100K."""

    model: Literal["fisheye"] = "fisheye"
    image_width: int = 456
    image_height: int = 256
    fx: float = 137.98
    fy: float = 138.23
    cx: float = 232.17
    cy: float = 125.37
    k1: float = 0.3948
    k2: float = 0.1798
    k3: float = -0.2753
    k4: float = 0.0793


class EgocentricDatasetSource(BaseModel):
    """Canonical dataset source metadata for Egocentric-100K."""

    dataset_name: str = EGOCENTRIC_DATASET_NAME
    huggingface_repo_id: str = EGOCENTRIC_HF_REPO_ID
    huggingface_url: str = EGOCENTRIC_HF_DATASET_URL
    s3_mirror_uri: str = EGOCENTRIC_S3_MIRROR_URI
    access_requires_hf_agreement: bool = True
    storage_format: str = "webdataset/mp4+h265"
    resolution: str = "456x256"
    fps: float = 30.0
    camera_model: str = "OpenCV fisheye (Kannala-Brandt)"
    metadata_schema_version: str = "egocentric100k/v1"


class EgocentricVisualizerConfig(BaseModel):
    """Configuration for the four-panel 3D visualization pipeline."""

    enabled: bool = False
    spawn_viewer: bool = True
    hand_tracking_enabled: bool = True
    point_cloud_enabled: bool = True
    expects_depth: bool = False
    intrinsics: EgocentricIntrinsics = Field(default_factory=EgocentricIntrinsics)


class EgocentricTrainingProfile(BaseModel):
    """Resolved training/runtime profile for local or OpenEnv-backed training."""

    mode: Literal["local", "openenv-http"] = "local"
    runtime: Literal["replay", "mujoco-humanoid"] = "replay"
    backend_url: str = "http://127.0.0.1:8000"
    task: str = "manual labor"
    cache_dir: str = "egocentric_dataset_test/data/egocentric100k_cache"
    output_dir: str = "egocentric_dataset_test/checkpoints/ego_openenv"
    episode_source: Literal["local", "s3", "simulator"] = "local"
    aws_region: str = "us-east-1"
    s3_uri: str = EGOCENTRIC_S3_MIRROR_URI
    s3_episode_name: str = ""
    simulator_env_id: str = "Humanoid-v5"
    max_episode_steps: int = 200
    action_dim: int = 4
    state_dim: int = 8
    dataset: EgocentricDatasetSource = Field(default_factory=EgocentricDatasetSource)
    visualizer: EgocentricVisualizerConfig = Field(default_factory=EgocentricVisualizerConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Egocentric-specific typed models
# ─────────────────────────────────────────────────────────────────────────────


class EgocentricAction(BaseModel):
    """
    Action for the Egocentric-100K environment.

    Mirrors the action space of the underlying environment:

    - **replay mode** (4-D): optical-flow derived pseudo-action
      [flow_magnitude, flow_angle, delta_red, delta_green]
    - **mujoco-humanoid mode** (17-D): MuJoCo humanoid torque action

    Attributes
    ----------
    joint_targets
        Joint target values; 4-D for replay, 17-D for humanoid.
    """

    joint_targets: list[float] = Field(
        default_factory=lambda: [0.0] * 4,
        description="Joint target values; 4-D for replay, 17-D for MuJoCo humanoid",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EgocentricObservation(BaseModel):
    """
    Observation from the Egocentric-100K environment.

    Attributes
    ----------
    step_idx
        Current step index in the episode.
    task
        Task description string derived from clip JSON metadata.
    state_values
        Proprioceptive state vector; 8-D for replay, 348-D for humanoid.
    image_shape
        Shape of the image observation [H, W, C].
    intrinsics
        Camera intrinsics for this worker (loaded from intrinsics.json).
    done
        Whether the episode has terminated.
    reward
        Reward signal from the last action.
    """

    step_idx: int = Field(default=0, description="Current step index in the episode")
    task: str = Field(
        default="Egocentric-100K manual labor clip",
        description="Task description derived from clip JSON metadata",
    )
    state_values: list[float] = Field(
        default_factory=list,
        description="Proprioceptive state vector; 8-D replay / 348-D humanoid",
    )
    image_shape: list[int] = Field(
        default_factory=lambda: [256, 456, 3],
        description="Image observation shape [H, W, C]",
    )
    intrinsics: EgocentricIntrinsics = Field(
        default_factory=EgocentricIntrinsics,
        description="OpenCV Kannala-Brandt fisheye camera intrinsics for this worker",
    )
    rgb_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded RGB frame payload for HTTP/OpenEnv visualization",
    )
    rgb_encoding: Literal["png", "jpeg"] | None = Field(
        default=None,
        description="Encoding used for rgb_base64 when present",
    )
    depth_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded depth frame payload for HTTP/OpenEnv visualization",
    )
    depth_encoding: Literal["png"] | None = Field(
        default=None,
        description="Encoding used for depth_base64 when present",
    )
    depth_shape: list[int] | None = Field(
        default=None,
        description="Optional decoded depth image shape [H, W] when depth_base64 is present",
    )
    depth_unit: Literal["millimeters"] | None = Field(
        default=None,
        description="Unit for transported depth values after HTTP normalization",
    )
    frame_payload_version: int = Field(
        default=0,
        description="Version of the serialized frame payload contract",
    )
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float | None = Field(default=None, description="Reward signal from the last action")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EgocentricState(BaseModel):
    """
    Internal state of the Egocentric-100K environment.

    Attributes
    ----------
    episode_id
        Unique episode identifier (e.g., "factory001_worker001_part000").
    step_count
        Number of steps taken in the current episode.
    cumulative_reward
        Cumulative reward for the current episode.
    runtime
        Active runtime backend ("replay" or "mujoco-humanoid").
    simulator_env_id
        Simulator environment id if running humanoid mode.
    intrinsics
        Camera intrinsics for this worker.
    total_steps
        Total steps in the episode (for replay mode, = episode length).
    """

    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(default=0, description="Number of steps taken in the current episode")
    cumulative_reward: float = Field(
        default=0.0, description="Cumulative reward for the current episode"
    )
    runtime: Literal["replay", "mujoco-humanoid"] = Field(
        default="replay", description="Active runtime backend"
    )
    simulator_env_id: str = Field(default="", description="Simulator environment id if applicable")
    intrinsics: EgocentricIntrinsics = Field(
        default_factory=EgocentricIntrinsics,
        description="Camera intrinsics for this worker",
    )
    total_steps: int = Field(default=0, description="Total steps in the episode")
    metadata: dict[str, Any] = Field(default_factory=dict)


EgocentricActionCatalog = ActionCatalog


# ─────────────────────────────────────────────────────────────────────────────
# Egocentric Action Catalog
# ─────────────────────────────────────────────────────────────────────────────

#: Runtime modes supported by the Egocentric-100K environment.
EGOCENTRIC_PLAY_MODES = [
    PlayMode.CONTINUOUS_CONTROL,
    PlayMode.RECORDING_REPLAY,
]


def build_egocentric_action_catalog(
    runtime: Literal["replay", "mujoco-humanoid"] = "replay",
) -> ActionCatalog:
    """
    Build the Egocentric-100K action catalog.

    Parameters
    ----------
    runtime
        "replay" → 4-D optical-flow pseudo-action space.
        "mujoco-humanoid" → 17-D MuJoCo humanoid torque space.

    Returns
    -------
    ActionCatalog
    """
    if runtime == "mujoco-humanoid":
        return ActionCatalog(
            name="egocentric_humanoid_actions",
            description="17-D MuJoCo humanoid torque action space for Egocentric-100K "
            "imitation + RL training. Each dimension maps to a torque actuator.",
            play_modes=EGOCENTRIC_PLAY_MODES,
            actions=[
                ActionDescriptor(
                    id="torque_joint_0",
                    label="Torque Joint 0",
                    description="Apply torque to humanoid joint 0",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_1",
                    label="Torque Joint 1",
                    description="Apply torque to humanoid joint 1",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_2",
                    label="Torque Joint 2",
                    description="Apply torque to humanoid joint 2",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_3",
                    label="Torque Joint 3",
                    description="Apply torque to humanoid joint 3",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_4",
                    label="Torque Joint 4",
                    description="Apply torque to humanoid joint 4",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_5",
                    label="Torque Joint 5",
                    description="Apply torque to humanoid joint 5",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_6",
                    label="Torque Joint 6",
                    description="Apply torque to humanoid joint 6",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_7",
                    label="Torque Joint 7",
                    description="Apply torque to humanoid joint 7",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_8",
                    label="Torque Joint 8",
                    description="Apply torque to humanoid joint 8",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_9",
                    label="Torque Joint 9",
                    description="Apply torque to humanoid joint 9",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_10",
                    label="Torque Joint 10",
                    description="Apply torque to humanoid joint 10",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_11",
                    label="Torque Joint 11",
                    description="Apply torque to humanoid joint 11",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_12",
                    label="Torque Joint 12",
                    description="Apply torque to humanoid joint 12",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_13",
                    label="Torque Joint 13",
                    description="Apply torque to humanoid joint 13",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_14",
                    label="Torque Joint 14",
                    description="Apply torque to humanoid joint 14",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_15",
                    label="Torque Joint 15",
                    description="Apply torque to humanoid joint 15",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
                ActionDescriptor(
                    id="torque_joint_16",
                    label="Torque Joint 16",
                    description="Apply torque to humanoid joint 16 (gripper)",
                    kind=ActionKind.CONTINUOUS,
                    requires_value=True,
                ),
            ],
        )

    # Replay mode — 4-D optical-flow pseudo-action
    return ActionCatalog(
        name="egocentric_replay_actions",
        description="4-D optical-flow derived pseudo-action space for Egocentric-100K "
        "Zarr episode replay. [flow_magnitude, flow_angle, delta_red, delta_green]",
        play_modes=EGOCENTRIC_PLAY_MODES,
        actions=[
            ActionDescriptor(
                id="flow_magnitude",
                label="Flow Magnitude",
                description="Magnitude of optical flow vector (normalized)",
                kind=ActionKind.CONTINUOUS,
                requires_value=True,
            ),
            ActionDescriptor(
                id="flow_angle",
                label="Flow Angle",
                description="Angle of optical flow vector (radians)",
                kind=ActionKind.CONTINUOUS,
                requires_value=True,
            ),
            ActionDescriptor(
                id="delta_red",
                label="Delta Red",
                description="Change in red channel mean (ΔR)",
                kind=ActionKind.CONTINUOUS,
                requires_value=True,
            ),
            ActionDescriptor(
                id="delta_green",
                label="Delta Green",
                description="Change in green channel mean (ΔG)",
                kind=ActionKind.CONTINUOUS,
                requires_value=True,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Egocentric Environment Descriptor
# ─────────────────────────────────────────────────────────────────────────────


def build_egocentric_descriptor(
    runtime: Literal["replay", "mujoco-humanoid"] = "replay",
    max_episode_steps: int = 200,
    task_description: str = "Egocentric-100K manual labor clip",
) -> EnvironmentDescriptor:
    """
    Build the Egocentric-100K environment descriptor.

    Parameters
    ----------
    runtime
        "replay" → 4-D action, 8-D state, Zarr episode replay.
        "mujoco-humanoid" → 17-D action, 348-D state, Gymnasium Humanoid-v5.
    max_episode_steps
        Maximum number of steps per episode.
    task_description
        Default task description string.

    Returns
    -------
    EnvironmentDescriptor
    """
    action_dim = 17 if runtime == "mujoco-humanoid" else 4
    state_dim = 348 if runtime == "mujoco-humanoid" else 8

    catalog = build_egocentric_action_catalog(runtime=runtime)

    projection = RLProjectionSpec(
        observation_dim=state_dim,
        action_count=action_dim,
        observation_encoder="egocentric_state_encoder_v1",
        action_encoder="egocentric_action_encoder_v1",
    )

    return EnvironmentDescriptor(
        env_id="egocentric_100k",
        title="Egocentric-100K OpenEnv Environment",
        description="Egocentric-100K: the world's largest manual labor dataset "
        "(100,405 hours, 10.8B frames, 24.79 TB). "
        "Supports Zarr episode replay and MuJoCo humanoid simulation.",
        play_modes=EGOCENTRIC_PLAY_MODES,
        action_catalog=catalog,
        observation_fields=[
            ObservationFieldDescriptor(
                name="step_idx",
                dtype="int",
                shape=[1],
                description="Current step index in the episode",
            ),
            ObservationFieldDescriptor(
                name="task",
                dtype="str",
                shape=[1],
                description="Task description from clip JSON metadata",
            ),
            ObservationFieldDescriptor(
                name="state_values",
                dtype="float",
                shape=[state_dim],
                description="Proprioceptive state vector",
            ),
            ObservationFieldDescriptor(
                name="image_shape",
                dtype="int",
                shape=[3],
                description="Image observation shape [H, W, C]",
            ),
            ObservationFieldDescriptor(
                name="intrinsics",
                dtype="float",
                shape=[8],  # fx, fy, cx, cy, k1, k2, k3, k4
                description="Camera intrinsics: Kannala-Brandt fisheye (fx, fy, cx, cy, k1, k2, k3, k4)",
                exposed_to_policy=False,  # metadata only — not sent to policy
            ),
            ObservationFieldDescriptor(
                name="rgb_base64",
                dtype="str",
                shape=[1],
                description="Optional base64-encoded RGB frame for OpenEnv HTTP visualization payloads",
                exposed_to_policy=False,
            ),
            ObservationFieldDescriptor(
                name="depth_base64",
                dtype="str",
                shape=[1],
                description="Optional base64-encoded uint16 depth frame for OpenEnv HTTP visualization payloads",
                exposed_to_policy=False,
            ),
        ],
        supports_recordings=True,
        supports_swarms=False,
        supports_server=True,
        max_episode_steps=max_episode_steps,
        rl_projection=projection,
        metadata={
            "dataset": EGOCENTRIC_HF_REPO_ID,
            "dataset_name": EGOCENTRIC_DATASET_NAME,
            "huggingface_url": EGOCENTRIC_HF_DATASET_URL,
            "s3_mirror": EGOCENTRIC_S3_MIRROR_URI,
            "access_requires_hf_agreement": True,
            "total_hours": 100405,
            "total_frames": 10_800_000_000,
            "total_clips": 2_010_759,
            "resolution": "456x256",
            "fps": 30.0,
            "codec": "h265",
            "camera_model": "fisheye (Kannala-Brandt)",
            "intrinsics_defaults": EGOCENTRIC_INTRINSICS,
        },
    )


def build_egocentric_training_profile(
    *,
    mode: Literal["local", "openenv-http"] = "local",
    runtime: Literal["replay", "mujoco-humanoid"] = "replay",
    task: str = "manual labor",
    backend_url: str = "http://127.0.0.1:8000",
    cache_dir: str = "egocentric_dataset_test/data/egocentric100k_cache",
    output_dir: str = "egocentric_dataset_test/checkpoints/ego_openenv",
    episode_source: Literal["local", "s3", "simulator"] = "local",
    aws_region: str = "us-east-1",
    s3_uri: str = EGOCENTRIC_S3_MIRROR_URI,
    s3_episode_name: str = "",
    simulator_env_id: str = "Humanoid-v5",
    max_episode_steps: int = 200,
    visualizer_enabled: bool = False,
    visualizer_spawn: bool = True,
    intrinsics: EgocentricIntrinsics | None = None,
) -> EgocentricTrainingProfile:
    action_dim = 17 if runtime == "mujoco-humanoid" else 4
    state_dim = 348 if runtime == "mujoco-humanoid" else 8
    return EgocentricTrainingProfile(
        mode=mode,
        runtime=runtime,
        backend_url=backend_url,
        task=task,
        cache_dir=cache_dir,
        output_dir=output_dir,
        episode_source=episode_source,
        aws_region=aws_region,
        s3_uri=s3_uri,
        s3_episode_name=s3_episode_name,
        simulator_env_id=simulator_env_id,
        max_episode_steps=max_episode_steps,
        action_dim=action_dim,
        state_dim=state_dim,
        visualizer=EgocentricVisualizerConfig(
            enabled=visualizer_enabled,
            spawn_viewer=visualizer_spawn,
            intrinsics=intrinsics or EgocentricIntrinsics(),
        ),
    )


# Alias for convenience
EgocentricEnvironmentDescriptor = EnvironmentDescriptor
"""Type alias: EgocentricEnvironmentDescriptor is just EnvironmentDescriptor."""
