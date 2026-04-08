"""Typed environment, action, recording, and training descriptors for OpenEnv games."""

from .actions import ActionCatalog, ActionDescriptor, ActionKind, PlayMode
from .arc import ArcCommand, build_arc_descriptor
from .egocentric import (
    EgocentricAction,
    EgocentricActionCatalog,
    EgocentricDatasetSource,
    EgocentricIntrinsics,
    EgocentricObservation,
    EgocentricState,
    EgocentricEnvironmentDescriptor,
    EgocentricTrainingProfile,
    EgocentricVisualizerConfig,
    build_egocentric_descriptor,
    build_egocentric_action_catalog,
    build_egocentric_training_profile,
    EGOCENTRIC_DATASET_NAME,
    EGOCENTRIC_HF_DATASET_URL,
    EGOCENTRIC_HF_REPO_ID,
    EGOCENTRIC_INTRINSICS,
    EGOCENTRIC_S3_MIRROR_URI,
)
from .environment import EnvironmentDescriptor, ObservationFieldDescriptor, RLProjectionSpec
from .records import EpisodeRecording, StepTrace
from .training import AlgorithmCapabilities, RLTechnique, TrainingTemplate

__all__ = [
    "ActionCatalog",
    "ActionDescriptor",
    "ActionKind",
    "AlgorithmCapabilities",
    "ArcCommand",
    "EnvironmentDescriptor",
    "EpisodeRecording",
    "ObservationFieldDescriptor",
    "PlayMode",
    "RLProjectionSpec",
    "RLTechnique",
    "StepTrace",
    "TrainingTemplate",
    "build_arc_descriptor",
    # Egocentric-100K types
    "EgocentricAction",
    "EgocentricActionCatalog",
    "EgocentricDatasetSource",
    "EgocentricIntrinsics",
    "EgocentricObservation",
    "EgocentricState",
    "EgocentricEnvironmentDescriptor",
    "EgocentricTrainingProfile",
    "EgocentricVisualizerConfig",
    "build_egocentric_descriptor",
    "build_egocentric_action_catalog",
    "build_egocentric_training_profile",
    "EGOCENTRIC_DATASET_NAME",
    "EGOCENTRIC_HF_DATASET_URL",
    "EGOCENTRIC_HF_REPO_ID",
    "EGOCENTRIC_INTRINSICS",
    "EGOCENTRIC_S3_MIRROR_URI",
]
