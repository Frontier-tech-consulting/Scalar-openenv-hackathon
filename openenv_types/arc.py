from __future__ import annotations

from enum import Enum

from .actions import ActionCatalog, ActionDescriptor, ActionKind, PlayMode
from .environment import EnvironmentDescriptor, ObservationFieldDescriptor, RLProjectionSpec


class ArcCommand(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"
    SUBMIT = "SUBMIT"


def build_arc_descriptor(max_grid_size: int = 6, max_episode_steps: int = 50) -> EnvironmentDescriptor:
    coordinate_bins = max_grid_size * max_grid_size
    catalog = ActionCatalog(
        name="arc_arcprize_actions",
        description="ARC Prize inspired action set aligned with the documented ACTION1-7 toolkit style.",
        play_modes=[PlayMode.TURN_BASED, PlayMode.GRID_EDIT, PlayMode.RECORDING_REPLAY],
        reset_action_id=ArcCommand.RESET.value,
        submit_action_id=ArcCommand.SUBMIT.value,
        actions=[
            ActionDescriptor(id=ArcCommand.ACTION1.value, label="Cursor Up", description="Move selection cursor up."),
            ActionDescriptor(id=ArcCommand.ACTION2.value, label="Cursor Down", description="Move selection cursor down."),
            ActionDescriptor(id=ArcCommand.ACTION3.value, label="Cursor Left", description="Move selection cursor left."),
            ActionDescriptor(id=ArcCommand.ACTION4.value, label="Cursor Right", description="Move selection cursor right."),
            ActionDescriptor(id=ArcCommand.ACTION5.value, label="Cycle Color", description="Cycle the current paint color."),
            ActionDescriptor(id=ArcCommand.ACTION6.value, label="Paint Coordinate", description="Paint a cell at (x,y) with the selected color.", kind=ActionKind.COORDINATE, requires_coordinates=True),
            ActionDescriptor(id=ArcCommand.ACTION7.value, label="Undo", description="Undo the previous edit."),
            ActionDescriptor(id=ArcCommand.SUBMIT.value, label="Submit", description="Submit the current grid as the answer."),
        ],
    )
    projection = RLProjectionSpec(
        observation_dim=(max_grid_size * max_grid_size * 3) + 4,
        action_count=catalog.discrete_count(coordinate_bins=coordinate_bins),
        observation_encoder="arc_grid_encoder_v2",
        action_encoder="arc_cursor_paint_encoder_v1",
    )
    return EnvironmentDescriptor(
        env_id="arc_agi",
        title="ARC-AGI OpenEnv Environment",
        description="OpenEnv-native ARC-style grid reasoning environment with train examples, recordings, and a compact RL action space.",
        play_modes=[PlayMode.TURN_BASED, PlayMode.GRID_EDIT, PlayMode.RECORDING_REPLAY],
        action_catalog=catalog,
        observation_fields=[
            ObservationFieldDescriptor(name="current_grid", dtype="int", shape=[max_grid_size, max_grid_size], description="Current editable test grid."),
            ObservationFieldDescriptor(name="training_examples", dtype="int", shape=[2, 2, max_grid_size, max_grid_size], description="Example input/output pairs defining the task rule."),
            ObservationFieldDescriptor(name="cursor", dtype="int", shape=[2], description="Current cursor position."),
            ObservationFieldDescriptor(name="selected_color", dtype="int", shape=[1], description="Currently selected paint color."),
            ObservationFieldDescriptor(name="similarity", dtype="float", shape=[1], description="Similarity to hidden target."),
        ],
        supports_recordings=True,
        supports_swarms=True,
        supports_server=True,
        max_episode_steps=max_episode_steps,
        max_grid_size=max_grid_size,
        rl_projection=projection,
        metadata={"coordinate_bins": coordinate_bins, "docs": ["https://docs.arcprize.org/game-schema", "https://docs.arcprize.org/toolkit/list-actions", "https://docs.arcprize.org/recordings", "https://docs.arcprize.org/swarms"]},
    )
