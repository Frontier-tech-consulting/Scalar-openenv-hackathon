from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from egocentric_dataset_test.competition.mujoco_sim import MYOHAND_MUSCLE_NAMES, MYOFINGER_MUSCLE_NAMES

OPENENV_TO_MYOSIM_TASK_MAP = {
    "easy_bin_pick": "easy_finger_reach",
    "medium_sort_and_place": "medium_hand_grasp",
    "hard_precision_assembly": "hard_precision_assembly",
}


@dataclass(slots=True)
class OpenEnvMyoBridgeState:
    openenv_task_id: str
    myosim_task_id: str
    current_stage: str
    stage_index: int
    progress: float
    action_hint: np.ndarray
    state_values: np.ndarray
    guidance: str
    active_stage_target: np.ndarray
    trace_length: int


class OpenEnvToMyoSimAdapter:
    def __init__(self, bridge_state: OpenEnvMyoBridgeState) -> None:
        self.bridge_state = bridge_state
        self.target_position = self._map_hint_to_target_position()
        self.grip_prior = self._phase_grip_level()
        self.wrist_roll_prior = float(np.clip(self.bridge_state.action_hint[3], -1.0, 1.0))
        self.rotation_target = self._map_rotation_target()

    @classmethod
    def from_openenv(
        cls,
        observation: Any,
        state: Any,
        trace: list[dict[str, Any]] | None = None,
    ) -> "OpenEnvToMyoSimAdapter":
        metadata = getattr(observation, "metadata", None) or {}
        action_hint = np.asarray(getattr(observation, "action_hint", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        if action_hint.size < 4:
            action_hint = np.pad(action_hint, (0, 4 - action_hint.size))
        action_hint = action_hint[:4]
        state_values = np.asarray(getattr(observation, "state_values", []), dtype=np.float32)
        active_stage_target = np.asarray(metadata.get("active_stage_target", action_hint.tolist()), dtype=np.float32)
        if active_stage_target.size < 4:
            active_stage_target = np.pad(active_stage_target, (0, 4 - active_stage_target.size))
        active_stage_target = active_stage_target[:4]
        myosim_task_id = OPENENV_TO_MYOSIM_TASK_MAP.get(observation.task_id, "easy_finger_reach")
        bridge_state = OpenEnvMyoBridgeState(
            openenv_task_id=str(observation.task_id),
            myosim_task_id=myosim_task_id,
            current_stage=str(getattr(observation, "current_stage", "")),
            stage_index=int(getattr(observation, "stage_index", 0)),
            progress=float(getattr(state, "progress", getattr(observation, "progress", 0.0))),
            action_hint=action_hint,
            state_values=state_values,
            guidance=str(getattr(observation, "stage_guidance", "")),
            active_stage_target=active_stage_target,
            trace_length=len(trace or []),
        )
        return cls(bridge_state)

    def configure_task(self, myosim_task: Any) -> None:
        target = self.target_position.astype(np.float32)
        if hasattr(myosim_task, "_target"):
            myosim_task._target = target.copy()
        if hasattr(myosim_task, "_object_target"):
            myosim_task._object_target = target.copy()
        if hasattr(myosim_task, "_rotation_target"):
            myosim_task._rotation_target = self.rotation_target

    def adapt_openenv_observation(self) -> dict[str, Any]:
        conditioning = np.array(
            [
                float(self.target_position[0]),
                float(self.target_position[1]),
                float(self.target_position[2]),
                float(self.grip_prior),
                float(self.wrist_roll_prior),
                float(np.clip(self.bridge_state.progress, 0.0, 1.0)),
                float(np.clip(self.bridge_state.stage_index / 5.0, 0.0, 1.0)),
                float(np.clip(self.bridge_state.trace_length / 8.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return {
            "phase": self.bridge_state.current_stage,
            "guidance": self.bridge_state.guidance,
            "myosim_task_id": self.bridge_state.myosim_task_id,
            "target_position": self.target_position.tolist(),
            "grip_prior": float(self.grip_prior),
            "wrist_roll_prior": float(self.wrist_roll_prior),
            "rotation_target": float(self.rotation_target),
            "conditioning_vector": conditioning.tolist(),
        }

    def adapt_myosim_observation(self, sim_obs: dict[str, Any]) -> dict[str, Any]:
        current_position = self._extract_current_position(sim_obs)
        target_position = self._extract_target_position(sim_obs)
        delta = target_position - current_position
        fingertip_summary = np.asarray(sim_obs.get("fingertip_position", sim_obs.get("fingertip_positions", [])), dtype=np.float32).reshape(-1)
        if fingertip_summary.size >= 3:
            fingertip_summary = fingertip_summary[:3]
        elif fingertip_summary.size == 0:
            fingertip_summary = np.zeros(3, dtype=np.float32)
        else:
            fingertip_summary = np.pad(fingertip_summary, (0, 3 - fingertip_summary.size))

        joint_positions = np.asarray(sim_obs.get("joint_positions", []), dtype=np.float32)
        joint_velocities = np.asarray(sim_obs.get("joint_velocities", []), dtype=np.float32)
        muscle_activations = np.asarray(sim_obs.get("muscle_activations", []), dtype=np.float32)
        policy_observation = np.concatenate(
            [
                fingertip_summary,
                delta.astype(np.float32),
                np.array(
                    [
                        float(self.grip_prior),
                        float(self.wrist_roll_prior),
                        float(np.clip(self.bridge_state.progress, 0.0, 1.0)),
                        float(np.clip(self.bridge_state.stage_index / 5.0, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
                joint_positions[: min(6, joint_positions.size)],
                joint_velocities[: min(6, joint_velocities.size)],
                muscle_activations[: min(6, muscle_activations.size)],
            ]
        )
        return {
            "current_position": current_position.tolist(),
            "target_position": target_position.tolist(),
            "delta_to_target": delta.tolist(),
            "policy_observation": policy_observation.tolist(),
        }

    def action(self, sim_obs: dict[str, Any], n_actuators: int) -> np.ndarray:
        adapted_obs = self.adapt_myosim_observation(sim_obs)
        delta = np.asarray(adapted_obs["delta_to_target"], dtype=np.float32)
        if self.bridge_state.myosim_task_id == "easy_finger_reach":
            return self._finger_action(delta, n_actuators)
        return self._hand_action(delta, n_actuators)

    def summary(self) -> dict[str, Any]:
        return {
            "openenv_task_id": self.bridge_state.openenv_task_id,
            "myosim_task_id": self.bridge_state.myosim_task_id,
            "phase": self.bridge_state.current_stage,
            "target_position": self.target_position.tolist(),
            "grip_prior": float(self.grip_prior),
            "wrist_roll_prior": float(self.wrist_roll_prior),
            "rotation_target": float(self.rotation_target),
            "conditioning": self.adapt_openenv_observation(),
        }

    def _map_hint_to_target_position(self) -> np.ndarray:
        reach_x, reach_y, grip_force, _wrist_roll = [float(v) for v in self.bridge_state.action_hint[:4]]
        progress = float(np.clip(self.bridge_state.progress, 0.0, 1.0))
        stage_name = self.bridge_state.current_stage.lower()

        if self.bridge_state.myosim_task_id == "easy_finger_reach":
            base = np.array([0.19, 0.00, 0.25], dtype=np.float32)
            scale = np.array([0.08, 0.05, 0.06], dtype=np.float32)
        elif self.bridge_state.myosim_task_id == "medium_hand_grasp":
            base = np.array([0.15, 0.00, 0.33], dtype=np.float32)
            scale = np.array([0.07, 0.06, 0.05], dtype=np.float32)
        else:
            base = np.array([0.18, 0.00, 0.31], dtype=np.float32)
            scale = np.array([0.08, 0.07, 0.05], dtype=np.float32)

        target = base + np.array(
            [
                scale[0] * np.clip(reach_x, -1.0, 1.0),
                scale[1] * np.clip(reach_y, -1.0, 1.0),
                scale[2] * np.clip(grip_force, 0.0, 1.0) + 0.02 * progress,
            ],
            dtype=np.float32,
        )
        if any(token in stage_name for token in ("lift", "inspect", "hold")):
            target[2] += 0.02
        if any(token in stage_name for token in ("move_to_tray", "sort", "transfer")):
            target[0] -= 0.015
        if any(token in stage_name for token in ("insert", "seat", "align")):
            target[0] += 0.02
            target[1] += 0.01
        return target.astype(np.float32)

    def _map_rotation_target(self) -> float:
        wrist_roll = float(np.clip(self.bridge_state.action_hint[3], -1.0, 1.0))
        return float((wrist_roll + 1.0) * 0.5 * 1.57)

    def _phase_grip_level(self) -> float:
        stage_name = self.bridge_state.current_stage.lower()
        hinted = float(np.clip(self.bridge_state.action_hint[2], 0.0, 1.0))
        if any(token in stage_name for token in ("approach", "reach", "release")):
            return min(hinted, 0.25)
        if any(token in stage_name for token in ("secure", "grasp", "pinch", "insert", "seat", "hold")):
            return max(hinted, 0.7)
        if any(token in stage_name for token in ("align", "rotate")):
            return max(hinted, 0.55)
        return hinted

    def _extract_current_position(self, sim_obs: dict[str, Any]) -> np.ndarray:
        if "object_position" in sim_obs:
            return np.asarray(sim_obs.get("object_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        if "fingertip_position" in sim_obs:
            return np.asarray(sim_obs.get("fingertip_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        fingertips = np.asarray(sim_obs.get("fingertip_positions", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if fingertips.size >= 3:
            return fingertips[:3]
        if fingertips.size == 0:
            return np.zeros(3, dtype=np.float32)
        return np.pad(fingertips, (0, 3 - fingertips.size))

    def _extract_target_position(self, sim_obs: dict[str, Any]) -> np.ndarray:
        if "target_position" in sim_obs:
            return np.asarray(sim_obs.get("target_position", self.target_position.tolist()), dtype=np.float32)
        if "object_target" in sim_obs:
            return np.asarray(sim_obs.get("object_target", self.target_position.tolist()), dtype=np.float32)
        return self.target_position.copy()

    def _finger_action(self, delta: np.ndarray, n_actuators: int) -> np.ndarray:
        action = np.zeros(n_actuators, dtype=np.float32)
        if n_actuators <= 0:
            return action
        x_err, y_err, z_err = [float(v) for v in np.pad(delta, (0, max(0, 3 - delta.size)))[:3]]
        grip = float(self.grip_prior)
        open_level = 1.0 - grip
        wrist = float(self.wrist_roll_prior)

        extensor = np.clip(0.65 * open_level + 0.35 * max(-z_err, 0.0), 0.0, 1.0)
        right_abduction = np.clip(3.0 * max(y_err, 0.0) + 0.2 * max(wrist, 0.0), 0.0, 1.0)
        left_abduction = np.clip(3.0 * max(-y_err, 0.0) + 0.2 * max(-wrist, 0.0), 0.0, 1.0)
        flex_drive = np.clip(2.5 * max(x_err, 0.0) + 1.8 * max(z_err, 0.0) + grip, 0.0, 1.0)
        action[: min(n_actuators, 5)] = np.array(
            [extensor, right_abduction, left_abduction, 0.75 * flex_drive, flex_drive],
            dtype=np.float32,
        )[: min(n_actuators, 5)]
        if "release" in self.bridge_state.current_stage.lower():
            action[0] = max(action[0], 0.7)
            if n_actuators >= 5:
                action[3] *= 0.25
                action[4] *= 0.2
        return np.clip(action, 0.0, 1.0)

    def _hand_action(self, delta: np.ndarray, n_actuators: int) -> np.ndarray:
        action = np.zeros(n_actuators, dtype=np.float32)
        if n_actuators <= 0:
            return action

        x_err, y_err, z_err = [float(v) for v in np.pad(delta, (0, max(0, 3 - delta.size)))[:3]]
        grip = float(self.grip_prior)
        wrist = float(self.wrist_roll_prior)
        stage_name = self.bridge_state.current_stage.lower()

        extensor = float(np.clip((1.0 - grip) * 0.8 + max(-z_err, 0.0) * 0.4, 0.0, 1.0))
        flexor = float(np.clip(grip * 0.85 + max(z_err, 0.0) * 0.25 + max(x_err, 0.0) * 0.25, 0.0, 1.0))
        wrist_ext = float(np.clip(max(-x_err, 0.0) * 0.7 + (1.0 - grip) * 0.1, 0.0, 1.0))
        wrist_flex = float(np.clip(0.25 + max(x_err, 0.0) * 0.7, 0.0, 1.0))
        radial = float(np.clip(max(wrist, 0.0) * 0.8 + max(y_err, 0.0) * 0.5, 0.0, 1.0))
        ulnar = float(np.clip(max(-wrist, 0.0) * 0.8 + max(-y_err, 0.0) * 0.5, 0.0, 1.0))

        if any(token in stage_name for token in ("align", "insert", "seat")):
            flexor = max(flexor, 0.45)
            extensor = max(extensor, 0.25)
        if "release" in stage_name:
            flexor *= 0.3
            extensor = max(extensor, 0.65)

        if n_actuators >= len(MYOHAND_MUSCLE_NAMES):
            name_to_index = {name: idx for idx, name in enumerate(MYOHAND_MUSCLE_NAMES)}
            action[name_to_index["ECRL"]] = wrist_ext + 0.2 * radial
            action[name_to_index["ECRB"]] = wrist_ext
            action[name_to_index["ECU"]] = wrist_ext + 0.2 * ulnar
            action[name_to_index["FCR"]] = wrist_flex + 0.2 * radial
            action[name_to_index["FCU"]] = wrist_flex + 0.2 * ulnar
            action[name_to_index["PL"]] = wrist_flex * 0.7
            action[name_to_index["EIP"]] = extensor * 0.9
            action[name_to_index["EPL"]] = extensor * 0.6
            action[name_to_index["EPB"]] = extensor * 0.4
            action[name_to_index["FPL"]] = flexor * 0.8
            action[name_to_index["APL"]] = 0.3 + 0.3 * max(wrist, 0.0)
            action[name_to_index["OP"]] = 0.35 + 0.55 * grip
            for muscle_name in ("FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5"):
                action[name_to_index[muscle_name]] = flexor
            for muscle_name in ("EDC2", "EDC3", "EDC4", "EDC5", "EDM"):
                action[name_to_index[muscle_name]] = extensor
            for muscle_name in ("RI2", "RI3", "RI4", "RI5", "LU2", "LU3", "LU4", "LU5", "UI2", "UI3", "UI4", "UI5"):
                action[name_to_index[muscle_name]] = 0.25 + 0.65 * grip
        else:
            action[:] = np.clip(0.3 + flexor, 0.0, 1.0)

        return np.clip(action, 0.0, 1.0)
