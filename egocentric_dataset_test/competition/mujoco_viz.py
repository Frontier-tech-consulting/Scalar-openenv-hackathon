"""
MuJoCo 3D Visualization Component for Gradio Frontend
=====================================================

Provides interactive 3D simulation panels for MyoFinger and MyoHand tasks,
rendered as live images in the Gradio UI. Supports both real MuJoCo rendering
and procedural fallback visualization.
"""

from __future__ import annotations

import io
import tempfile
from typing import Any

import numpy as np

try:
    import gradio as gr
except Exception:
    gr = None  # type: ignore[assignment]

from egocentric_dataset_test.competition.mujoco_sim import (
    HAS_MUJOCO,
    MUJOCO_TASK_REGISTRY,
    MyoFingerReachTask,
    MyoHandGraspTask,
    MyoHandPrecisionTask,
    create_mujoco_task,
    list_mujoco_tasks,
)


# ---------------------------------------------------------------------------
# Task descriptions for the UI
# ---------------------------------------------------------------------------
TASK_DESCRIPTIONS = {
    "easy_finger_reach": (
        "## MyoFinger Tip Reach (Easy)\n\n"
        "Control the **MyoFinger** — a 4-DoF musculoskeletal finger model with 5 muscle-tendon "
        "actuators (extn, adabR, adabL, mflx, dflx) — to reach a target position.\n\n"
        "This task mirrors the MyoSuite `myoFingerReachFixed-v0` environment, grounded in "
        "the Egocentric-100K factory assembly pipeline.\n\n"
        "**Joints:** IFadb (abduction), IFmcp (MCP flexion), IFpip (PIP flexion), IFdip (DIP flexion)\n\n"
        "**Muscles:** Central Extensor, Abduction R/L, PIP Flexor, DIP Flexor"
    ),
    "medium_hand_grasp": (
        "## MyoHand Object Grasp (Medium)\n\n"
        "Control the **MyoHand** — a 23-DoF dexterous hand model with 39 muscle-tendon "
        "actuators — to grasp and hold an object.\n\n"
        "This task mirrors the MyoSuite `myoHandObjHoldFixed-v0` environment, requiring "
        "coordination of wrist, thumb, and four fingers with intermittent contacts.\n\n"
        "**Key muscles:** ECRL, ECRB, FCR, FCU (wrist), FDS/FDP (flexors), EDC (extensors), "
        "RI/LU/UI (interossei), FPL/EPL (thumb)"
    ),
    "hard_precision_assembly": (
        "## MyoHand Precision Assembly (Hard)\n\n"
        "Control the **MyoHand** to perform a precision key-turn / insertion task, "
        "requiring fine motor coordination of all 39 muscles.\n\n"
        "This task mirrors the MyoSuite `myoHandKeyTurnFixed-v0` environment, demanding "
        "simultaneous grasp stability and rotational manipulation.\n\n"
        "**Challenge:** Coordinate thumb-index pinch with wrist rotation while maintaining "
        "object stability through intermittent contacts."
    ),
}

MUSCLE_GROUP_LABELS = {
    "myofinger": {
        "extn": "Central Extensor",
        "adabR": "Abduction (R)",
        "adabL": "Abduction (L)",
        "mflx": "PIP Flexor",
        "dflx": "DIP Flexor",
    },
    "myohand": {
        "ECRL": "Ext. Carpi Rad. Longus",
        "ECRB": "Ext. Carpi Rad. Brevis",
        "ECU": "Ext. Carpi Ulnaris",
        "FCR": "Flex. Carpi Radialis",
        "FCU": "Flex. Carpi Ulnaris",
        "FDS2": "Flex. Dig. Superf. (Index)",
        "FDP2": "Flex. Dig. Prof. (Index)",
        "EDC2": "Ext. Dig. Comm. (Index)",
        "FPL": "Flex. Pollicis Longus",
        "EPL": "Ext. Pollicis Longus",
    },
}


# ---------------------------------------------------------------------------
# Active simulation state (module-level for Gradio state management)
# ---------------------------------------------------------------------------
_active_task = None


def _get_active_task(task_id: str, seed: int = 7):
    global _active_task
    if _active_task is not None:
        _active_task.close()
    _active_task = create_mujoco_task(task_id, seed=seed)
    return _active_task


def _reset_mujoco_task(task_id: str, seed: float) -> list[Any]:
    """Reset a MuJoCo simulation task and return initial observation + rendered frame."""
    task = _get_active_task(task_id, seed=int(seed))
    obs = task.reset(seed=int(seed))
    frame = task.render()

    # Save frame to temp file for Gradio Image component
    frame_path = None
    if frame is not None:
        try:
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(frame).save(tmp.name)
            frame_path = tmp.name
        except Exception:
            frame_path = None

    desc = TASK_DESCRIPTIONS.get(task_id, f"## {task_id}")
    mujoco_status = "✅ MuJoCo 3D physics active" if HAS_MUJOCO else "⚠️ Procedural fallback (install mujoco for 3D)"

    return [
        task,                          # state
        obs,                           # observation json
        frame_path,                     # rendered frame
        desc,                           # task description
        mujoco_status,                  # status
        obs.get("reward", 0.0),        # reward
        obs.get("done", False),         # done
        obs.get("step_count", 0),       # step count
    ]


def _step_mujoco_task(
    task,
    muscle_0: float, muscle_1: float, muscle_2: float, muscle_3: float, muscle_4: float,
    *extra_muscles: float,
) -> list[Any]:
    """Apply muscle activations and step the simulation."""
    if task is None:
        return [None, None, "No active task. Reset first.", "", 0.0, False, 0]

    action = [muscle_0, muscle_1, muscle_2, muscle_3, muscle_4]
    for m in extra_muscles:
        action.append(float(m))

    # Pad or truncate to match actuator count
    n_act = task.sim.n_actuators
    while len(action) < n_act:
        action.append(0.0)
    action = action[:n_act]

    obs = task.step(np.array(action, dtype=np.float64))
    frame = task.render()

    frame_path = None
    if frame is not None:
        try:
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(frame).save(tmp.name)
            frame_path = tmp.name
        except Exception:
            frame_path = None

    return [
        task,
        obs,
        frame_path,
        obs.get("reward", 0.0),
        obs.get("done", False),
        obs.get("step_count", 0),
    ]


def _run_auto_policy(task) -> list[Any]:
    """Run the built-in PD policy to completion."""
    if task is None:
        return [None, None, "No active task. Reset first.", 0.0, False, 0]

    frames = []
    obs = task.reset()
    for _ in range(task.max_steps):
        # Simple PD controller: activate flexors proportional to distance
        target = np.array(obs.get("target_position", obs.get("object_target", [0.25, 0.01, 0.25])))
        tip = np.array(obs.get("fingertip_position", [0.0, 0.0, 0.3]))
        error = target - tip
        # Map error to muscle activations
        n_act = task.sim.n_actuators
        action = np.zeros(n_act)
        if n_act >= 5:
            action[0] = max(0.0, -error[0]) * 2.0  # extn for x correction
            action[3] = max(0.0, error[2]) * 2.0    # mflx for z correction
            action[4] = max(0.0, error[2]) * 1.5     # dflx for z correction
            if n_act >= 5:
                action[1] = max(0.0, error[1]) * 2.0  # adabR
                action[2] = max(0.0, -error[1]) * 2.0  # adabL
        action = np.clip(action, 0.0, 1.0)

        obs = task.step(action)
        frame = task.render()
        if frame is not None:
            frames.append(frame)
        if obs.get("done", False):
            break

    # Return last frame
    frame_path = None
    if frames:
        try:
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(frames[-1]).save(tmp.name)
            frame_path = tmp.name
        except Exception:
            frame_path = None

    return [
        task,
        obs,
        frame_path,
        obs.get("reward", 0.0),
        obs.get("done", False),
        obs.get("step_count", 0),
    ]


def create_mujoco_tab() -> Any:
    """Create the MuJoCo 3D Simulation tab for the Gradio UI."""
    if gr is None:
        return None

    task_state = gr.State(value=None)

    gr.Markdown(
        "## MyoFinger / MyoHand MuJoCo 3D Simulation\n\n"
        "Real-time musculoskeletal simulation grounded in the **MyoSuite** (MyoHub/myo_sim) models. "
        "This panel runs actual MuJoCo physics with tendon-driven muscle actuators, matching the "
        "Egocentric-100K factory assembly pipeline architecture.\n\n"
        f"**MuJoCo Status:** {'✅ Available — full 3D physics rendering' if HAS_MUJOCO else '⚠️ Not installed — using procedural fallback'}"
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[t["task_id"] for t in list_mujoco_tasks()],
            value="easy_finger_reach",
            label="Simulation Task",
        )
        seed_input = gr.Number(value=7, precision=0, label="Seed")
        reset_btn = gr.Button("🔄 Reset Simulation", variant="primary")
        auto_btn = gr.Button("▶️ Run Auto Policy", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            sim_frame = gr.Image(label="3D Simulation View", type="filepath")
            mujoco_status = gr.Markdown(value="Reset a task to start the 3D simulation.")
        with gr.Column(scale=1):
            task_desc = gr.Markdown(value=TASK_DESCRIPTIONS.get("easy_finger_reach", ""))
            obs_json = gr.JSON(label="Observation")
            with gr.Row():
                reward_display = gr.Number(label="Reward", value=0.0, precision=3)
                done_display = gr.Checkbox(label="Done", value=False)
                step_display = gr.Number(label="Steps", value=0, precision=0)

    gr.Markdown("### Muscle Activation Controls")
    gr.Markdown(
        "Adjust muscle activations [0, 1] to control the musculoskeletal model. "
        "For MyoFinger: extn, adabR, adabL, mflx, dflx. "
        "For MyoHand: 39 muscles (first 5 shown; rest default to 0)."
    )

    with gr.Row():
        m0 = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Muscle 0 (extn/ECRL)")
        m1 = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Muscle 1 (adabR/ECRB)")
        m2 = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Muscle 2 (adabL/ECU)")
        m3 = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Muscle 3 (mflx/FCR)")
        m4 = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Muscle 4 (dflx/FCU)")

    step_btn = gr.Button("⚡ Apply & Step", variant="secondary")

    # Wire up events
    reset_outputs = [
        task_state, obs_json, sim_frame, task_desc, mujoco_status,
        reward_display, done_display, step_display,
    ]

    reset_btn.click(
        _reset_mujoco_task,
        inputs=[task_dropdown, seed_input],
        outputs=reset_outputs,
        api_name=False,
    )

    step_outputs = [
        task_state, obs_json, sim_frame,
        reward_display, done_display, step_display,
    ]

    step_btn.click(
        _step_mujoco_task,
        inputs=[task_state, m0, m1, m2, m3, m4],
        outputs=step_outputs,
        api_name=False,
    )

    auto_outputs = [
        task_state, obs_json, sim_frame,
        reward_display, done_display, step_display,
    ]

    auto_btn.click(
        _run_auto_policy,
        inputs=[task_state],
        outputs=auto_outputs,
        api_name=False,
    )

    return task_state
