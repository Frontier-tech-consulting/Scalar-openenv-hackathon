from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

from egocentric_dataset_test.competition.environment import EgocentricFactoryAction, EgocentricFactoryCompetitionEnv
from egocentric_dataset_test.competition.real_preview import (
    build_dataset_markdown,
    build_episode_rows,
    build_live_s3_markdown,
    build_metric_rows,
    build_training_markdown,
    inspect_live_s3,
    load_bundled_preview,
)
from egocentric_dataset_test.competition.mujoco_viz import create_mujoco_tab
from egocentric_dataset_test.competition.mujoco_sim import HAS_MUJOCO
from egocentric_dataset_test.competition.s3_rl_bridge import (
    EGOCENTRIC_S3_URI,
    EGOCENTRIC_STATS,
    EGOCENTRIC_INTRINSICS,
    S3RLEnvironmentBridge,
    S3TrainingRenderer,
    create_s3_rl_bridge,
)
from egocentric_dataset_test.competition.tasks import CompetitionTaskSpec, list_task_specs


TASK_SPECS = {spec.task_id: spec for spec in list_task_specs()}
REAL_PREVIEW = load_bundled_preview()


def _task_markdown(spec: CompetitionTaskSpec) -> str:
    stage_lines = "\n".join(
        f"- **{index + 1}. {stage.name}** — {stage.guidance} Target: `{list(stage.target)}`"
        for index, stage in enumerate(spec.stages)
    )
    return (
        f"## {spec.title}\n\n"
        f"**Task ID:** `{spec.task_id}`  \n"
        f"**Difficulty:** `{spec.difficulty}`  \n"
        f"**Objective:** {spec.objective}\n\n"
        f"**Real dataset grounding:** This task is distilled from the Egocentric-100K + MyoSuite/MuJoCo assembly pipeline "
        f"into a deterministic `surrogate_myo` runtime suitable for Hugging Face Spaces and OpenEnv judging.\n\n"
        f"**Grader:** {spec.grader_description}\n\n"
        f"### Stage sequence\n{stage_lines}"
    )


def _status_markdown(observation: dict[str, Any], state: dict[str, Any]) -> str:
    grader = state.get("grader_breakdown", {})
    reward = float(observation.get("reward") or 0.0)
    return (
        f"### Runtime Status\n\n"
        f"- **Episode:** `{state.get('episode_id', 'n/a')}`\n"
        f"- **Current stage:** `{observation.get('current_stage', 'n/a')}`\n"
        f"- **Guidance:** {observation.get('stage_guidance', 'n/a')}\n"
        f"- **Backend:** `{state.get('backend_mode', 'n/a')}` on shard `{state.get('shard_id', 'n/a')}`\n"
        f"- **Progress:** `{observation.get('progress', 0.0):.2f}`\n"
        f"- **Reward:** `{reward:.3f}`\n"
        f"- **Score:** `{float(state.get('grader_score', 0.0)):.3f}`\n"
        f"- **Success:** `{state.get('success', False)}`\n"
        f"- **Precision / Efficiency / Smoothness:** `{grader.get('precision', 0.0)}` / `{grader.get('efficiency', 0.0)}` / `{grader.get('smoothness', 0.0)}`\n"
    )


def _history_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No actions taken yet. Click **Reset Episode** to start the demo."
    lines = ["| Step | Action | Reward | Stage | Done |", "|---:|---|---:|---|---|"]
    for row in rows[-12:]:
        lines.append(
            f"| {row['step']} | `{row['action']}` | {row['reward']:.3f} | {row['stage']} | {row['done']} |"
        )
    return "\n".join(lines)


def _serialize_env(env: EgocentricFactoryCompetitionEnv, observation: Any, history: list[dict[str, Any]]) -> list[Any]:
    state = env.state.model_dump(mode="json")
    observation_payload = observation.model_dump(mode="json")
    hint = observation_payload["action_hint"]
    return [
        env,
        history,
        hint[0],
        hint[1],
        hint[2],
        hint[3],
        _task_markdown(TASK_SPECS[observation_payload["task_id"]]),
        _status_markdown(observation_payload, state),
        observation_payload,
        state,
        _history_markdown(history),
    ]


def reset_episode(task_id: str, seed: float) -> list[Any]:
    resolved_seed = int(seed)
    env = EgocentricFactoryCompetitionEnv(task_id=task_id, seed=resolved_seed)
    observation = env.reset(task_id=task_id, seed=resolved_seed)
    return _serialize_env(env, observation, [])


def step_episode(
    env: EgocentricFactoryCompetitionEnv | None,
    history: list[dict[str, Any]] | None,
    reach_x: float,
    reach_y: float,
    grip_force: float,
    wrist_roll: float,
) -> list[Any]:
    if env is None:
        return reset_episode(next(iter(TASK_SPECS)), 7)
    action_values = [float(reach_x), float(reach_y), float(grip_force), float(wrist_roll)]
    observation = env.step(EgocentricFactoryAction(joint_targets=action_values))
    step_index = int(observation.step_idx)
    history = list(history or [])
    history.append(
        {
            "step": step_index,
            "action": action_values,
            "reward": float(observation.reward or 0.0),
            "stage": observation.current_stage,
            "done": bool(observation.done),
        }
    )
    return _serialize_env(env, observation, history)


def run_suggested_policy(
    env: EgocentricFactoryCompetitionEnv | None,
    history: list[dict[str, Any]] | None,
) -> list[Any]:
    if env is None:
        return reset_episode(next(iter(TASK_SPECS)), 7)
    history = list(history or [])
    observation = env.state
    while not env.state.success and env.state.step_count < env.state.total_steps:
        current_observation = env._build_observation(reward=0.0, done=env.state.success)  # type: ignore[attr-defined]
        action_values = [float(value) for value in current_observation.action_hint]
        stepped = env.step(EgocentricFactoryAction(joint_targets=action_values))
        history.append(
            {
                "step": int(stepped.step_idx),
                "action": action_values,
                "reward": float(stepped.reward or 0.0),
                "stage": stepped.current_stage,
                "done": bool(stepped.done),
            }
        )
        if stepped.done:
            return _serialize_env(env, stepped, history)
    final_observation = env._build_observation(reward=0.0, done=env.state.success)  # type: ignore[attr-defined]
    return _serialize_env(env, final_observation, history)


def use_suggested_action(env: EgocentricFactoryCompetitionEnv | None) -> list[float]:
    if env is None:
        return [0.0, 0.0, 0.0, 0.0]
    observation = env._build_observation(reward=0.0, done=env.state.success)  # type: ignore[attr-defined]
    return [float(value) for value in observation.action_hint]


def _create_s3_rl_tab() -> Any:
    """Create the S3-Backed RL Environment tab for the Gradio UI."""
    if gr is None:
        return None

    s3_bridge_state = gr.State(value=None)

    gr.Markdown(
        "## S3-Backed Mini RL Environment\n\n"
        "Real-time RL training environment grounded in the **AWS S3 Egocentric-100K** dataset "
        "(`s3://buildai-egocentric/raw`). This panel connects the mini RL environment to the "
        "remote S3 architecture, rendering training episodes from real factory worker data.\n\n"
        f"**Dataset:** Egocentric-100K — {EGOCENTRIC_STATS['total_hours']:,} hours, "
        f"{EGOCENTRIC_STATS['video_clips']:,} clips, {EGOCENTRIC_STATS['storage_size_tb']} TB\n\n"
        f"**Camera:** {EGOCENTRIC_STATS['camera_type']} — "
        f"{EGOCENTRIC_INTRINSICS['image_width']}×{EGOCENTRIC_INTRINSICS['image_height']} "
        f"@ {EGOCENTRIC_STATS['fps']}fps (fisheye, Kannala-Brandt)"
    )

    with gr.Row():
        s3_uri_input = gr.Textbox(
            label="S3 URI",
            value=EGOCENTRIC_S3_URI,
            placeholder="s3://buildai-egocentric/raw",
        )
        max_episodes = gr.Slider(1, 20, value=5, step=1, label="Max Episodes")
        inspect_s3_btn = gr.Button("🔍 Inspect S3", variant="primary")
        run_rl_btn = gr.Button("▶️ Run RL Episodes", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            s3_status_img = gr.Image(label="S3 Data Source Status")
            reward_curve_img = gr.Image(label="RL Reward Curves")
            episode_progress_img = gr.Image(label="Episode Progress")
        with gr.Column(scale=1):
            s3_status_md = gr.Markdown(value="Click **Inspect S3** to check the connection.")
            rl_summary_json = gr.JSON(label="RL Training Summary")
            s3_episodes_json = gr.JSON(label="S3 Episodes")

    # --- Callbacks ---
    def _inspect_s3(s3_uri: str):
        bridge = S3RLEnvironmentBridge(s3_uri=s3_uri)
        result = bridge.inspect_s3(max_keys=50)
        renderer = S3TrainingRenderer(bridge=bridge)
        status_img = renderer.render_s3_status_panel()
        status_path = _save_frame(status_img)
        if result.available:
            status_md = (
                f"### ✅ S3 Connected\n\n"
                f"- **Bucket:** `{result.bucket}`\n"
                f"- **Prefix:** `{result.prefix}`\n"
                f"- **Objects found:** `{result.total_objects}`\n"
                f"- **Episodes parsed:** `{len(result.episodes)}`\n"
                f"- **Latency:** `{result.elapsed_ms:.0f}ms`"
            )
        else:
            status_md = (
                f"### ⚠️ S3 Fallback\n\n"
                f"- **Reason:** {result.error}\n"
                f"- **Using:** Bundled checkpoint data from `egocentric_dataset_test/checkpoints/ego_openenv`"
            )
        return [bridge, status_path, status_md, result.to_dict()]

    def _run_rl_episodes(bridge, s3_uri: str, max_ep: float):
        if bridge is None:
            bridge = S3RLEnvironmentBridge(s3_uri=s3_uri)
        episodes = bridge.build_rl_episodes(max_episodes=int(max_ep))
        renderer = S3TrainingRenderer(bridge=bridge)

        # Render visualizations
        reward_img = renderer.render_reward_curve(episodes)
        progress_img = renderer.render_episode_progress(episodes)

        reward_path = _save_frame(reward_img)
        progress_path = _save_frame(progress_img)

        summary = bridge.get_training_summary()
        episodes_data = [ep.to_dict() for ep in episodes]

        return [bridge, reward_path, progress_path, summary, episodes_data]

    inspect_s3_btn.click(
        _inspect_s3,
        inputs=[s3_uri_input],
        outputs=[s3_bridge_state, s3_status_img, s3_status_md, s3_episodes_json],
        api_name=False,
    )

    run_rl_btn.click(
        _run_rl_episodes,
        inputs=[s3_bridge_state, s3_uri_input, max_episodes],
        outputs=[s3_bridge_state, reward_curve_img, episode_progress_img, rl_summary_json, s3_episodes_json],
        api_name=False,
    )

    return s3_bridge_state


def _save_frame(frame: np.ndarray | None) -> str | None:
    """Save a numpy frame to a temp PNG file for Gradio Image component."""
    if frame is None:
        return None
    try:
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(frame).save(tmp.name)
        return tmp.name
    except Exception:
        return None


def create_demo() -> Any:
    if gr is None:
        return None

    with gr.Blocks(title="Egocentric Factory Competition Environment", analytics_enabled=False) as demo:
        env_state = gr.State(value=None)
        history_state = gr.State(value=[])

        gr.Markdown(
            "# Egocentric Factory Competition Environment\n\n"
            "Interactive OpenEnv + real-dataset preview + **MuJoCo 3D simulation** for the `surrogate_myo` runtime. "
            "This Space exposes four layers: the official OpenEnv API, the competition-safe interactive control demo, "
            "a real Egocentric-100K S3-grounded RL preview, and a live **MyoFinger/MyoHand musculoskeletal simulation** "
            "powered by MuJoCo physics and MyoSuite models (MyoHub/myo_sim)."
        )

        with gr.Tabs():
            with gr.Tab("OpenEnv Control"):
                with gr.Row():
                    task_id = gr.Dropdown(
                        choices=[spec.task_id for spec in list_task_specs()],
                        value=list_task_specs()[0].task_id,
                        label="Task",
                    )
                    seed = gr.Number(value=7, precision=0, label="Seed")
                    reset_btn = gr.Button("Reset Episode", variant="primary")
                    hint_btn = gr.Button("Load Suggested Action")
                    auto_btn = gr.Button("Run Suggested Policy")

                with gr.Row():
                    reach_x = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="reach_x")
                    reach_y = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="reach_y")
                    grip_force = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="grip_force")
                    wrist_roll = gr.Slider(-1.0, 1.0, value=0.0, step=0.01, label="wrist_roll")
                    step_btn = gr.Button("Apply Action", variant="secondary")

                with gr.Row():
                    task_md = gr.Markdown(value=_task_markdown(list_task_specs()[0]))
                    status_md = gr.Markdown(value="Reset an episode to start the interactive demo.")

                with gr.Row():
                    observation_json = gr.JSON(label="Observation")
                    state_json = gr.JSON(label="State")

                history_md = gr.Markdown(value="No actions taken yet. Click **Reset Episode** to start the demo.")

                reset_outputs = [
                    env_state,
                    history_state,
                    reach_x,
                    reach_y,
                    grip_force,
                    wrist_roll,
                    task_md,
                    status_md,
                    observation_json,
                    state_json,
                    history_md,
                ]

                reset_btn.click(reset_episode, inputs=[task_id, seed], outputs=reset_outputs, api_name=False)
                step_btn.click(
                    step_episode,
                    inputs=[env_state, history_state, reach_x, reach_y, grip_force, wrist_roll],
                    outputs=reset_outputs,
                    api_name=False,
                )
                auto_btn.click(run_suggested_policy, inputs=[env_state, history_state], outputs=reset_outputs, api_name=False)
                hint_btn.click(use_suggested_action, inputs=[env_state], outputs=[reach_x, reach_y, grip_force, wrist_roll], api_name=False)

            with gr.Tab("Real Dataset"):
                s3_uri_box = gr.Textbox(
                    label="Egocentric-100K S3 URI",
                    value=str(REAL_PREVIEW.get("dataset_reference", {}).get("source_uri", "s3://buildai-egocentric/raw")),
                )
                inspect_btn = gr.Button("Inspect Live S3 (if creds are set)")
                dataset_md = gr.Markdown(value=build_dataset_markdown(REAL_PREVIEW))
                live_s3_md = gr.Markdown(
                    value=build_live_s3_markdown(
                        str(REAL_PREVIEW.get("dataset_reference", {}).get("source_uri", "s3://buildai-egocentric/raw")),
                        {"available": False, "error": "No live inspection yet.", "rows": []},
                    )
                )
                episodes_df = gr.Dataframe(
                    headers=["factory_id", "worker_id", "clip_number", "duration_sec", "fps", "source_uri"],
                    value=build_episode_rows(REAL_PREVIEW),
                    label="Prepared Egocentric-100K episodes",
                    interactive=False,
                )
                dataset_json = gr.JSON(value=REAL_PREVIEW.get("dataset_reference", {}), label="Dataset reference")

            with gr.Tab("RL Training Preview"):
                training_md = gr.Markdown(value=build_training_markdown(REAL_PREVIEW))
                metrics_df = gr.Dataframe(
                    headers=["phase", "epoch", "offline_loss", "episode_reward", "policy_loss", "value_loss", "entropy"],
                    value=build_metric_rows(REAL_PREVIEW),
                    label="Policy-gradient training metrics",
                    interactive=False,
                )
                summary_json = gr.JSON(value=REAL_PREVIEW.get("training_summary", {}), label="Training summary")
                with gr.Row():
                    training_curves = gr.Image(value=REAL_PREVIEW.get("training_curves"), label="Training curves")
                    batch_diagnostics = gr.Image(value=REAL_PREVIEW.get("batch_diagnostics"), label="Batch diagnostics")
                with gr.Row():
                    rollout_diagnostics = gr.Image(value=REAL_PREVIEW.get("rollout_diagnostics"), label="Rollout diagnostics")
                    rollout_video = gr.Video(value=REAL_PREVIEW.get("rollout_video"), label="Inference rollout video")

            with gr.Tab("S3 RL Environment"):
                _create_s3_rl_tab()

            with gr.Tab("MuJoCo 3D Simulation"):
                create_mujoco_tab()

        def _refresh_live_s3(s3_uri: str):
            inspection = inspect_live_s3(s3_uri)
            return build_live_s3_markdown(s3_uri, inspection)

        inspect_btn.click(_refresh_live_s3, inputs=[s3_uri_box], outputs=[live_s3_md], api_name=False)

    return demo