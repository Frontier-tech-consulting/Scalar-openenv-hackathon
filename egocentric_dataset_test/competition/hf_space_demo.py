from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
import numpy as np
import gradio as gr
import spaces as hf_spaces
from PIL import Image

from egocentric_dataset_test.competition.demo import _resolve_first_episode, _video_label, _worker_choices
from egocentric_dataset_test.competition.environment import EgocentricFactoryAction, EgocentricFactoryCompetitionEnv, list_task_specs
from egocentric_dataset_test.competition.mujoco_sim import create_mujoco_task
from egocentric_dataset_test.competition.openenv_myosim_adapter import (
    OPENENV_TO_MYOSIM_TASK_MAP,
    OpenEnvToMyoSimAdapter,
)
from egocentric_dataset_test.competition.real_pipeline import _prepare_clip_context, empty_plot_df
from egocentric_dataset_test.competition.s3_rl_bridge import S3DatasetClient
from egocentric_dataset_test.competition.hf_space_workflows import (
    DEFAULT_GRPO_MODEL,
    build_segmentation_assets,
    run_openenv_grpo_training,
    run_openenv_sdk_validation,
)


OPENENV_OUTPUT_S3_URI = os.getenv("OPENENV_OUTPUT_S3_URI", "").strip()
OPENENV_OUTPUT_PREFIX = os.getenv("OPENENV_OUTPUT_PREFIX", "hf-space").strip("/") or "hf-space"
OPENENV_ANALYSIS_GPU_DURATION = int(os.getenv("OPENENV_ANALYSIS_GPU_DURATION", "90"))

def _gpu(duration: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if hf_spaces is None:
            return func
        return hf_spaces.GPU(duration=duration)(func)

    return decorator


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    bucket_path = s3_uri.replace("s3://", "", 1)
    bucket, _, prefix = bucket_path.partition("/")
    return bucket, prefix.strip("/")


def _save_json(payload: dict[str, Any], suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Path(tmp.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tmp.name


def _save_png(image: np.ndarray, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Image.fromarray(image).save(tmp.name)
    return tmp.name


def _maybe_upload_artifact(local_path: str | None, artifact_name: str, content_type: str) -> str | None:
    if not local_path or not OPENENV_OUTPUT_S3_URI:
        return None
    path = Path(local_path)
    if not path.exists():
        return None
    try:
        dataset_client = S3DatasetClient(s3_uri=OPENENV_OUTPUT_S3_URI)
        s3_client = dataset_client._get_client()
        if s3_client is None:
            return None
        bucket, prefix = _parse_s3_uri(OPENENV_OUTPUT_S3_URI)
        key_prefix = f"{prefix}/{OPENENV_OUTPUT_PREFIX}" if prefix else OPENENV_OUTPUT_PREFIX
        object_key = f"{key_prefix}/{artifact_name.lstrip('/')}"
        s3_client.upload_file(str(path), bucket, object_key, ExtraArgs={"ContentType": content_type})
        return f"s3://{bucket}/{object_key}"
    except Exception:
        return None


@_gpu(duration=OPENENV_ANALYSIS_GPU_DURATION)
def analyze_worker_vision(worker_id: str) -> tuple[str, str | None, str]:
    episode = _resolve_first_episode(worker_id)
    if episode is None:
        return (
            "### Vision analysis unavailable\n\nNo bundled episode is available for the selected worker.",
            None,
            json.dumps({"worker_id": worker_id, "available": False}, indent=2),
        )

    clip_metadata = episode.get("clip_metadata") if isinstance(episode.get("clip_metadata"), dict) else {}
    context = _prepare_clip_context(
        worker_id=worker_id,
        video_label=_video_label(episode),
        clip_metadata=clip_metadata,
        episode=episode,
    )
    detections = context.get("detections", [])
    target_3d = np.asarray(context.get("target_3d", [0.0, 0.0, 0.0]), dtype=float).tolist()
    summary = {
        "worker_id": worker_id,
        "video_label": _video_label(episode),
        "detections": detections,
        "num_detections": len(detections),
        "target_3d": target_3d,
        "landmark_source": context.get("mp_source"),
        "research_note": {
            "reconstruction_readiness": "good for scene proxy reconstruction, weak for exact CAD-grade object geometry from a single worker video",
            "best_path": "segmentation + camera pose estimation + gaussian splatting or photogrammetry, then export simplified mesh/USD for simulation",
        },
    }
    summary_path = _save_json(summary, suffix="_vision_summary.json")
    preview_path = str(context.get("preview_path") or "") or None
    preview_uri = _maybe_upload_artifact(preview_path, f"vision/{worker_id}/preview.png", "image/png")
    summary_uri = _maybe_upload_artifact(summary_path, f"vision/{worker_id}/summary.json", "application/json")
    detection_lines = [
        f"- **Detection:** `{item.get('cls_name', 'object')}` @ `{float(item.get('conf', 0.0)):.2f}`"
        for item in detections[:5]
    ]
    upload_lines = [
        line
        for line in [
            f"- **S3 preview:** `{preview_uri}`" if preview_uri else "",
            f"- **S3 summary:** `{summary_uri}`" if summary_uri else "",
        ]
        if line
    ]
    markdown = "\n".join(
        [
            "### ZeroGPU Vision Analysis",
            "",
            f"- **Worker:** `{worker_id}`",
            f"- **Video:** `{summary['video_label']}`",
            f"- **YOLO detections:** `{len(detections)}`",
            f"- **3D target estimate:** `X={target_3d[0]:.3f} Y={target_3d[1]:.3f} Z={target_3d[2]:.3f}`",
            f"- **Hand source:** `{context.get('mp_source', 'unknown')}`",
            "- **3D path:** `use this stage for scene understanding, segmentation, pose estimation, and export of proxy geometry to later RL/simulation stages`",
            *detection_lines,
            *upload_lines,
        ]
    )
    return markdown, preview_path, json.dumps(summary, indent=2)


def evaluate_openenv_to_myosim(task_id: str) -> tuple[str, str | None, str]:
    env = EgocentricFactoryCompetitionEnv(task_id=task_id, seed=7)
    openenv_trace: list[dict[str, Any]] = []
    final_observation = None
    try:
        observation = env.reset(task_id=task_id, seed=7)
        for _ in range(min(8, env.state.total_steps)):
            action_values = [float(value) for value in observation.action_hint]
            openenv_trace.append(
                {
                    "stage": observation.current_stage,
                    "stage_index": int(observation.stage_index),
                    "progress": float(observation.progress),
                    "reward": float(observation.reward or 0.0),
                    "action_hint": action_values,
                    "state_values": [float(value) for value in observation.state_values],
                }
            )
            final_observation = observation
            if observation.done:
                break
            observation = env.step(EgocentricFactoryAction(joint_targets=action_values))
        final_observation = observation
        state = env.state
    finally:
        env.close()

    mapped_task = OPENENV_TO_MYOSIM_TASK_MAP[task_id]
    adapter = OpenEnvToMyoSimAdapter.from_openenv(final_observation, state, openenv_trace)
    myosim_task = create_mujoco_task(mapped_task, seed=7)
    myosim_rewards: list[float] = []
    myosim_trace: list[dict[str, Any]] = []
    render_path: str | None = None
    try:
        adapter.configure_task(myosim_task)
        sim_obs = myosim_task.reset(seed=7)
        for _ in range(32):
            adapted_obs = adapter.adapt_myosim_observation(sim_obs)
            action = adapter.action(sim_obs, myosim_task.sim.n_actuators)
            sim_obs = myosim_task.step(action)
            myosim_rewards.append(float(sim_obs.get("reward", 0.0)))
            myosim_trace.append(
                {
                    "step_count": int(sim_obs.get("step_count", 0)),
                    "reward": float(sim_obs.get("reward", 0.0)),
                    "done": bool(sim_obs.get("done", False)),
                    "delta_to_target": [float(value) for value in adapted_obs["delta_to_target"]],
                    "action": [float(value) for value in action.tolist()],
                }
            )
            if sim_obs.get("done", False):
                break
        rendered = myosim_task.render()
        if rendered is not None:
            render_path = _save_png(rendered, suffix="_myosim_eval.png")
    finally:
        myosim_task.close()

    transfer_summary = {
        "openenv_task": task_id,
        "myosim_task": mapped_task,
        "openenv_final_state": {
            "progress": float(state.progress),
            "grader_score": float(state.grader_score),
            "success": bool(state.success),
            "current_stage": state.current_stage,
        },
        "adapter": adapter.summary(),
        "openenv_trace": openenv_trace,
        "myosim_trace": myosim_trace,
        "myosim_reward_mean": float(np.mean(myosim_rewards)) if myosim_rewards else 0.0,
        "myosim_reward_max": float(np.max(myosim_rewards)) if myosim_rewards else 0.0,
        "transfer_assessment": {
            "status": "adapter_applied",
            "note": "OpenEnv stage targets now retarget the MyoSim task and condition actuator outputs through an explicit observation/action adapter rather than a generic fallback controller.",
        },
    }
    summary_path = _save_json(transfer_summary, suffix="_transfer_eval.json")
    render_uri = _maybe_upload_artifact(render_path, f"transfer/{task_id}/myosim_render.png", "image/png")
    summary_uri = _maybe_upload_artifact(summary_path, f"transfer/{task_id}/summary.json", "application/json")
    upload_lines = [
        line
        for line in [
            f"- **S3 render:** `{render_uri}`" if render_uri else "",
            f"- **S3 summary:** `{summary_uri}`" if summary_uri else "",
        ]
        if line
    ]
    markdown = "\n".join(
        [
            "### OpenEnv → MyoSim Evaluation",
            "",
            f"- **OpenEnv task:** `{task_id}`",
            f"- **Mapped MyoSim task:** `{mapped_task}`",
            f"- **OpenEnv progress / score:** `{float(state.progress):.2f}` / `{float(state.grader_score):.3f}`",
            f"- **OpenEnv success:** `{bool(state.success)}`",
            f"- **Adapter target:** `X={adapter.target_position[0]:.3f} Y={adapter.target_position[1]:.3f} Z={adapter.target_position[2]:.3f}`",
            f"- **Adapter priors:** `grip={adapter.grip_prior:.2f} wrist={adapter.wrist_roll_prior:.2f}`",
            f"- **MyoSim mean / max reward:** `{transfer_summary['myosim_reward_mean']:.3f}` / `{transfer_summary['myosim_reward_max']:.3f}`",
            "- **Transfer verdict:** `curriculum and stage hints are now translated into MyoSim retargeting plus actuator conditioning through a concrete adapter layer`",
            *upload_lines,
        ]
    )
    return markdown, render_path, json.dumps(transfer_summary, indent=2)


def create_hf_space_demo() -> Any:
    if gr is None:
        raise RuntimeError("Gradio is required to create the Hugging Face Space demo")

    task_choices = [spec.task_id for spec in list_task_specs()]
    default_worker = _worker_choices()[0]

    with gr.Blocks(title="Egocentric Factory ZeroGPU + OpenEnv Split", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Egocentric Factory Split Pipeline\n\n"
            "This Gradio SDK app now exposes the two production-facing pages requested for the OpenEnv workflow: "
            "(1) full cached worker-video segmentation + MuJoCo / MyoSim proxy asset export, and "
            "(2) strict OpenEnv SDK validation plus 3D MyoSim-coupled GRPO training hooks using the local competition environment."
        )

        with gr.Tab("Full Video Segmentation → MuJoCo Assets"):
            gr.Markdown(
                "Processes every cached worker clip for the selected worker, runs multi-object segmentation over the full clip timeline, "
                "then exports MuJoCo / MyoSim-ready proxy assets as `OBJ + USDA + MJCF + manifest + full-frame overlay archive`."
            )
            with gr.Row():
                worker_dropdown = gr.Dropdown(choices=_worker_choices(), value=default_worker, label="Worker")
                frame_stride = gr.Slider(minimum=1, maximum=24, step=1, value=6, label="Frame stride (1 = every cached frame)")
                analyze_button = gr.Button("Process Full Clips + Build Assets", variant="primary")
            segmentation_md = gr.Markdown()
            with gr.Row():
                segmentation_image = gr.Image(label="Segmentation contact sheet", type="filepath")
                segmentation_plot = gr.LinePlot(
                    value=empty_plot_df(),
                    x="step",
                    y="value",
                    color="metric",
                    title="Segmentation diagnostics",
                )
            segmentation_files = gr.File(label="Exported assets", file_count="multiple")
            segmentation_json = gr.Code(label="Segmentation asset manifest", language="json")
            analyze_button.click(
                build_segmentation_assets,
                inputs=[worker_dropdown, frame_stride],
                outputs=[segmentation_md, segmentation_image, segmentation_files, segmentation_json, segmentation_plot],
                api_name=False,
            )

        with gr.Tab("OpenEnv RL + 3D MyoSim GRPO"):
            gr.Markdown(
                "Runs both requested RL paths: a strict OpenEnv `reset → step → state` validation loop with reward plots, "
                "and a TRL `GRPOTrainer` entrypoint wired through `environment_factory` against the competition environment while stepping the mapped 3D MyoSim task."
            )
            with gr.Row():
                task_dropdown = gr.Dropdown(choices=task_choices, value=task_choices[0], label="OpenEnv task")
                max_steps = gr.Slider(minimum=4, maximum=24, step=1, value=12, label="Validation steps")
            with gr.Row():
                validate_button = gr.Button("Validate SDK Loop", variant="secondary")
                grpo_button = gr.Button("Run 3D MyoSim GRPO Training", variant="primary")
            with gr.Row():
                model_name = gr.Textbox(label="GRPO model", value=DEFAULT_GRPO_MODEL)
                prompt_count = gr.Slider(minimum=2, maximum=8, step=1, value=4, label="GRPO prompt count")

            validation_md = gr.Markdown()
            validation_plot = gr.LinePlot(
                value=empty_plot_df(),
                x="step",
                y="value",
                color="metric",
                title="OpenEnv validation rewards",
            )
            validation_json = gr.Code(label="Validation trace JSON", language="json")
            validate_button.click(
                run_openenv_sdk_validation,
                inputs=[task_dropdown, max_steps],
                outputs=[validation_md, validation_plot, validation_json],
                api_name=False,
            )

            grpo_md = gr.Markdown()
            grpo_plot = gr.LinePlot(
                value=empty_plot_df(),
                x="step",
                y="value",
                color="metric",
                title="GRPO training metrics",
            )
            grpo_json = gr.Code(label="GRPO run JSON", language="json")
            grpo_button.click(
                run_openenv_grpo_training,
                inputs=[task_dropdown, model_name, prompt_count],
                outputs=[grpo_md, grpo_plot, grpo_json],
                api_name=False,
            )

    demo.queue(default_concurrency_limit=1)
    return demo