from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import imageio
import pandas as pd
_HAS_PANDAS = True
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None  # type: ignore[assignment]

from egocentric_dataset_test.competition.real_pipeline import empty_plot_df, stream_full_pipeline
from egocentric_dataset_test.competition.real_preview import load_bundled_preview


REAL_PREVIEW = load_bundled_preview()
PREPARED_EPISODES = [
    episode for episode in REAL_PREVIEW.get("prepared_episodes", []) if isinstance(episode, dict)
]

DEMO_TASKS = {
    "easy_finger_reach": {
        "title": "MyoFinger Reach",
        "openenv_task": "easy_bin_pick",
        "focus": "finger alignment and contact warm-up",
    },
    "medium_hand_grasp": {
        "title": "MyoHand Grasp",
        "openenv_task": "medium_sort_and_place",
        "focus": "hand closure and PCB pickup stabilization",
    },
    "hard_precision_assembly": {
        "title": "MyoHand Precision Assembly",
        "openenv_task": "hard_precision_assembly",
        "focus": "fine wrist rotation and precision PCB placement",
    },
}
def _clip_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    clip = episode.get("clip_metadata")
    return clip if isinstance(clip, dict) else {}


def _worker_id(episode: dict[str, Any]) -> str:
    clip = _clip_metadata(episode)
    worker = clip.get("worker_id") or episode.get("metadata", {}).get("worker_id") or "worker_unknown"
    return str(worker)


def _factory_id(episode: dict[str, Any]) -> str:
    clip = _clip_metadata(episode)
    factory = clip.get("factory_id") or episode.get("metadata", {}).get("factory_id") or "factory_unknown"
    return str(factory)


def _video_sort_key(episode: dict[str, Any]) -> tuple[int, str]:
    clip = _clip_metadata(episode)
    clip_number = int(clip.get("clip_number", clip.get("video_index", 0)) or 0)
    return (clip_number, str(episode.get("source_uri", "")))


def _video_label(episode: dict[str, Any]) -> str:
    clip = _clip_metadata(episode)
    clip_number = int(clip.get("clip_number", clip.get("video_index", 0)) or 0)
    duration_sec = float(clip.get("duration_sec", 0.0) or 0.0)
    fps = float(clip.get("fps", 0.0) or 0.0)
    return f"Video {clip_number} · {duration_sec:.1f}s · {fps:.0f} fps"


def _episodes_for_worker(worker_id: str) -> list[dict[str, Any]]:
    matches = [episode for episode in PREPARED_EPISODES if _worker_id(episode) == worker_id]
    return sorted(matches, key=_video_sort_key)


def _worker_choices() -> list[str]:
    workers = sorted({_worker_id(episode) for episode in PREPARED_EPISODES})
    return workers or ["worker_001"]


def _resolve_first_episode(worker_id: str) -> dict[str, Any] | None:
    episodes = _episodes_for_worker(worker_id)
    return episodes[0] if episodes else None


def _worker_summary_markdown(worker_id: str) -> str:
    episodes = _episodes_for_worker(worker_id)
    if not episodes:
        return "### Worker Pipeline\n\nNo bundled episodes are available for the selected worker."
    lines = [
        "### Worker Pipeline\n",
        f"- **Worker:** `{worker_id}`",
        f"- **Factory:** `{_factory_id(episodes[0])}`",
        f"- **Videos queued:** `{len(episodes)}`",
        "- **Execution mode:** `all worker videos are chunked sequentially into one coherent training + render pass`",
    ]
    for episode in episodes[:6]:
        lines.append(f"- **Queued clip:** `{_video_label(episode)}`")
    if len(episodes) > 6:
        lines.append(f"- **Additional clips:** `+{len(episodes) - 6}`")
    return "\n".join(lines)


def _perception_markdown(episode: dict[str, Any] | None) -> str:
    if episode is None:
        return "### Perception + Simulation Flow\n\nNo clip selected."
    return (
        "### Perception + Simulation Flow\n\n"
        "- **Input:** all egocentric videos for the selected worker, processed one after another\n"
        "- **Detector:** local Ultralytics YOLO5-preferred weights with segmentation overlay on each chunked clip\n"
        "- **Alignment:** each clip conditions a MuJoCo / MyoSim training run before full side-by-side replay\n"
        "- **Observability:** live training curves, preview frames, and a final worker-level combined chart are produced\n"
        f"- **Clip-conditioned context:** `{episode.get('task_description', 'PCB manipulation sequence')}`"
    )


def _concat_plot_frames(plot_frames: list[Any]) -> Any:
    if not _HAS_PANDAS:
        return None
    frames = [frame for frame in plot_frames if frame is not None and not frame.empty]
    if not frames:
        return empty_plot_df()
    return pd.concat(frames, ignore_index=True)


def _concatenate_videos(video_paths: list[str]) -> str | None:
    valid_paths = [path for path in video_paths if path and Path(path).exists()]
    if not valid_paths:
        return None
    if len(valid_paths) == 1:
        return valid_paths[0]

    tmp = tempfile.NamedTemporaryFile(suffix="_worker_rollup.mp4", delete=False)
    output_path = tmp.name
    writer = None
    try:
        fps = 15.0
        first_reader = imageio.get_reader(valid_paths[0])
        try:
            meta = first_reader.get_meta_data()
            fps = float(meta.get("fps", 15.0) or 15.0)
        finally:
            first_reader.close()

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            quality=7,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        )
        for path in valid_paths:
            reader = imageio.get_reader(path)
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()
    except Exception:
        return valid_paths[-1]
    finally:
        if writer is not None:
            writer.close()
    return output_path


def _worker_progress_markdown(worker_id: str, clip_index: int, clip_total: int, clip_label: str, inner_markdown: str) -> str:
    return (
        "### Worker Training Rollup\n\n"
        f"- **Worker:** `{worker_id}`\n"
        f"- **Clip progress:** `{clip_index}/{clip_total}`\n"
        f"- **Current clip:** `{clip_label}`\n\n"
        f"{inner_markdown}"
    )


def _worker_complete_markdown(worker_id: str, episodes: list[dict[str, Any]], video_count: int) -> str:
    lines = [
        "### ✅ Worker Rollup Complete\n",
        f"- **Worker:** `{worker_id}`",
        f"- **Videos processed:** `{video_count}`",
        "- **Detector + segmentation:** `Ultralytics YOLO5-preferred weights + box-guided segmentation overlay`",
        "- **Simulation:** `MuJoCo / MyoSim side-by-side manipulation replay for every worker clip`",
        "- **Artifacts:** `combined charts plus a stitched full worker rollout video`",
    ]
    for episode in episodes[:6]:
        lines.append(f"- **Included clip:** `{_video_label(episode)}`")
    return "\n".join(lines)


def _start_worker_training_stream(worker_id: str):
    episodes = _episodes_for_worker(worker_id)
    if not episodes:
        yield (
            "### Training Rollup\n\nNo bundled episode matched the current selection.",
            empty_plot_df(),
            empty_plot_df(),
            None,
            REAL_PREVIEW.get("rollout_video"),
        )
        return

    yield (
        _worker_summary_markdown(worker_id),
        empty_plot_df(),
        empty_plot_df(),
        None,
        None,
    )

    final_reward_plots: list[Any] = []
    final_telemetry_plots: list[Any] = []
    final_video_paths: list[str] = []
    final_preview_video_path: str | None = None
    for clip_index, episode in enumerate(episodes, start=1):
        clip = _clip_metadata(episode)
        clip_number = int(clip.get("clip_number", clip.get("video_index", 0)) or 0)
        total_ts = 1024 if clip_index == 1 or clip_number == 0 else 768
        last_update: tuple[str, Any, Any, str | None, str | None] | None = None
        for training_md, reward_plot, telemetry_plot, preview_video_path, rollout_video_path in stream_full_pipeline(
            worker_id=_worker_id(episode),
            video_label=_video_label(episode),
            clip_metadata=clip,
            episode=episode,
            total_timesteps=total_ts,
            algorithm="ppo",
            preset="balanced",
        ):
            last_update = (training_md, reward_plot, telemetry_plot, preview_video_path, rollout_video_path)
            if preview_video_path:
                final_preview_video_path = preview_video_path
            yield (
                _worker_progress_markdown(worker_id, clip_index, len(episodes), _video_label(episode), training_md),
                reward_plot,
                telemetry_plot,
                preview_video_path,
                rollout_video_path,
            )
        if last_update is not None:
            if last_update[1] is not None:
                final_reward_plots.append(last_update[1])
            if last_update[2] is not None:
                final_telemetry_plots.append(last_update[2])
            if last_update[4]:
                final_video_paths.append(last_update[4])

    combined_reward_plot = _concat_plot_frames(final_reward_plots)
    combined_telemetry_plot = _concat_plot_frames(final_telemetry_plots)
    stitched_video_path = _concatenate_videos(final_video_paths) or REAL_PREVIEW.get("rollout_video")
    yield (
        _worker_complete_markdown(worker_id, episodes, len(final_video_paths)),
        combined_reward_plot,
        combined_telemetry_plot,
        final_preview_video_path or stitched_video_path,
        stitched_video_path,
    )
def create_demo() -> Any:
    if gr is None:
        return None

    workers = _worker_choices()
    default_worker = workers[0]
    default_episode = _resolve_first_episode(default_worker)

    with gr.Blocks(title="PCB MyoSim Training Rollup", analytics_enabled=False) as demo:
        gr.Markdown(
            "# PCB Worker Training Rollup\n\n"
            "Select a worker and launch a simple end-to-end pipeline. "
            "The demo automatically processes every bundled video for that worker, runs Ultralytics YOLO detection + segmentation, "
            "trains a MuJoCo / MyoSim manipulation policy per clip, and returns combined charts plus one stitched side-by-side rollout video."
        )

        with gr.Row():
            worker_dropdown = gr.Dropdown(choices=workers, value=default_worker, label="Worker")
            train_button = gr.Button("Process Worker Videos", variant="primary")

        with gr.Row():
            selected_clip_md = gr.Markdown(value=_worker_summary_markdown(default_worker))
            perception_md = gr.Markdown(value=_perception_markdown(default_episode))

        training_md = gr.Markdown(value="### Training Rollup\n\nPress **Process Worker Videos** to run the full worker-level segmentation, training, live plotting, and MuJoCo replay pipeline.")

        with gr.Row():
            reward_plot = gr.LinePlot(
                value=empty_plot_df(),
                x="step",
                y="value",
                color="metric",
                title="Reward Metrics",
            )
            telemetry_plot = gr.LinePlot(
                value=empty_plot_df(),
                x="step",
                y="value",
                color="metric",
                title="Training Telemetry",
            )

        with gr.Row():
            preview_video = gr.Video(
                label="MyoSim Preview",
                autoplay=True,
                streaming=True,
            )

        rollout_video = gr.Video(
            value=REAL_PREVIEW.get("rollout_video"),
            label="Saved Rollout Video",
            autoplay=True,
            streaming=True,
        )

        worker_dropdown.change(
            lambda worker_id: (_worker_summary_markdown(worker_id), _perception_markdown(_resolve_first_episode(worker_id))),
            inputs=[worker_dropdown],
            outputs=[selected_clip_md, perception_md],
            api_name=False,
        )
        train_button.click(
            _start_worker_training_stream,
            inputs=[worker_dropdown],
            outputs=[training_md, reward_plot, telemetry_plot, preview_video, rollout_video],
            api_name=False,
        )

    demo.queue(default_concurrency_limit=1)
    return demo