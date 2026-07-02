from pathlib import Path

from egocentric_dataset_test.competition.real_preview import load_bundled_preview
from egocentric_dataset_test.competition.real_pipeline import stream_full_pipeline

preview = load_bundled_preview()
episode = preview["prepared_episodes"][0]
video_label = (
    f"Video {episode['clip_metadata']['clip_number']} · "
    f"{episode['clip_metadata']['duration_sec']:.1f}s · "
    f"{episode['clip_metadata']['fps']:.0f} fps"
)

stream = stream_full_pipeline(
    worker_id=episode["clip_metadata"]["worker_id"],
    video_label=video_label,
    clip_metadata=episode["clip_metadata"],
    episode=episode,
    total_timesteps=128,
)

chunk_path = None
for idx, update in enumerate(stream, start=1):
    markdown, reward_plot, telemetry_plot, preview_video_path, video_path = update
    print(idx, markdown.splitlines()[0], bool(reward_plot is not None), bool(telemetry_plot is not None), bool(preview_video_path), bool(video_path))
    if video_path:
        chunk_path = video_path
        print("chunk_or_video_path", chunk_path)
        break
    if idx > 20:
        break

if chunk_path is None:
    raise SystemExit("No chunk emitted in first 20 updates")

path = Path(chunk_path)
print("exists", path.exists(), path.stat().st_size if path.exists() else 0)
