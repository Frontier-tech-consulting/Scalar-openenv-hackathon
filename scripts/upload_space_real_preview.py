from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Dhruv/egocentric-factory-competition-env"
FILES = [
    # Core config
    "Dockerfile",
    "README.md",
    "pyproject.toml",
    "requirements-space.txt",
    "uv.lock",
    "openenv.yaml",
    "inference.py",
    # Server
    "server/__init__.py",
    "server/app.py",
    # Competition package
    "egocentric_dataset_test/__init__.py",
    "egocentric_dataset_test/competition/__init__.py",
    "egocentric_dataset_test/competition/environment.py",
    "egocentric_dataset_test/competition/surrogate_backend.py",
    "egocentric_dataset_test/competition/server.py",
    "egocentric_dataset_test/competition/tasks.py",
    "egocentric_dataset_test/competition/shards.py",
    "egocentric_dataset_test/competition/demo.py",
    "egocentric_dataset_test/competition/real_preview.py",
    # MuJoCo simulation
    "egocentric_dataset_test/competition/mujoco_sim.py",
    "egocentric_dataset_test/competition/mujoco_viz.py",
    # S3 RL bridge
    "egocentric_dataset_test/competition/s3_rl_bridge.py",
    # Checkpoints / data
    "egocentric_dataset_test/checkpoints/ego_openenv/dataset_reference.json",
    "egocentric_dataset_test/checkpoints/ego_openenv/prepared_episodes.json",
    "egocentric_dataset_test/checkpoints/ego_openenv/task_spec.json",
    "egocentric_dataset_test/checkpoints/ego_openenv/training_summary.json",
    "egocentric_dataset_test/checkpoints/ego_openenv/training_metrics.jsonl",
    "egocentric_dataset_test/checkpoints/ego_openenv/training_curves.png",
    "egocentric_dataset_test/checkpoints/ego_openenv/batch_diagnostics.png",
    "egocentric_dataset_test/checkpoints/ego_openenv/rollout_diagnostics.png",
    "egocentric_dataset_test/checkpoints/ego_openenv/rollout_render.mp4",
    "egocentric_dataset_test/checkpoints/ego_openenv/egocentric_report.html",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    root = Path(__file__).resolve().parents[1]
    for relative_path in FILES:
        path = root / relative_path
        if not path.exists():
            print(f"⚠️  Skipping missing file: {relative_path}")
            continue
        url = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=relative_path,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Add S3 RL bridge + MuJoCo 3D simulation + Egocentric-100K integration",
        )
        print(f"✅ {relative_path} → {url}")


if __name__ == "__main__":
    main()
