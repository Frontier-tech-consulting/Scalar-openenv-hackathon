"""
S3-Backed RL Environment Bridge for Egocentric-100K
====================================================

Connects the AWS S3-hosted Egocentric-100K dataset to the mini RL environment,
enabling the remote architecture to render RL training episodes grounded in
real factory worker data.

S3 URI: s3://buildai-egocentric/raw
HF Dataset: builddotai/Egocentric-100K
Format: WebDataset (tar shards with mp4+json pairs)
Structure: factory{NNN}/worker{NNN}/part{NNN}.tar

This module provides:
1. S3DatasetClient — Lists and fetches episode metadata from the S3 bucket
2. S3RLEnvironmentBridge — Bridges S3 episode data into the RL environment
3. S3TrainingRenderer — Renders RL training visualizations grounded in S3 data
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants from the Egocentric-100K dataset specification
# ─────────────────────────────────────────────────────────────────────────────

EGOCENTRIC_S3_URI = "s3://buildai-egocentric/raw"
EGOCENTRIC_S3_BUCKET = "buildai-egocentric"
EGOCENTRIC_S3_PREFIX = "raw"
EGOCENTRIC_HF_REPO_ID = "builddotai/Egocentric-100K"
EGOCENTRIC_HF_URL = "https://huggingface.co/datasets/builddotai/Egocentric-100K"

# Dataset statistics (from HF dataset card)
EGOCENTRIC_STATS = {
    "total_hours": 100405,
    "total_frames": 10_800_000_000,
    "video_clips": 2_010_759,
    "median_clip_length_sec": 180.0,
    "mean_hours_per_worker": 7.06,
    "storage_size_tb": 24.79,
    "format": "H.265/MP4",
    "resolution": "456x256",
    "fps": 30.0,
    "camera_type": "Monocular head-mounted fisheye",
    "device": "Build AI Gen 1",
}

# Camera intrinsics (Kannala-Brandt fisheye model)
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

# WebDataset structure: factory{NNN}/worker{NNN}/part{NNN}.tar
# Each tar contains pairs: factory{NNN}_worker{NNN}_{NNNNN}.mp4 + .json
EPISODE_JSON_SCHEMA = {
    "factory_id": "string",
    "worker_id": "string",
    "video_index": "int64",
    "duration_sec": "float64",
    "width": "int64",
    "height": "int64",
    "fps": "float64",
    "size_bytes": "int64",
    "codec": "string",
}


# ─────────────────────────────────────────────────────────────────────────────
# S3 Dataset Client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class S3EpisodeInfo:
    """Metadata for a single Egocentric-100K episode from S3."""
    factory_id: str
    worker_id: str
    video_index: int
    duration_sec: float
    width: int = 456
    height: int = 256
    fps: float = 30.0
    size_bytes: int = 0
    codec: str = "h265"
    s3_key: str = ""
    last_modified: str = ""

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 3)

    @property
    def clip_name(self) -> str:
        return f"{self.factory_id}_{self.worker_id}_{self.video_index:05d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factory_id": self.factory_id,
            "worker_id": self.worker_id,
            "video_index": self.video_index,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "size_bytes": self.size_bytes,
            "codec": self.codec,
            "s3_key": self.s3_key,
            "size_mb": self.size_mb,
            "clip_name": self.clip_name,
        }


@dataclass
class S3InspectionResult:
    """Result of an S3 bucket inspection."""
    available: bool = False
    bucket: str = EGOCENTRIC_S3_BUCKET
    prefix: str = EGOCENTRIC_S3_PREFIX
    episodes: list[S3EpisodeInfo] = field(default_factory=list)
    total_objects: int = 0
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "total_objects": self.total_objects,
            "episodes": [e.to_dict() for e in self.episodes],
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class S3DatasetClient:
    """Client for listing and fetching Egocentric-100K episode metadata from S3.

    Supports both direct AWS S3 access (with credentials) and Cloudflare R2
    mirrors. Falls back to bundled checkpoint data when credentials are absent.
    """

    def __init__(
        self,
        s3_uri: str = EGOCENTRIC_S3_URI,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        self.s3_uri = s3_uri
        self.bucket, _, self.prefix = s3_uri.replace("s3://", "").partition("/")
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL_S3")
        self.region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self._client = None

    @property
    def has_credentials(self) -> bool:
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
        return bool(access_key and secret_key)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError:
            return None
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN") or os.getenv("R2_SESSION_TOKEN")
        if not access_key or not secret_key:
            return None
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=self.region_name,
        )
        return self._client

    def inspect(self, max_keys: int = 50) -> S3InspectionResult:
        """List objects in the S3 bucket and extract episode metadata."""
        start = time.monotonic()
        client = self._get_client()
        if client is None:
            return S3InspectionResult(
                available=False,
                error="S3 client unavailable: missing boto3 or AWS credentials",
                elapsed_ms=(time.monotonic() - start) * 1000,
            )

        try:
            normalized_prefix = self.prefix.strip("/")
            response = client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=normalized_prefix,
                MaxKeys=max_keys,
            )
        except Exception as exc:
            return S3InspectionResult(
                available=False,
                error=f"S3 list_objects_v2 failed: {exc}",
                elapsed_ms=(time.monotonic() - start) * 1000,
            )

        episodes: list[S3EpisodeInfo] = []
        total_objects = 0
        json_keys: dict[str, dict] = {}

        for obj in response.get("Contents", []):
            key = obj.get("Key", "")
            total_objects += 1
            if key.endswith(".json"):
                # Try to fetch the JSON metadata
                try:
                    resp = client.get_object(Bucket=self.bucket, Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                    json_keys[key] = data
                except Exception:
                    json_keys[key] = {}
            elif key.endswith(".mp4"):
                # Parse from key pattern: factory{NNN}_worker{NNN}_{NNNNN}.mp4
                parts = Path(key).stem.split("_")
                if len(parts) >= 3:
                    try:
                        episodes.append(S3EpisodeInfo(
                            factory_id=parts[0],
                            worker_id=parts[1],
                            video_index=int(parts[2]),
                            duration_sec=0.0,
                            size_bytes=obj.get("Size", 0),
                            s3_key=key,
                            last_modified=str(obj.get("LastModified", "")),
                        ))
                    except (ValueError, IndexError):
                        pass

        # Enrich episodes with JSON metadata where available
        for ep in episodes:
            json_key = ep.s3_key.replace(".mp4", ".json")
            meta = json_keys.get(json_key, {})
            if meta:
                ep.factory_id = meta.get("factory_id", ep.factory_id)
                ep.worker_id = meta.get("worker_id", ep.worker_id)
                ep.video_index = meta.get("video_index", ep.video_index)
                ep.duration_sec = meta.get("duration_sec", ep.duration_sec)
                ep.width = meta.get("width", ep.width)
                ep.height = meta.get("height", ep.height)
                ep.fps = meta.get("fps", ep.fps)
                ep.size_bytes = meta.get("size_bytes", ep.size_bytes) or ep.size_bytes
                ep.codec = meta.get("codec", ep.codec)

        elapsed = (time.monotonic() - start) * 1000
        return S3InspectionResult(
            available=True,
            bucket=self.bucket,
            prefix=self.prefix,
            episodes=episodes,
            total_objects=total_objects,
            elapsed_ms=elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# S3-Backed RL Environment Bridge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class S3RLEpisode:
    """An RL training episode grounded in S3 Egocentric-100K data."""
    episode_id: str
    factory_id: str
    worker_id: str
    clip_number: int
    task_id: str
    duration_sec: float
    fps: float
    steps: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    cumulative_reward: float = 0.0
    s3_source_key: str = ""
    has_mujoco: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "factory_id": self.factory_id,
            "worker_id": self.worker_id,
            "clip_number": self.clip_number,
            "task_id": self.task_id,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "total_steps": len(self.steps),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "s3_source_key": self.s3_source_key,
            "has_mujoco": self.has_mujoco,
        }


class S3RLEnvironmentBridge:
    """Bridges S3 Egocentric-100K episode data into the mini RL environment.

    This bridge:
    1. Fetches episode metadata from S3 (or uses bundled checkpoint data)
    2. Maps S3 episodes to RL task configurations
    3. Runs the surrogate_myo backend with S3-grounded parameters
    4. Produces RL training visualizations grounded in real worker data
    """

    def __init__(self, s3_uri: str = EGOCENTRIC_S3_URI) -> None:
        self.s3_client = S3DatasetClient(s3_uri=s3_uri)
        self._episodes: list[S3RLEpisode] = []
        self._inspection: S3InspectionResult | None = None

    @property
    def s3_available(self) -> bool:
        return self.s3_client.has_credentials

    def inspect_s3(self, max_keys: int = 50) -> S3InspectionResult:
        """Inspect the S3 bucket and cache the result."""
        self._inspection = self.s3_client.inspect(max_keys=max_keys)
        return self._inspection

    def get_bundled_episodes(self) -> list[dict[str, Any]]:
        """Load bundled checkpoint episodes as fallback when S3 is unavailable."""
        try:
            from egocentric_dataset_test.competition.real_preview import load_bundled_preview
            preview = load_bundled_preview()
            return preview.get("prepared_episodes", [])
        except Exception:
            return []

    def build_rl_episodes(self, max_episodes: int = 10) -> list[S3RLEpisode]:
        """Build RL episodes from S3 data or bundled checkpoint fallback."""
        episodes: list[S3RLEpisode] = []

        # Try S3 first
        if self._inspection is None:
            self.inspect_s3()

        if self._inspection and self._inspection.available:
            for i, ep_info in enumerate(self._inspection.episodes[:max_episodes]):
                # Map S3 episode to RL task
                task_id = self._map_episode_to_task(ep_info)
                rl_ep = S3RLEpisode(
                    episode_id=f"s3-{ep_info.factory_id}-{ep_info.worker_id}-{i:04d}",
                    factory_id=ep_info.factory_id,
                    worker_id=ep_info.worker_id,
                    clip_number=ep_info.video_index,
                    task_id=task_id,
                    duration_sec=ep_info.duration_sec,
                    fps=ep_info.fps,
                    s3_source_key=ep_info.s3_key,
                    has_mujoco=self._check_mujoco(),
                )
                # Run a mini RL episode using the surrogate backend
                self._run_mini_episode(rl_ep)
                episodes.append(rl_ep)

        # Fallback to bundled data
        if not episodes:
            bundled = self.get_bundled_episodes()
            for i, ep_data in enumerate(bundled[:max_episodes]):
                clip = ep_data.get("clip_metadata", {})
                rl_ep = S3RLEpisode(
                    episode_id=f"bundled-{i:04d}",
                    factory_id=clip.get("factory_id", "factory_002"),
                    worker_id=clip.get("worker_id", "worker_002"),
                    clip_number=clip.get("clip_number", i),
                    task_id=ep_data.get("task_id", "easy_bin_pick"),
                    duration_sec=clip.get("duration_sec", 180.0),
                    fps=clip.get("fps", 30.0),
                    s3_source_key=ep_data.get("source_uri", EGOCENTRIC_S3_URI),
                    has_mujoco=self._check_mujoco(),
                )
                self._run_mini_episode(rl_ep)
                episodes.append(rl_ep)

        self._episodes = episodes
        return episodes

    def _map_episode_to_task(self, ep_info: S3EpisodeInfo) -> str:
        """Map an S3 episode to the closest RL task based on metadata."""
        # Use duration as a heuristic for task difficulty
        if ep_info.duration_sec < 60:
            return "easy_bin_pick"
        elif ep_info.duration_sec < 300:
            return "medium_sort_and_place"
        else:
            return "hard_precision_assembly"

    def _check_mujoco(self) -> bool:
        """Check if MuJoCo is available for 3D rendering."""
        try:
            from egocentric_dataset_test.competition.mujoco_sim import HAS_MUJOCO
            return HAS_MUJOCO
        except Exception:
            return False

    def _run_mini_episode(self, episode: S3RLEpisode, max_steps: int = 20) -> None:
        """Run a mini RL episode using the surrogate_myo backend."""
        try:
            from egocentric_dataset_test.competition.environment import (
                EgocentricFactoryAction,
                EgocentricFactoryCompetitionEnv,
            )
            env = EgocentricFactoryCompetitionEnv(task_id=episode.task_id, seed=7)
            obs = env.reset(task_id=episode.task_id, seed=7)

            for step_idx in range(max_steps):
                # Use the action hint as a deterministic policy
                hint = obs.action_hint if hasattr(obs, "action_hint") else [0.0] * 4
                action = EgocentricFactoryAction(joint_targets=hint)
                obs = env.step(action)
                reward = float(obs.reward or 0.0)
                episode.rewards.append(reward)
                episode.cumulative_reward += reward
                episode.steps.append({
                    "step": step_idx,
                    "reward": reward,
                    "progress": float(obs.progress),
                    "stage": obs.current_stage,
                    "done": obs.done,
                })
                if obs.done:
                    break
            env.close()
        except Exception as exc:
            # If the environment fails, record a minimal episode
            episode.steps.append({"step": 0, "error": str(exc), "reward": 0.0})
            episode.rewards.append(0.0)

    def get_training_summary(self) -> dict[str, Any]:
        """Get a summary of the S3-grounded RL training state."""
        episodes = self._episodes or self.build_rl_episodes()
        total_reward = sum(ep.cumulative_reward for ep in episodes)
        avg_reward = total_reward / max(len(episodes), 1)
        total_steps = sum(len(ep.steps) for ep in episodes)

        return {
            "s3_uri": self.s3_client.s3_uri,
            "s3_available": self.s3_available,
            "inspection": self._inspection.to_dict() if self._inspection else None,
            "total_episodes": len(episodes),
            "total_steps": total_steps,
            "total_reward": round(total_reward, 4),
            "avg_reward_per_episode": round(avg_reward, 4),
            "episodes": [ep.to_dict() for ep in episodes],
            "dataset_stats": EGOCENTRIC_STATS,
            "camera_intrinsics": EGOCENTRIC_INTRINSICS,
        }


# ─────────────────────────────────────────────────────────────────────────────
# S3 Training Renderer — Generates visualizations for the RL environment
# ─────────────────────────────────────────────────────────────────────────────

class S3TrainingRenderer:
    """Renders RL training visualizations grounded in S3 Egocentric-100K data.

    Produces:
    - Reward curves per episode
    - Episode progress heatmaps
    - S3 data source status panels
    - MuJoCo 3D simulation frames (when available)
    """

    def __init__(self, bridge: S3RLEnvironmentBridge | None = None) -> None:
        self.bridge = bridge or S3RLEnvironmentBridge()

    def render_reward_curve(self, episodes: list[S3RLEpisode] | None = None) -> np.ndarray:
        """Render a reward curve visualization as an RGB image."""
        episodes = episodes or self.bridge._episodes
        if not episodes:
            episodes = self.bridge.build_rl_episodes()

        # Create a 480x640 RGB image
        h, w = 480, 640
        img = np.ones((h, w, 3), dtype=np.uint8) * 30  # Dark background

        # Title
        self._draw_text(img, "S3-Grounded RL Training Rewards", 20, 15, color=(100, 200, 255))

        # S3 status
        s3_status = "S3: LIVE" if self.bridge.s3_available else "S3: BUNDLED FALLBACK"
        s3_color = (100, 255, 100) if self.bridge.s3_available else (255, 200, 100)
        self._draw_text(img, s3_status, 20, 40, color=s3_color, size=12)

        # Dataset info
        self._draw_text(img, f"Episodes: {len(episodes)} | Dataset: Egocentric-100K", 20, 58, color=(180, 180, 180), size=10)

        # Plot reward curves
        plot_x, plot_y = 60, 80
        plot_w, plot_h = w - 100, h - 140

        # Draw axes
        img[plot_y:plot_y + plot_h, plot_x:plot_x + 1] = [150, 150, 150]
        img[plot_y + plot_h:plot_y + plot_h + 1, plot_x:plot_x + plot_w] = [150, 150, 150]

        # Y-axis label
        self._draw_text(img, "Reward", plot_x - 45, plot_y + plot_h // 2, color=(150, 150, 150), size=9)
        # X-axis label
        self._draw_text(img, "Steps", plot_x + plot_w // 2, plot_y + plot_h + 15, color=(150, 150, 150), size=9)

        # Plot each episode's reward curve
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (150, 200, 100), (100, 150, 200),
            (200, 100, 150),
        ]

        max_steps = max((len(ep.rewards) for ep in episodes), default=1) or 1
        max_reward = max((max(ep.rewards) if ep.rewards else 1.0 for ep in episodes), default=1.0) or 1.0

        for i, ep in enumerate(episodes[:10]):
            if not ep.rewards:
                continue
            color = colors[i % len(colors)]
            points = []
            for j, r in enumerate(ep.rewards):
                x = plot_x + int(j / max_steps * plot_w)
                y = plot_y + plot_h - int(r / max_reward * plot_h)
                points.append((x, y))
            # Draw line segments
            for k in range(len(points) - 1):
                x1, y1 = points[k]
                x2, y2 = points[k + 1]
                # Simple line drawing
                for t in np.linspace(0, 1, max(abs(x2 - x1), abs(y2 - y1), 1) + 1):
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = color

            # Episode label
            label = f"Ep{i+1}: {ep.episode_id[:20]}"
            self._draw_text(img, label, plot_x + plot_w + 5, plot_y + i * 14, color=color, size=8)

        # Summary stats
        total_reward = sum(ep.cumulative_reward for ep in episodes)
        avg_reward = total_reward / max(len(episodes), 1)
        self._draw_text(img, f"Total: {total_reward:.2f} | Avg: {avg_reward:.3f}", 20, h - 25, color=(200, 200, 200), size=10)

        return img

    def render_s3_status_panel(self) -> np.ndarray:
        """Render an S3 data source status panel as an RGB image."""
        h, w = 360, 480
        img = np.ones((h, w, 3), dtype=np.uint8) * 25

        # Title
        self._draw_text(img, "AWS S3 Egocentric-100K Integration", 15, 15, color=(100, 200, 255))

        # S3 connection status
        s3_ok = self.bridge.s3_available
        status_text = "● CONNECTED" if s3_ok else "○ BUNDLED FALLBACK"
        status_color = (100, 255, 100) if s3_ok else (255, 200, 100)
        self._draw_text(img, f"S3 Status: {status_text}", 15, 40, color=status_color, size=12)

        # S3 URI
        self._draw_text(img, f"Bucket: s3://{self.bridge.s3_client.bucket}/{self.bridge.s3_client.prefix}", 15, 60, color=(180, 180, 180), size=10)

        # Dataset stats
        y = 85
        stats = EGOCENTRIC_STATS
        self._draw_text(img, "─── Dataset Statistics ───", 15, y, color=(150, 200, 255), size=10)
        y += 18
        for key, val in [
            ("Total Hours", f"{stats['total_hours']:,}"),
            ("Video Clips", f"{stats['video_clips']:,}"),
            ("Total Frames", f"{stats['total_frames']:,}"),
            ("Storage", f"{stats['storage_size_tb']} TB"),
            ("Format", stats["format"]),
            ("Resolution", stats["resolution"]),
            ("FPS", str(stats["fps"])),
            ("Camera", stats["camera_type"]),
        ]:
            self._draw_text(img, f"{key}: {val}", 25, y, color=(170, 170, 170), size=9)
            y += 15

        # Camera intrinsics
        y += 5
        self._draw_text(img, "─── Camera Intrinsics ───", 15, y, color=(150, 200, 255), size=10)
        y += 18
        intr = EGOCENTRIC_INTRINSICS
        for key in ["model", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]:
            self._draw_text(img, f"{key}: {intr[key]}", 25, y, color=(170, 170, 170), size=9)
            y += 14

        # HF dataset link
        y += 5
        self._draw_text(img, f"HF: {EGOCENTRIC_HF_REPO_ID}", 15, y, color=(100, 180, 255), size=9)

        return img

    def render_episode_progress(self, episodes: list[S3RLEpisode] | None = None) -> np.ndarray:
        """Render episode progress bars as an RGB image."""
        episodes = episodes or self.bridge._episodes
        if not episodes:
            episodes = self.bridge.build_rl_episodes()

        h, w = 300, 480
        img = np.ones((h, w, 3), dtype=np.uint8) * 25

        self._draw_text(img, "S3-Grounded Episode Progress", 15, 15, color=(100, 200, 255))

        y = 40
        bar_w = w - 120
        for i, ep in enumerate(episodes[:8]):
            # Episode label
            label = f"Ep{i+1} ({ep.task_id[:15]})"
            self._draw_text(img, label, 15, y, color=(180, 180, 180), size=9)

            # Progress bar background
            bar_x = 15
            bar_y = y + 12
            bar_h = 12
            img[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = [50, 50, 50]

            # Progress fill
            if ep.rewards:
                progress = min(ep.cumulative_reward / max(len(ep.rewards), 1), 1.0)
            else:
                progress = 0.0
            fill_w = int(progress * bar_w)
            if fill_w > 0:
                # Green gradient
                color = (int(50 + 150 * progress), int(200 * progress), 50)
                img[bar_y:bar_y + bar_h, bar_x:bar_x + fill_w] = color

            # Reward text
            self._draw_text(img, f"{ep.cumulative_reward:.2f}", bar_x + bar_w + 5, bar_y, color=(200, 200, 200), size=9)

            y += 30

        return img

    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = (255, 255, 255),
        size: int = 14,
    ) -> None:
        """Draw text on an image using PIL or a simple fallback."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (IOError, OSError):
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
                except (IOError, OSError):
                    font = ImageFont.load_default()
            draw.text((x, y), text, fill=color, font=font)
            img[:] = np.array(pil_img)
        except ImportError:
            # Fallback: just skip text rendering
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def create_s3_rl_bridge(s3_uri: str = EGOCENTRIC_S3_URI) -> S3RLEnvironmentBridge:
    """Create an S3 RL environment bridge."""
    return S3RLEnvironmentBridge(s3_uri=s3_uri)


def create_s3_training_renderer(s3_uri: str = EGOCENTRIC_S3_URI) -> S3TrainingRenderer:
    """Create an S3 training renderer."""
    bridge = S3RLEnvironmentBridge(s3_uri=s3_uri)
    return S3TrainingRenderer(bridge=bridge)


def get_s3_rl_summary(s3_uri: str = EGOCENTRIC_S3_URI) -> dict[str, Any]:
    """Get a complete S3-grounded RL training summary."""
    bridge = S3RLEnvironmentBridge(s3_uri=s3_uri)
    return bridge.get_training_summary()
