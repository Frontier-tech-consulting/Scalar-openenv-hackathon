from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from egocentric_dataset_test.components.metadata_enricher import EpisodeMetadataEnricher
from egocentric_dataset_test.components.pose_extractor import PoseExtractor, PoseSequence
from egocentric_dataset_test.data.ego4d_metadata import Ego4DMetadataStore, FrameEvalRecord
from egocentric_dataset_test.realistic_sim.supply_chain_mjcf import SupplyChainSceneBuilder, WorkerConfig

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None  # type: ignore[assignment]


@dataclass(slots=True)
class SimResult:
    video_path: Path | None
    reward_curve: np.ndarray
    pose_sequence: PoseSequence
    eval_metadata: list[FrameEvalRecord]
    mjmodel_xml: str
    episode_stats: dict[str, Any]


class VideoToSimPipeline:
    def __init__(
        self,
        metadata_store: Ego4DMetadataStore,
        pose_extractor: PoseExtractor | None = None,
        scene_builder: SupplyChainSceneBuilder | None = None,
        enricher: EpisodeMetadataEnricher | None = None,
        outputs_dir: Path | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.pose_extractor = pose_extractor or PoseExtractor()
        self.scene_builder = scene_builder or SupplyChainSceneBuilder()
        self.enricher = enricher
        self.outputs_dir = outputs_dir or Path("outputs") / "video_to_sim"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def from_zarr(self, episode_key: str) -> SimResult:
        if zarr is None:
            raise RuntimeError("zarr is required for from_zarr()")
        root = zarr.open(str(episode_key), mode="r")
        metadata = dict(getattr(root, "attrs", {})).get("ego4d_meta", {})
        clip_id = metadata.get("clip_id") if isinstance(metadata, dict) else None
        frames = []
        if clip_id:
            sample = self.metadata_store.sample_frame(task_class=None)
            if sample:
                frames = [sample]
        return self.from_jpeg_frames(frames, clip_id=clip_id, episode_key=episode_key)

    def from_mp4(self, video_path: Path, clip_id: str | None = None) -> SimResult:
        reader = imageio.get_reader(str(video_path))
        frames: list[bytes] = []
        for idx, frame in enumerate(reader):
            if idx >= 8:
                break
            encoded = imageio.imwrite(imageio.RETURN_BYTES, frame, format="jpg")
            frames.append(encoded)
        return self.from_jpeg_frames(frames, clip_id=clip_id, episode_key=None)

    def from_jpeg_frames(
        self,
        frames: list[bytes],
        clip_id: str | None = None,
        episode_key: str | None = None,
    ) -> SimResult:
        pose_sequence = self.pose_extractor.extract_from_clip(frames)
        eval_metadata = self.metadata_store.lookup_by_clip(clip_id) if clip_id else []

        hand_orientation = pose_sequence.dominant_hand()
        task_class = eval_metadata[0].task_class if eval_metadata else "assembly"
        config = WorkerConfig(hand_orientation=hand_orientation if hand_orientation != "none" else "both", task_class=task_class)
        mjcf = self.scene_builder.generate_mjcf(config)

        if episode_key and self.enricher and eval_metadata:
            self.enricher.enrich_zarr(episode_key, eval_metadata)

        reward_curve = np.linspace(0.1, 1.0, max(len(frames), 1), dtype=np.float32)
        episode_stats = {
            "clip_id": clip_id,
            "frame_count": len(frames),
            "task_class": task_class,
            "hand_orientation": config.hand_orientation,
            "pose_confidence_mean": float(np.mean([record.confidence for record in pose_sequence.records])) if pose_sequence.records else 0.0,
        }
        return SimResult(
            video_path=None,
            reward_curve=reward_curve,
            pose_sequence=pose_sequence,
            eval_metadata=eval_metadata,
            mjmodel_xml=mjcf,
            episode_stats=episode_stats,
        )
