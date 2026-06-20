from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import time
from pathlib import Path
from typing import Iterable

from egocentric_dataset_test.data.ego4d_metadata import Ego4DMetadataStore, FrameEvalRecord

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None  # type: ignore[assignment]


@dataclass(slots=True)
class EnrichmentReport:
    total_episodes: int
    enriched: int
    skipped_no_match: int
    skipped_error: int
    elapsed_s: float


def _extract_factory_worker(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    lowered = text.lower().replace("-", "_")
    factory = None
    worker = None
    factory_match = re.search(r"factory_?(\d{1,4})", lowered)
    worker_match = re.search(r"worker_?(\d{1,6})", lowered)
    if factory_match:
        factory = f"factory_{int(factory_match.group(1)):03d}"
    if worker_match:
        worker = f"worker_{int(worker_match.group(1)):03d}"
    return factory, worker


class EpisodeMetadataEnricher:
    def __init__(self, store: Ego4DMetadataStore, zarr_dir: Path) -> None:
        self.store = store
        self.zarr_dir = Path(zarr_dir)

    def match_clip_to_episode(self, clip_id: str) -> str | None:
        records = self.store.lookup_by_clip(clip_id)
        for record in records:
            if record.zarr_episode_key:
                candidate = self.zarr_dir / record.zarr_episode_key
                if candidate.exists():
                    return record.zarr_episode_key

        factory_id, worker_id = _extract_factory_worker(clip_id)
        if factory_id and worker_id and self.zarr_dir.exists():
            for path in self.zarr_dir.glob("*.zarr"):
                name = path.name.lower()
                if factory_id in name and worker_id in name:
                    return path.name
        return None

    def enrich_zarr(self, episode_key: str, records: list[FrameEvalRecord]) -> None:
        if zarr is None:
            raise RuntimeError("zarr is required for enrichment")
        path = Path(episode_key)
        if not path.is_absolute():
            path = self.zarr_dir / episode_key
        root = zarr.open(str(path), mode="a")

        hand_orientation = Counter(record.hand_orientation for record in records if record.hand_orientation).most_common(1)
        task_class = Counter(record.task_class for record in records if record.task_class).most_common(1)
        pct_active = sum(1 for record in records if record.active_manipulation) / max(len(records), 1)
        mean_hand_count = sum(record.hand_count for record in records) / max(len(records), 1)
        attrs = dict(getattr(root, "attrs", {}))
        attrs["ego4d_meta"] = {
            "clip_id": records[0].clip_id if records else None,
            "hand_orientation": hand_orientation[0][0] if hand_orientation else "none",
            "task_class": task_class[0][0] if task_class else "idle",
            "pct_active": round(float(pct_active), 4),
            "hand_count_mean": round(float(mean_hand_count), 4),
            "frame_count": len(records),
        }
        root.attrs.update(attrs)

    def batch_enrich(self, max_workers: int = 4) -> EnrichmentReport:
        started = time.time()
        clip_groups: dict[str, list[FrameEvalRecord]] = defaultdict(list)
        for record in self.store.records:
            if record.clip_id:
                clip_groups[record.clip_id].append(record)

        work_items: list[tuple[str, list[FrameEvalRecord]]] = []
        skipped_no_match = 0
        for clip_id, records in clip_groups.items():
            episode_key = self.match_clip_to_episode(clip_id)
            if not episode_key:
                skipped_no_match += 1
                continue
            for record in records:
                record.zarr_episode_key = episode_key
            work_items.append((episode_key, records))

        enriched = 0
        skipped_error = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [executor.submit(self.enrich_zarr, episode_key, records) for episode_key, records in work_items]
            for future in as_completed(futures):
                try:
                    future.result()
                    enriched += 1
                except Exception:
                    skipped_error += 1

        return EnrichmentReport(
            total_episodes=len(work_items) + skipped_no_match,
            enriched=enriched,
            skipped_no_match=skipped_no_match,
            skipped_error=skipped_error,
            elapsed_s=round(time.time() - started, 3),
        )
