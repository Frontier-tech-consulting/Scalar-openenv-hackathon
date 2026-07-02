"""Ego4D metadata access helpers.

Provides :class:`FrameEvalRecord` and :class:`Ego4DMetadataStore`, which load
frame-level evaluation metadata from a HuggingFace dataset (via the
``datasets`` pip library) with a graceful fallback to an empty in-memory store
when the dataset is unavailable.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure the repository root is resolvable via sys.path so that sibling
# packages (egocentric_dataset_test.*) are always importable regardless of
# the working directory from which the module is loaded.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from datasets import load_dataset as _load_dataset  # type: ignore[import]

    _HAS_DATASETS = True
except Exception:  # pragma: no cover
    _load_dataset = None  # type: ignore[assignment]
    _HAS_DATASETS = False


@dataclass
class FrameEvalRecord:
    """Single-frame evaluation record sourced from an Ego4D-style metadata store."""

    clip_id: str | None = None
    task_class: str | None = None
    confidence: float = 0.0
    hand_orientation: str | None = None
    active_manipulation: bool = False
    hand_count: int = 0
    zarr_episode_key: str | None = None


def _row_to_record(row: dict[str, Any]) -> FrameEvalRecord:
    return FrameEvalRecord(
        clip_id=row.get("clip_id") or row.get("video_uid"),
        task_class=row.get("task_class") or row.get("narration_class"),
        confidence=float(row.get("confidence", 0.0)),
        hand_orientation=row.get("hand_orientation"),
        active_manipulation=bool(row.get("active_manipulation", False)),
        hand_count=int(row.get("hand_count", 0)),
        zarr_episode_key=row.get("zarr_episode_key"),
    )


class Ego4DMetadataStore:
    """In-memory store of :class:`FrameEvalRecord` objects.

    On construction the store tries to stream the first shard of *dataset_id*
    from HuggingFace Hub (requires the ``datasets`` package and network access).
    If the dataset cannot be loaded the store starts empty and all lookup
    methods return empty / ``None`` results.

    Parameters
    ----------
    dataset_id:
        HuggingFace Hub dataset repository, e.g.
        ``"builddotai/Egocentric-100K-Evaluation"``.
    split:
        Dataset split to load (default ``"train"``).
    max_records:
        Maximum number of records to keep in memory (default 10 000).
    """

    def __init__(
        self,
        dataset_id: str = "builddotai/Egocentric-100K-Evaluation",
        split: str = "train",
        max_records: int = 10_000,
    ) -> None:
        self.dataset_id = dataset_id
        self.records: list[FrameEvalRecord] = []
        self._by_clip: dict[str, list[FrameEvalRecord]] = {}

        if _HAS_DATASETS and _load_dataset is not None:
            try:
                ds = _load_dataset(dataset_id, split=split, streaming=True)
                for i, row in enumerate(ds):
                    if i >= max_records:
                        break
                    record = _row_to_record(row)  # type: ignore[arg-type]
                    self.records.append(record)
                    if record.clip_id:
                        self._by_clip.setdefault(record.clip_id, []).append(record)
            except Exception:  # pragma: no cover
                pass  # leave store empty; callers handle empty results

    def lookup_by_clip(self, clip_id: str) -> list[FrameEvalRecord]:
        """Return all records matching *clip_id*, or an empty list."""
        return list(self._by_clip.get(clip_id, []))

    def sample_frame(self, task_class: str | None = None) -> FrameEvalRecord | None:
        """Return a random record, optionally filtered by *task_class*."""
        pool = (
            [r for r in self.records if r.task_class == task_class]
            if task_class is not None
            else self.records
        )
        return random.choice(pool) if pool else None
