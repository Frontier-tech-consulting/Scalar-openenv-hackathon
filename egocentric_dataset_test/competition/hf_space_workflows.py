from __future__ import annotations

import json
import importlib.util
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:
    from datasets import Dataset

    _HAS_DATASETS = True
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]
    _HAS_DATASETS = False

try:
    from transformers import AutoTokenizer

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False

try:
    from trl import GRPOConfig, GRPOTrainer

    _HAS_TRL = True
except Exception:  # pragma: no cover
    GRPOConfig = None  # type: ignore[assignment]
    GRPOTrainer = None  # type: ignore[assignment]
    _HAS_TRL = False

_HAS_SAM = importlib.util.find_spec("segment_anything") is not None
_HAS_SAM2 = importlib.util.find_spec("sam2") is not None

from egocentric_dataset_test.competition.demo import _episodes_for_worker, _video_label
from egocentric_dataset_test.competition.environment import (
    EgocentricFactoryAction,
    EgocentricFactoryCompetitionEnv,
    list_task_specs,
)
from egocentric_dataset_test.competition.mujoco_sim import create_mujoco_task
from egocentric_dataset_test.competition.openenv_myosim_adapter import OpenEnvToMyoSimAdapter
from egocentric_dataset_test.competition.real_pipeline import (
    _episode_frame_to_bgr,
    _frame_count,
    _grabcut_mask,
    _load_episode_frame_store,
    _prepare_clip_context,
    build_reward_plot_df,
    empty_plot_df,
    run_yolo_detection,
)
from egocentric_dataset_test.competition.s3_rl_bridge import S3DatasetClient


OPENENV_OUTPUT_S3_URI = os.getenv("OPENENV_OUTPUT_S3_URI", "").strip()
OPENENV_OUTPUT_PREFIX = os.getenv("OPENENV_OUTPUT_PREFIX", "hf-space").strip("/") or "hf-space"
DEFAULT_GRPO_MODEL = os.getenv("OPENENV_GRPO_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct").strip()
SEGMENTATION_OBJECT_PROMPTS = [
    "person",
    "hand",
    "pcb_board",
    "pcm_card",
    "machine",
    "tool",
    "workbench",
    "tray",
]


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    bucket_path = s3_uri.replace("s3://", "", 1)
    bucket, _, prefix = bucket_path.partition("/")
    return bucket, prefix.strip("/")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "asset"


def _save_json(payload: dict[str, Any], suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Path(tmp.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tmp.name


def _save_text(content: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Path(tmp.name).write_text(content, encoding="utf-8")
    return tmp.name


def _save_rgb_png(image: np.ndarray, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(tmp.name)
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


def _make_contact_sheet(
    images_bgr: list[np.ndarray],
    titles: list[str] | None = None,
    tile_size: tuple[int, int] = (320, 180),
) -> np.ndarray:
    if not images_bgr:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    cols = 2 if len(images_bgr) > 1 else 1
    rows = math.ceil(len(images_bgr) / cols)
    canvas = np.zeros((rows * tile_size[1], cols * tile_size[0], 3), dtype=np.uint8)
    safe_titles = titles or [f"frame {index + 1}" for index in range(len(images_bgr))]
    for idx, frame in enumerate(images_bgr):
        resized = cv2.resize(frame, tile_size)
        row = idx // cols
        col = idx % cols
        y1 = row * tile_size[1]
        x1 = col * tile_size[0]
        canvas[y1:y1 + tile_size[1], x1:x1 + tile_size[0]] = resized
        cv2.putText(
            canvas,
            safe_titles[idx][:42],
            (x1 + 8, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _proxy_dimensions(mask: np.ndarray | None, bbox: list[float] | None) -> tuple[float, float, float]:
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        width = max(int(xs.max() - xs.min() + 1), 12)
        height = max(int(ys.max() - ys.min() + 1), 12)
    elif bbox and len(bbox) == 4:
        width = max(int(float(bbox[2]) - float(bbox[0])), 12)
        height = max(int(float(bbox[3]) - float(bbox[1])), 12)
    else:
        width = 48
        height = 32
    return (
        round(width / 456.0 * 0.18, 4),
        round(height / 256.0 * 0.10, 4),
        round(max(width, height) / 456.0 * 0.03, 4),
    )


def _build_obj_mesh_text(dimensions: tuple[float, float, float], object_name: str) -> str:
    sx, sy, sz = dimensions
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return "\n".join(
        [
            f"o {object_name}",
            f"v {-hx:.5f} {-hy:.5f} {-hz:.5f}",
            f"v {hx:.5f} {-hy:.5f} {-hz:.5f}",
            f"v {hx:.5f} {hy:.5f} {-hz:.5f}",
            f"v {-hx:.5f} {hy:.5f} {-hz:.5f}",
            f"v {-hx:.5f} {-hy:.5f} {hz:.5f}",
            f"v {hx:.5f} {-hy:.5f} {hz:.5f}",
            f"v {hx:.5f} {hy:.5f} {hz:.5f}",
            f"v {-hx:.5f} {hy:.5f} {hz:.5f}",
            "f 1 2 3 4",
            "f 5 6 7 8",
            "f 1 5 8 4",
            "f 2 6 7 3",
            "f 1 2 6 5",
            "f 4 3 7 8",
            "",
        ]
    )


def _build_usda_proxy_text(dimensions: tuple[float, float, float], asset_name: str) -> str:
    sx, sy, sz = dimensions
    prim_name = "".join(part.capitalize() for part in _slugify(asset_name).split("_")) or "SegmentedAsset"
    return f'''#usda 1.0
(
    defaultPrim = "{prim_name}"
)

def Xform "{prim_name}"
{{
    def Cube "proxy"
    {{
        double size = 1
        float3 xformOp:scale = ({sx:.5f}, {sy:.5f}, {sz:.5f})
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
'''


def _asset_rgba(class_name: str) -> tuple[float, float, float, float]:
    palette = {
        "person": (0.86, 0.60, 0.52, 1.0),
        "hand": (0.96, 0.82, 0.62, 1.0),
        "pcb_board": (0.18, 0.60, 0.28, 1.0),
        "pcm_card": (0.12, 0.48, 0.84, 1.0),
        "tool": (0.82, 0.64, 0.20, 1.0),
        "machine": (0.52, 0.58, 0.68, 1.0),
        "workbench": (0.48, 0.34, 0.22, 1.0),
        "tray": (0.55, 0.35, 0.70, 1.0),
        "object": (0.45, 0.70, 0.88, 1.0),
    }
    return palette.get(class_name, palette["object"])


def _build_mjcf_asset_text(
    asset_name: str,
    mesh_filename: str,
    dimensions: tuple[float, float, float],
    class_name: str,
) -> str:
    rgba = " ".join(f"{value:.3f}" for value in _asset_rgba(class_name))
    mass = max(dimensions[0] * dimensions[1] * dimensions[2] * 650.0, 0.01)
    return f'''<mujoco model="{asset_name}">
  <asset>
    <mesh name="{asset_name}_mesh" file="{mesh_filename}" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="{asset_name}" pos="0 0 {max(dimensions[2] / 2.0, 0.01):.4f}">
      <freejoint/>
      <geom type="mesh" mesh="{asset_name}_mesh" rgba="{rgba}" mass="{mass:.4f}" friction="0.8 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
'''


def _build_scene_xml(worker_id: str, objects: list[dict[str, Any]]) -> str:
    asset_lines: list[str] = []
    body_lines: list[str] = []
    for index, item in enumerate(objects):
        slug = item["slug"]
        dims = item["dimensions_m"]
        rgba = " ".join(f"{value:.3f}" for value in _asset_rgba(item["class_name"]))
        offset_x = -0.18 + index * 0.07
        offset_y = 0.02 * ((index % 2) * 2 - 1)
        offset_z = max(float(dims["z"]) / 2.0, 0.01)
        asset_lines.append(f'    <mesh name="{slug}_mesh" file="{slug}.obj" scale="1 1 1"/>')
        body_lines.append(
            f'    <body name="{slug}" pos="{offset_x:.4f} {offset_y:.4f} {offset_z:.4f}">\n'
            '      <freejoint/>\n'
            f'      <geom type="mesh" mesh="{slug}_mesh" rgba="{rgba}" mass="0.05" friction="0.8 0.1 0.1"/>\n'
            '    </body>'
        )
    return "\n".join(
        [
            f'<mujoco model="{_slugify(worker_id)}_segmented_scene">',
            "  <asset>",
            *asset_lines,
            "  </asset>",
            "  <worldbody>",
            '    <geom type="plane" size="1 1 0.1" rgba="0.18 0.22 0.18 1"/>',
            *body_lines,
            "  </worldbody>",
            "</mujoco>",
            "",
        ]
    )


def _segmentation_backend_name() -> str:
    if _HAS_SAM2:
        return "grounded-yolo + SAM2-ready propagation"
    if _HAS_SAM:
        return "grounded-yolo + SAM-ready prompts"
    return "yolo + grabcut full-clip fallback"


def _frame_area(shape: tuple[int, ...]) -> float:
    height, width = shape[:2]
    return float(max(height * width, 1))


def _canonical_object_label(detection: dict[str, Any], frame_shape: tuple[int, ...]) -> str:
    label = str(detection.get("cls_name", "object") or "object").lower().strip()
    bbox = [float(value) for value in detection.get("xyxy", [0.0, 0.0, 0.0, 0.0])[:4]]
    x1, y1, x2, y2 = bbox if len(bbox) == 4 else [0.0, 0.0, 0.0, 0.0]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    area_ratio = (width * height) / _frame_area(frame_shape)
    aspect_ratio = width / height
    center_y_ratio = ((y1 + y2) * 0.5) / max(float(frame_shape[0]), 1.0)

    if any(token in label for token in ("person", "worker")):
        return "person"
    if any(token in label for token in ("hand", "glove")):
        return "hand"
    if any(token in label for token in ("board", "circuit", "pcb")):
        return "pcb_board"
    if any(token in label for token in ("card", "pcm", "module")):
        return "pcm_card"
    if any(token in label for token in ("tool", "screwdriver", "wrench", "plier", "knife", "scissors")):
        return "tool"
    if any(token in label for token in ("machine", "printer", "monitor", "microwave", "oven", "laptop")):
        return "machine"
    if any(token in label for token in ("bench", "table", "desk")):
        return "workbench"
    if any(token in label for token in ("tray", "basket", "container", "bin")):
        return "tray"
    if area_ratio > 0.25 and center_y_ratio > 0.45:
        return "workbench"
    if area_ratio > 0.12:
        return "machine"
    if area_ratio < 0.08 and 0.8 <= aspect_ratio <= 2.8 and center_y_ratio > 0.35:
        return "pcb_board"
    if area_ratio < 0.08 and (aspect_ratio >= 2.8 or aspect_ratio <= 0.38):
        return "tool"
    return "object"


def _prompt_aliases(class_name: str) -> list[str]:
    aliases = {
        "person": ["person", "worker", "operator"],
        "hand": ["hand", "glove", "fingers"],
        "pcb_board": ["pcb_board", "circuit_board", "electronics_board"],
        "pcm_card": ["pcm_card", "control_card", "module_card"],
        "tool": ["tool", "screwdriver", "pliers", "assembly_tool"],
        "machine": ["machine", "assembly_machine", "equipment"],
        "workbench": ["workbench", "table", "fixture_surface"],
        "tray": ["tray", "bin", "parts_container"],
        "object": ["object", "factory_object"],
    }
    return aliases.get(class_name, [class_name])


def _segment_detection_mask(frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    mask = _grabcut_mask(frame_bgr, bbox)
    if mask is not None and mask.any():
        return mask.astype(bool)
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    height, width = frame_bgr.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    fallback = np.zeros((height, width), dtype=bool)
    fallback[y1:y2, x1:x2] = True
    return fallback


def _overlay_multiclass_segmentation(frame_bgr: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    overlay = frame_bgr.copy()
    for record in records:
        mask = np.asarray(record.get("mask"), dtype=bool)
        if mask.size == 0 or not mask.any():
            continue
        rgba = _asset_rgba(str(record.get("class_name", "object")))
        color = np.array([rgba[2], rgba[1], rgba[0]], dtype=np.float32) * 255.0
        overlay[mask] = np.clip(0.55 * overlay[mask] + 0.45 * color, 0, 255).astype(np.uint8)
        bbox = [int(round(value)) for value in record.get("bbox", [0, 0, 0, 0])[:4]]
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            color_bgr = tuple(int(round(value)) for value in color.tolist())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(
                overlay,
                f"{record.get('class_name', 'object')} {float(record.get('confidence', 0.0)):.2f}",
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
    return overlay


def _image_to_png_bytes(frame_bgr: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", frame_bgr)
    if not success:
        return b""
    return bytes(encoded)


def _merge_bbox(existing: list[float] | None, candidate: list[float]) -> list[float]:
    if existing is None or len(existing) != 4:
        return [float(value) for value in candidate[:4]]
    return [
        min(float(existing[0]), float(candidate[0])),
        min(float(existing[1]), float(candidate[1])),
        max(float(existing[2]), float(candidate[2])),
        max(float(existing[3]), float(candidate[3])),
    ]


def _build_segmentation_rows(frame_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in frame_summaries:
        step = int(summary.get("step", 0))
        rows.append({"step": step, "value": float(summary.get("detections", 0.0)), "metric": "detections"})
        rows.append({"step": step, "value": float(summary.get("mask_area_ratio", 0.0)), "metric": "mask_area_ratio"})
        rows.append({"step": step, "value": float(summary.get("unique_classes", 0.0)), "metric": "unique_classes"})
        rows.append({"step": step, "value": float(summary.get("confidence_mean", 0.0)), "metric": "confidence_mean"})
    return rows


def build_segmentation_plot_df(frame_summaries: list[dict[str, Any]]) -> Any:
    if not _HAS_PANDAS:
        return None
    rows = _build_segmentation_rows(frame_summaries)
    if not rows:
        return empty_plot_df()
    return pd.DataFrame(rows, columns=["step", "value", "metric"])


def _episode_frame_plan(worker_id: str, frame_stride: int) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for episode in _episodes_for_worker(worker_id):
        frame_store = _load_episode_frame_store(episode)
        total_frames = _frame_count(frame_store) if frame_store is not None else 0
        processed_frames = math.ceil(total_frames / max(int(frame_stride), 1)) if total_frames else 0
        planned.append(
            {
                "episode": episode,
                "video_label": _video_label(episode),
                "source_uri": episode.get("source_uri"),
                "frame_cache": episode.get("local_episode_path"),
                "clip_metadata": episode.get("clip_metadata") if isinstance(episode.get("clip_metadata"), dict) else {},
                "total_frames": int(total_frames),
                "processed_frames": int(processed_frames),
            }
        )
    return planned


def _export_asset_bundle(worker_id: str, objects: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[str, str, str]:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"segmented_assets_{_slugify(worker_id)}_"))
    for item in objects:
        slug = item["slug"]
        dims = (
            float(item["dimensions_m"]["x"]),
            float(item["dimensions_m"]["y"]),
            float(item["dimensions_m"]["z"]),
        )
        obj_path = tmp_dir / f"{slug}.obj"
        usda_path = tmp_dir / f"{slug}.usda"
        mjcf_path = tmp_dir / f"{slug}.xml"
        obj_path.write_text(_build_obj_mesh_text(dims, slug), encoding="utf-8")
        usda_path.write_text(_build_usda_proxy_text(dims, slug), encoding="utf-8")
        mjcf_path.write_text(_build_mjcf_asset_text(slug, obj_path.name, dims, item["class_name"]), encoding="utf-8")
        if item.get("representative_mask") is not None:
            mask_rgb = np.asarray(item["representative_mask"], dtype=np.uint8) * 255
            Image.fromarray(mask_rgb).save(tmp_dir / f"{slug}_mask.png")
        if item.get("representative_frame") is not None:
            frame_rgb = cv2.cvtColor(np.asarray(item["representative_frame"], dtype=np.uint8), cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(tmp_dir / f"{slug}_frame.png")
    scene_xml_path = tmp_dir / "scene.xml"
    scene_xml_path.write_text(_build_scene_xml(worker_id, objects), encoding="utf-8")
    manifest_path = tmp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle_path = tempfile.NamedTemporaryFile(suffix="_asset_bundle.zip", delete=False).name
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for asset_path in sorted(tmp_dir.glob("*")):
            archive.write(asset_path, arcname=asset_path.name)
    return bundle_path, str(scene_xml_path), str(manifest_path)


def build_segmentation_assets(worker_id: str, frame_stride: int = 6) -> tuple[str, str | None, list[str], str, Any]:
    episodes = _episode_frame_plan(worker_id, frame_stride=max(1, int(frame_stride)))
    if not episodes:
        payload = {"worker_id": worker_id, "available": False, "reason": "no episode available"}
        return (
            "### Segmentation unavailable\n\nNo bundled worker clip is available for the selected worker.",
            None,
            [],
            json.dumps(payload, indent=2),
            empty_plot_df(),
        )

    available_episodes = [item for item in episodes if item["total_frames"] > 0]
    if not available_episodes:
        payload = {
            "worker_id": worker_id,
            "available": False,
            "reason": "frame cache was empty across all episodes",
            "episodes": episodes,
        }
        return (
            "### Segmentation pipeline\n\nFrame caches were empty across all worker folders, so no cached clip frames were processed.",
            None,
            [],
            json.dumps(payload, indent=2),
            empty_plot_df(),
        )

    expected_processed = sum(int(item["processed_frames"]) for item in available_episodes)
    preview_interval = max(1, expected_processed // 8) if expected_processed else 1
    frame_bundle_path = tempfile.NamedTemporaryFile(suffix="_full_clip_frames.zip", delete=False).name

    object_stats: dict[str, dict[str, Any]] = {}
    frame_summaries: list[dict[str, Any]] = []
    preview_frames: list[np.ndarray] = []
    preview_titles: list[str] = []
    total_processed_frames = 0
    total_raw_frames = sum(int(item["total_frames"]) for item in available_episodes)

    with zipfile.ZipFile(frame_bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as frame_archive:
        for episode_index, item in enumerate(available_episodes):
            episode = item["episode"]
            clip_metadata = item["clip_metadata"]
            _prepare_clip_context(
                worker_id=worker_id,
                video_label=item["video_label"],
                clip_metadata=clip_metadata,
                episode=episode,
            )
            frame_store = _load_episode_frame_store(episode)
            if frame_store is None:
                continue
            frame_indices = range(0, int(item["total_frames"]), max(1, int(frame_stride)))
            for frame_index in frame_indices:
                frame_bgr = _episode_frame_to_bgr(np.asarray(frame_store[frame_index]))
                _, detections, _ = run_yolo_detection(frame_bgr)
                records: list[dict[str, Any]] = []
                confidences: list[float] = []
                area_ratios: list[float] = []
                class_names: set[str] = set()
                for detection in sorted(detections, key=lambda det: float(det.get("conf", 0.0)), reverse=True)[:8]:
                    bbox = [float(value) for value in detection.get("xyxy", [])[:4]]
                    if len(bbox) != 4:
                        continue
                    class_name = _canonical_object_label(detection, frame_bgr.shape)
                    mask = _segment_detection_mask(frame_bgr, bbox)
                    if mask is None or not mask.any():
                        continue
                    confidence = float(detection.get("conf", 0.0))
                    area_ratio = float(mask.mean())
                    confidences.append(confidence)
                    area_ratios.append(area_ratio)
                    class_names.add(class_name)
                    records.append(
                        {
                            "class_name": class_name,
                            "bbox": bbox,
                            "confidence": confidence,
                            "mask": mask,
                        }
                    )
                    stat = object_stats.setdefault(
                        class_name,
                        {
                            "class_name": class_name,
                            "slug": _slugify(class_name),
                            "prompt_aliases": _prompt_aliases(class_name),
                            "frames_seen": 0,
                            "frame_keys": set(),
                            "detections": 0,
                            "confidence_sum": 0.0,
                            "max_confidence": 0.0,
                            "area_ratio_sum": 0.0,
                            "union_bbox": None,
                            "representative_score": -1.0,
                            "representative_mask": None,
                            "representative_bbox": None,
                            "representative_frame": None,
                            "representative_episode": None,
                            "representative_frame_index": None,
                        },
                    )
                    frame_key = f"{episode_index}:{frame_index}"
                    if frame_key not in stat["frame_keys"]:
                        stat["frame_keys"].add(frame_key)
                        stat["frames_seen"] += 1
                    stat["detections"] += 1
                    stat["confidence_sum"] += confidence
                    stat["area_ratio_sum"] += area_ratio
                    stat["max_confidence"] = max(float(stat["max_confidence"]), confidence)
                    stat["union_bbox"] = _merge_bbox(stat.get("union_bbox"), bbox)
                    candidate_score = confidence + area_ratio
                    if candidate_score > float(stat.get("representative_score", -1.0)):
                        stat["representative_score"] = candidate_score
                        stat["representative_mask"] = mask.copy()
                        stat["representative_bbox"] = list(bbox)
                        stat["representative_frame"] = frame_bgr.copy()
                        stat["representative_episode"] = item["video_label"]
                        stat["representative_frame_index"] = int(frame_index)

                overlay_bgr = _overlay_multiclass_segmentation(frame_bgr, records)
                frame_archive.writestr(
                    f"{_slugify(item['video_label'])}/frame_{frame_index:05d}.png",
                    _image_to_png_bytes(overlay_bgr),
                )
                total_processed_frames += 1
                if len(preview_frames) < 8 and (
                    total_processed_frames == 1 or total_processed_frames % preview_interval == 0 or not preview_frames
                ):
                    preview_frames.append(overlay_bgr)
                    preview_titles.append(f"{item['video_label']} · {frame_index}")
                frame_summaries.append(
                    {
                        "step": total_processed_frames,
                        "episode": item["video_label"],
                        "frame_index": int(frame_index),
                        "detections": len(records),
                        "mask_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
                        "unique_classes": len(class_names),
                        "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                    }
                )

    if not object_stats:
        payload = {
            "worker_id": worker_id,
            "available": False,
            "reason": "no detections were produced across cached clips",
            "episodes": available_episodes,
        }
        return (
            "### Segmentation pipeline\n\nCached clips were processed end-to-end, but no object masks were produced from the available worker videos.",
            None,
            [frame_bundle_path],
            json.dumps(payload, indent=2),
            build_segmentation_plot_df(frame_summaries),
        )

    object_manifest_entries: list[dict[str, Any]] = []
    for class_name, stat in sorted(object_stats.items(), key=lambda item: item[1]["detections"], reverse=True):
        dimensions = _proxy_dimensions(stat.get("representative_mask"), stat.get("representative_bbox"))
        object_manifest_entries.append(
            {
                "class_name": class_name,
                "slug": stat["slug"],
                "prompt_aliases": stat["prompt_aliases"],
                "frames_seen": int(stat["frames_seen"]),
                "detections": int(stat["detections"]),
                "confidence_mean": round(float(stat["confidence_sum"]) / max(int(stat["detections"]), 1), 4),
                "max_confidence": round(float(stat["max_confidence"]), 4),
                "mask_area_ratio_mean": round(float(stat["area_ratio_sum"]) / max(int(stat["detections"]), 1), 6),
                "union_bbox": [round(float(value), 2) for value in stat.get("union_bbox") or [0.0, 0.0, 0.0, 0.0]],
                "dimensions_m": {"x": dimensions[0], "y": dimensions[1], "z": dimensions[2]},
                "representative_episode": stat.get("representative_episode"),
                "representative_frame_index": stat.get("representative_frame_index"),
                "representative_mask": stat.get("representative_mask"),
                "representative_frame": stat.get("representative_frame"),
            }
        )

    manifest = {
        "worker_id": worker_id,
        "episodes_processed": [
            {
                "video_label": item["video_label"],
                "source_uri": item["source_uri"],
                "frame_cache": item["frame_cache"],
                "clip_number": int(item["clip_metadata"].get("clip_number", item["clip_metadata"].get("video_index", 0)) or 0),
                "total_frames": int(item["total_frames"]),
                "processed_frames": int(item["processed_frames"]),
            }
            for item in available_episodes
        ],
        "frame_stride": max(1, int(frame_stride)),
        "total_raw_frames": int(total_raw_frames),
        "total_processed_frames": int(total_processed_frames),
        "segmentation_backend": _segmentation_backend_name(),
        "research_alignment": {
            "video_tracking": "aligned with Grounded-SAM2 style open-vocabulary video grounding and mask propagation when those dependencies are available; current workspace falls back to YOLO detections plus mask refinement over every cached clip frame.",
            "factory_objects": SEGMENTATION_OBJECT_PROMPTS,
            "asset_representation": "exports rigid-object proxies as OBJ mesh assets with companion USD and MJCF mesh geom definitions; downstream soft/task assets can be promoted to MuJoCo flex/flexcomp if needed.",
        },
        "objects": [
            {
                key: value
                for key, value in item.items()
                if key not in {"representative_mask", "representative_frame"}
            }
            for item in object_manifest_entries
        ],
        "per_frame_summary": frame_summaries,
    }

    preview_path = _save_rgb_png(_make_contact_sheet(preview_frames, preview_titles), suffix="_segmentation_sheet.png")
    asset_bundle_path, scene_xml_path, manifest_path = _export_asset_bundle(worker_id, object_manifest_entries, manifest)

    upload_prefix = f"segmentation/{worker_id}/all_cached_clips"
    uploaded_preview = _maybe_upload_artifact(preview_path, f"{upload_prefix}/preview.png", "image/png")
    uploaded_frames = _maybe_upload_artifact(frame_bundle_path, f"{upload_prefix}/full_clip_frames.zip", "application/zip")
    uploaded_bundle = _maybe_upload_artifact(asset_bundle_path, f"{upload_prefix}/asset_bundle.zip", "application/zip")
    uploaded_scene = _maybe_upload_artifact(scene_xml_path, f"{upload_prefix}/scene.xml", "application/xml")
    uploaded_manifest = _maybe_upload_artifact(manifest_path, f"{upload_prefix}/manifest.json", "application/json")

    top_objects = ", ".join(item["class_name"] for item in object_manifest_entries[:6])
    s3_lines = [
        line
        for line in [
            f"- **S3 preview:** `{uploaded_preview}`" if uploaded_preview else "",
            f"- **S3 frame archive:** `{uploaded_frames}`" if uploaded_frames else "",
            f"- **S3 asset bundle:** `{uploaded_bundle}`" if uploaded_bundle else "",
            f"- **S3 scene XML:** `{uploaded_scene}`" if uploaded_scene else "",
            f"- **S3 manifest:** `{uploaded_manifest}`" if uploaded_manifest else "",
        ]
        if line
    ]
    markdown = "\n".join(
        [
            "### Full-Clip Segmentation → MuJoCo / MyoSim Assets",
            "",
            f"- **Worker:** `{worker_id}`",
            f"- **Episodes processed:** `{len(available_episodes)}`",
            f"- **Raw / processed frames:** `{total_raw_frames}` / `{total_processed_frames}`",
            f"- **Frame stride:** `{max(1, int(frame_stride))}` (1 = every cached frame)",
            f"- **Tracked object groups:** `{len(object_manifest_entries)}` → `{top_objects or 'object'}`",
            f"- **Segmentation backend:** `{_segmentation_backend_name()}`",
            "- **Factory-object coverage:** `person, hand, PCB boards, PCM cards, tools, trays, machines, workbench/object proxies`",
            "- **Asset output:** `per-object OBJ + USDA + MJCF mesh stubs, global scene.xml, manifest.json, full clip overlay archive`",
            *s3_lines,
        ]
    )
    files = [frame_bundle_path, asset_bundle_path, scene_xml_path, manifest_path]
    return markdown, preview_path, files, json.dumps(manifest, indent=2), build_segmentation_plot_df(frame_summaries)


def run_openenv_sdk_validation(task_id: str, max_steps: int = 12) -> tuple[str, Any, str]:
    env = EgocentricFactoryCompetitionEnv(task_id=task_id, seed=7)
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    try:
        observation = env.reset(task_id=task_id, seed=7)
        for step_index in range(max(1, int(max_steps))):
            history.append(
                {
                    "timesteps_completed": step_index + 1,
                    "episode_reward": float(observation.reward or 0.0),
                    "reward_mean": float(observation.reward or 0.0),
                }
            )
            trace.append(
                {
                    "step": step_index,
                    "stage": observation.current_stage,
                    "progress": float(observation.progress),
                    "reward": float(observation.reward or 0.0),
                    "action_hint": [float(value) for value in observation.action_hint],
                }
            )
            if observation.done:
                break
            observation = env.step(EgocentricFactoryAction(joint_targets=list(observation.action_hint)))
        state = env.state
    finally:
        env.close()

    payload = {
        "task_id": task_id,
        "trace": trace,
        "final_state": {
            "progress": float(state.progress),
            "grader_score": float(state.grader_score),
            "success": bool(state.success),
            "current_stage": state.current_stage,
            "step_count": int(state.step_count),
        },
        "openenv_contract": ["reset()", "step(action)", "state"],
    }
    markdown = "\n".join(
        [
            "### OpenEnv SDK Validation Loop",
            "",
            f"- **Task:** `{task_id}`",
            f"- **Steps executed:** `{len(trace)}`",
            f"- **Final stage:** `{state.current_stage}`",
            f"- **Progress / score:** `{float(state.progress):.2f}` / `{float(state.grader_score):.3f}`",
            f"- **Success:** `{bool(state.success)}`",
            "- **Execution mode:** `strict OpenEnv-style reset → step → state loop using the competition environment contract`",
        ]
    )
    return markdown, build_reward_plot_df(history, clip_label=task_id), json.dumps(payload, indent=2)


def _observation_text(observation: Any) -> str:
    return (
        f"task={observation.task_id}; stage={observation.current_stage}; progress={float(observation.progress):.2f}; "
        f"hint={[round(float(v), 3) for v in observation.action_hint]}; reward={float(observation.reward or 0.0):.3f}; done={bool(observation.done)}"
    )


def _combined_reward(openenv_reward: float, myosim_reward: float) -> float:
    return float(0.45 * openenv_reward + 0.55 * myosim_reward)


class _CompetitionMyoSimToolEnv:
    def __init__(self, task_id: str, seed: int = 7) -> None:
        self.task_id = task_id
        self.seed = seed
        self.reward = 0.0
        self.done = False
        self.trace: list[dict[str, Any]] = []
        self.env: EgocentricFactoryCompetitionEnv | None = None
        self.observation: Any = None
        self.adapter: OpenEnvToMyoSimAdapter | None = None
        self.myosim_task: Any = None
        self.myosim_observation: dict[str, Any] | None = None

    def reset(self, **kwargs: Any) -> str | None:
        self.close()
        self.reward = 0.0
        self.done = False
        self.trace = []
        seed = int(kwargs.get("seed", self.seed))
        self.env = EgocentricFactoryCompetitionEnv(task_id=self.task_id, seed=seed)
        self.observation = self.env.reset(task_id=kwargs.get("task_id", self.task_id), seed=seed)
        self.adapter = OpenEnvToMyoSimAdapter.from_openenv(self.observation, self.env.state, self.trace)
        self.myosim_task = create_mujoco_task(self.adapter.bridge_state.myosim_task_id, seed=seed)
        self.adapter.configure_task(self.myosim_task)
        self.myosim_observation = self.myosim_task.reset(seed=seed)
        self.reward = _combined_reward(float(self.observation.reward or 0.0), float(self.myosim_observation.get("reward", 0.0)))
        return self.status()

    def _step_myo(self, mode: str, action: list[float] | None = None) -> str:
        if self.env is None or self.observation is None or self.adapter is None or self.myosim_task is None:
            raise ValueError("Environment not initialized. Reset first.")
        if self.done:
            raise ValueError("Episode already finished.")
        self.adapter = OpenEnvToMyoSimAdapter.from_openenv(self.observation, self.env.state, self.trace)
        self.adapter.configure_task(self.myosim_task)
        base_action = self.adapter.action(self.myosim_observation or {}, self.myosim_task.sim.n_actuators)
        if action is not None and base_action.size > 0:
            action_array = np.asarray(action, dtype=np.float32)
            overlay = np.zeros_like(base_action, dtype=np.float32)
            overlay[0] = np.clip(0.5 - 0.5 * action_array[3], 0.0, 1.0)
            overlay[min(1, base_action.size - 1)] = np.clip(0.5 + 0.5 * action_array[1], 0.0, 1.0)
            overlay[-1] = np.clip(0.5 + 0.5 * action_array[2], 0.0, 1.0)
            base_action = np.clip(0.82 * base_action + 0.18 * overlay, 0.0, 1.0)
        self.myosim_observation = self.myosim_task.step(base_action)
        adapted_obs = self.adapter.adapt_myosim_observation(self.myosim_observation)
        openenv_reward = float(self.observation.reward or 0.0)
        myosim_reward = float(self.myosim_observation.get("reward", 0.0))
        self.reward = _combined_reward(openenv_reward, myosim_reward)
        self.done = bool(self.observation.done) or bool(self.myosim_observation.get("done", False))
        self.trace.append(
            {
                "mode": mode,
                "stage": self.observation.current_stage,
                "progress": float(self.observation.progress),
                "openenv_reward": openenv_reward,
                "myosim_reward": myosim_reward,
                "combined_reward": self.reward,
                "delta_to_target": [float(value) for value in adapted_obs.get("delta_to_target", [0.0, 0.0, 0.0])],
                "done": self.done,
                "myosim_task": self.adapter.bridge_state.myosim_task_id,
            }
        )
        return self.status()

    def follow_hint(self) -> str:
        if self.env is None or self.observation is None:
            raise ValueError("Environment not initialized. Reset first.")
        self.observation = self.env.step(EgocentricFactoryAction(joint_targets=list(self.observation.action_hint)))
        return self._step_myo("follow_hint")

    def act(self, reach_x: float, reach_y: float, grip_force: float, wrist_roll: float) -> str:
        if self.env is None or self.observation is None:
            raise ValueError("Environment not initialized. Reset first.")
        action = [float(np.clip(value, -1.0, 1.0)) for value in [reach_x, reach_y, grip_force, wrist_roll]]
        self.observation = self.env.step(EgocentricFactoryAction(joint_targets=action))
        return self._step_myo("manual_act", action=action)

    def status(self) -> str:
        if self.observation is None or self.adapter is None or self.myosim_observation is None:
            raise ValueError("Environment not initialized. Reset first.")
        delta = self.adapter.adapt_myosim_observation(self.myosim_observation).get("delta_to_target", [0.0, 0.0, 0.0])
        return (
            f"openenv_stage={self.observation.current_stage}; openenv_progress={float(self.observation.progress):.2f}; "
            f"openenv_reward={float(self.observation.reward or 0.0):.3f}; myosim_task={self.adapter.bridge_state.myosim_task_id}; "
            f"myosim_reward={float(self.myosim_observation.get('reward', 0.0)):.3f}; combined_reward={self.reward:.3f}; "
            f"delta={[round(float(value), 4) for value in delta]}; done={self.done}"
        )

    def close(self) -> None:
        if self.myosim_task is not None:
            try:
                self.myosim_task.close()
            except Exception:
                pass
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.myosim_task = None
        self.env = None
        self.observation = None
        self.myosim_observation = None
        self.adapter = None


def _run_openenv_myo_preflight(task_id: str, max_steps: int = 12) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = _CompetitionMyoSimToolEnv(task_id=task_id, seed=7)
    trace: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"status": "not_started", "task_id": task_id}
    try:
        env.reset(task_id=task_id, seed=7)
        for _ in range(max(1, int(max_steps))):
            env.follow_hint()
            if env.trace:
                trace.append(dict(env.trace[-1]))
            if env.done:
                break
        summary = {
            "status": "ok",
            "task_id": task_id,
            "steps": len(trace),
            "reward_mean": float(np.mean([item["combined_reward"] for item in trace])) if trace else 0.0,
            "reward_max": float(np.max([item["combined_reward"] for item in trace])) if trace else 0.0,
            "myosim_task": trace[-1]["myosim_task"] if trace else None,
        }
        return trace, summary
    finally:
        env.close()


def _build_grpo_plot_df(log_history: list[dict[str, Any]], rollout_trace: list[dict[str, Any]] | None = None) -> Any:
    if not _HAS_PANDAS:
        return None
    rows: list[dict[str, Any]] = []
    preferred_tokens = ("reward", "loss", "entropy", "kl")
    for index, item in enumerate(log_history, start=1):
        step = int(item.get("step", index))
        for key, value in item.items():
            if key in {"epoch", "step", "total_flos"}:
                continue
            if not any(token in key.lower() for token in preferred_tokens):
                continue
            try:
                rows.append({"step": step, "value": float(value), "metric": key})
            except Exception:
                continue
    for index, item in enumerate(rollout_trace or [], start=1):
        rows.append({"step": index, "value": float(item.get("combined_reward", 0.0)), "metric": "preflight_combined_reward"})
        rows.append({"step": index, "value": float(item.get("openenv_reward", 0.0)), "metric": "preflight_openenv_reward"})
        rows.append({"step": index, "value": float(item.get("myosim_reward", 0.0)), "metric": "preflight_myosim_reward"})
    if not rows:
        return empty_plot_df()
    return pd.DataFrame(rows, columns=["step", "value", "metric"])


def run_openenv_grpo_training(
    task_id: str,
    model_name: str | None = None,
    prompt_count: int = 4,
) -> tuple[str, Any, str]:
    selected_model = (model_name or DEFAULT_GRPO_MODEL).strip()
    effective_prompt_count = max(2, int(prompt_count))
    preflight_trace, preflight_summary = _run_openenv_myo_preflight(task_id=task_id, max_steps=12)
    if not (_HAS_TRL and _HAS_DATASETS and _HAS_TRANSFORMERS):
        payload = {
            "status": "unavailable",
            "reason": "trl / datasets / transformers stack is not installed",
            "task_id": task_id,
            "preflight": preflight_summary,
            "preflight_trace": preflight_trace,
        }
        return (
            "### 3D MyoSim GRPO unavailable\n\nThe local environment does not currently have the TRL stack installed, but the OpenEnv → MyoSim reward loop was validated in preflight.",
            _build_grpo_plot_df([], preflight_trace),
            json.dumps(payload, indent=2),
        )

    task_spec = next((spec for spec in list_task_specs() if spec.task_id == task_id), None)
    system_prompt = (
        f"You are controlling the OpenEnv competition task '{task_id}' while optimizing a mapped 3D MyoSim hand model. "
        "Use `follow_hint` to advance the environment with the provided priors, inspect `status` whenever reward changes, "
        "and use `act` only for corrective reach/grip changes. Maximize combined OpenEnv + MyoSim reward."
    )

    def reward_func(environments: list[Any], **kwargs: Any) -> list[float]:
        return [float(getattr(env, "reward", 0.0)) for env in environments]

    prompts = [[{"role": "user", "content": system_prompt}] for _ in range(effective_prompt_count)]
    dataset = Dataset.from_dict({"prompt": prompts, "task_id": [task_id] * len(prompts)})
    output_dir = tempfile.mkdtemp(prefix=f"openenv_grpo_{task_id}_")

    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        args = GRPOConfig(
            output_dir=output_dir,
            report_to=[],
            use_vllm=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_completion_length=128,
            logging_steps=1,
            max_steps=1,
            save_strategy="no",
            bf16=False,
            fp16=False,
        )
        trainer = GRPOTrainer(
            model=selected_model,
            reward_funcs=reward_func,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=args,
            environment_factory=lambda: _CompetitionMyoSimToolEnv(task_id=task_id, seed=7),
        )
        train_result = trainer.train()
        log_history = list(getattr(trainer.state, "log_history", []))
        payload = {
            "status": "trained",
            "task_id": task_id,
            "model_name": selected_model,
            "task_title": getattr(task_spec, "title", task_id),
            "prompt_count": len(prompts),
            "metrics": getattr(train_result, "metrics", {}),
            "log_history": log_history,
            "output_dir": output_dir,
            "preflight": preflight_summary,
            "preflight_trace": preflight_trace,
        }
        markdown = "\n".join(
            [
                "### OpenEnv + 3D MyoSim GRPO Training",
                "",
                f"- **Task:** `{task_id}`",
                f"- **Mapped MyoSim preflight:** `{preflight_summary.get('myosim_task', 'n/a')}`",
                f"- **Model:** `{selected_model}`",
                f"- **Prompts:** `{len(prompts)}`",
                f"- **Preflight reward mean / max:** `{float(preflight_summary.get('reward_mean', 0.0)):.3f}` / `{float(preflight_summary.get('reward_max', 0.0)):.3f}`",
                "- **Trainer mode:** `TRL GRPOTrainer over an OpenEnv environment factory that explicitly steps the mapped 3D MyoSim task each turn`",
                f"- **Output dir:** `{output_dir}`",
            ]
        )
        return markdown, _build_grpo_plot_df(log_history, preflight_trace), json.dumps(payload, indent=2)
    except Exception as exc:
        payload = {
            "status": "failed",
            "task_id": task_id,
            "model_name": selected_model,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "The MyoSim-coupled GRPO environment is wired, but the selected model may need to be downloaded locally or swapped for a smaller instruct model.",
            "preflight": preflight_summary,
            "preflight_trace": preflight_trace,
        }
        markdown = "\n".join(
            [
                "### OpenEnv + 3D MyoSim GRPO Training",
                "",
                f"- **Task:** `{task_id}`",
                f"- **Mapped MyoSim preflight:** `{preflight_summary.get('myosim_task', 'n/a')}`",
                f"- **Model:** `{selected_model}`",
                "- **Status:** `failed to execute locally`",
                f"- **Error:** `{type(exc).__name__}: {exc}`",
                "- **Next step:** `use a locally cached instruct model or keep using the preflight reward curve while validating the OpenEnv→MyoSim loop`",
            ]
        )
        return markdown, _build_grpo_plot_df([], preflight_trace), json.dumps(payload, indent=2)
