from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "ego_openenv"
DATASET_REFERENCE_PATH = CHECKPOINT_DIR / "dataset_reference.json"
PREPARED_EPISODES_PATH = CHECKPOINT_DIR / "prepared_episodes.json"
TRAINING_SUMMARY_PATH = CHECKPOINT_DIR / "training_summary.json"
TRAINING_METRICS_PATH = CHECKPOINT_DIR / "training_metrics.jsonl"
TASK_SPEC_PATH = CHECKPOINT_DIR / "task_spec.json"
ROLLOUT_VIDEO_PATH = CHECKPOINT_DIR / "rollout_render.mp4"
TRAINING_CURVES_PATH = CHECKPOINT_DIR / "training_curves.png"
BATCH_DIAGNOSTICS_PATH = CHECKPOINT_DIR / "batch_diagnostics.png"
ROLLOUT_DIAGNOSTICS_PATH = CHECKPOINT_DIR / "rollout_diagnostics.png"
REPORT_PATH = CHECKPOINT_DIR / "egocentric_report.html"


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def load_bundled_preview() -> dict[str, Any]:
    dataset_reference = _load_json(DATASET_REFERENCE_PATH, {})
    prepared_episodes = _load_json(PREPARED_EPISODES_PATH, [])
    training_summary = _load_json(TRAINING_SUMMARY_PATH, {})
    training_metrics = _load_jsonl(TRAINING_METRICS_PATH)
    task_spec = _load_json(TASK_SPEC_PATH, {})
    return {
        "dataset_reference": dataset_reference,
        "prepared_episodes": prepared_episodes,
        "training_summary": training_summary,
        "training_metrics": training_metrics,
        "task_spec": task_spec,
        "rollout_video": str(ROLLOUT_VIDEO_PATH) if ROLLOUT_VIDEO_PATH.exists() else None,
        "training_curves": str(TRAINING_CURVES_PATH) if TRAINING_CURVES_PATH.exists() else None,
        "batch_diagnostics": str(BATCH_DIAGNOSTICS_PATH) if BATCH_DIAGNOSTICS_PATH.exists() else None,
        "rollout_diagnostics": str(ROLLOUT_DIAGNOSTICS_PATH) if ROLLOUT_DIAGNOSTICS_PATH.exists() else None,
        "report_path": str(REPORT_PATH) if REPORT_PATH.exists() else None,
        "source": "bundled_checkpoint",
    }


def inspect_live_s3(s3_uri: str, max_assets: int = 5) -> dict[str, Any]:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": f"boto3 unavailable: {exc}", "rows": []}

    if not s3_uri.startswith("s3://"):
        return {"available": False, "error": "s3_uri must start with s3://", "rows": []}

    bucket_and_prefix = s3_uri[5:]
    bucket, _, prefix = bucket_and_prefix.partition("/")
    endpoint_url = os.getenv("R2_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL_S3") or None
    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    profile_name = os.getenv("AWS_PROFILE") or os.getenv("R2_PROFILE") or None

    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        credentials = session.get_credentials()
        if credentials is None:
            profile_hint = f" profile '{profile_name}'" if profile_name else " default profile"
            return {
                "available": False,
                "error": (
                    "No AWS credentials were resolved from boto3's default chain. "
                    f"Run `aws configure` or `aws sso login` for the{profile_hint}, or set AWS_PROFILE."
                ),
                "rows": [],
            }

        client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
    except (BotoCoreError, ClientError, Exception) as exc:
        return {
            "available": False,
            "error": f"Failed to initialize AWS session/client: {exc}",
            "rows": [],
        }

    normalized_prefix = prefix.strip("/")
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=normalized_prefix, MaxKeys=max_assets * 4)
    except (BotoCoreError, ClientError, Exception) as exc:
        return {
            "available": False,
            "error": f"Failed to inspect S3 prefix via AWS session: {exc}",
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    for obj in response.get("Contents", []):
        key = obj.get("Key", "")
        if not key.endswith((".mp4", ".json")):
            continue
        rows.append(
            {
                "key": key,
                "size_mb": round(float(obj.get("Size", 0)) / (1024 * 1024), 3),
                "last_modified": str(obj.get("LastModified", "")),
            }
        )
        if len(rows) >= max_assets:
            break

    return {
        "available": True,
        "bucket": bucket,
        "prefix": normalized_prefix,
        "rows": rows,
        "error": None,
    }


def build_dataset_markdown(preview: dict[str, Any]) -> str:
    reference = preview.get("dataset_reference", {})
    prepared = preview.get("prepared_episodes", [])
    task_spec = preview.get("task_spec", {})
    first = prepared[0] if prepared else {}
    clip = first.get("clip_metadata", {})
    return (
        f"## Real Egocentric-100K Dataset Preview\n\n"
        f"- **Dataset:** `{reference.get('dataset_name', 'Egocentric-100K')}`\n"
        f"- **Source URI:** `{reference.get('source_uri', 's3://buildai-egocentric/raw')}`\n"
        f"- **Embodiment:** `{reference.get('embodiment', 'egocentric100k')}`\n"
        f"- **Prepared episodes:** `{len(prepared)}`\n"
        f"- **Factory / Worker:** `{clip.get('factory_id', 'n/a')}` / `{clip.get('worker_id', 'n/a')}`\n"
        f"- **Representative clip:** `{clip.get('clip_number', 'n/a')}` at `{clip.get('duration_sec', 'n/a')}`s, `{clip.get('fps', 'n/a')}` fps\n"
        f"- **Task prompt:** {task_spec.get('prompt', reference.get('prompt', 'n/a'))}\n\n"
        f"This preview is backed by real Egocentric-100K S3 references and cached training artifacts preserved in the repo."
    )


def build_training_markdown(preview: dict[str, Any]) -> str:
    summary = preview.get("training_summary", {})
    config = summary.get("config", {})
    result = summary.get("result", {})
    online = result.get("online_metrics", [])
    offline = result.get("offline_metrics", [])
    best_reward = max((float(row.get("episode_reward", 0.0)) for row in online), default=0.0)
    last_offline = float(offline[-1].get("offline_loss", 0.0)) if offline else 0.0
    last_online = online[-1] if online else {}
    return (
        f"## RL Training + Inference Preview\n\n"
        f"- **Agent family:** `PPO actor-critic / policy-gradient`\n"
        f"- **Runtime:** `{config.get('runtime', 'replay')}`\n"
        f"- **Timesteps:** `{config.get('total_timesteps', 'n/a')}`\n"
        f"- **Epochs:** `{result.get('total_epochs', 'n/a')}`\n"
        f"- **Best episode reward:** `{best_reward:.3f}`\n"
        f"- **Final offline loss:** `{last_offline:.6f}`\n"
        f"- **Final policy loss:** `{float(last_online.get('policy_loss', 0.0)):.6f}`\n"
        f"- **Final value loss:** `{float(last_online.get('value_loss', 0.0)):.6f}`\n"
        f"- **Final entropy:** `{float(last_online.get('entropy', 0.0)):.6f}`\n"
        f"- **Checkpoint:** `{result.get('model_path', 'n/a')}`\n\n"
        f"The gradient-agent preview surfaces the preserved actor/policy loss, critic/value loss, entropy, and rollout reward traces from the real Egocentric-100K-conditioned training run."
    )


def build_episode_rows(preview: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for episode in preview.get("prepared_episodes", []):
        clip = episode.get("clip_metadata", {})
        rows.append(
            [
                clip.get("factory_id", "n/a"),
                clip.get("worker_id", "n/a"),
                clip.get("clip_number", "n/a"),
                clip.get("duration_sec", "n/a"),
                clip.get("fps", "n/a"),
                episode.get("source_uri", "n/a"),
            ]
        )
    return rows


def build_metric_rows(preview: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in preview.get("training_metrics", []):
        rows.append(
            [
                item.get("phase", "n/a"),
                item.get("epoch", "n/a"),
                item.get("offline_loss", ""),
                item.get("episode_reward", ""),
                item.get("policy_loss", ""),
                item.get("value_loss", ""),
                item.get("entropy", ""),
            ]
        )
    return rows


def build_live_s3_markdown(s3_uri: str, inspection: dict[str, Any]) -> str:
    if not inspection.get("available"):
        return (
            f"### Live S3 Inspection\n\n"
            f"- **Requested URI:** `{s3_uri}`\n"
            f"- **Status:** fallback to bundled checkpoint artifacts\n"
            f"- **Reason:** {inspection.get('error', 'unknown')}"
        )

    rows = inspection.get("rows", [])
    preview_lines = "\n".join(
        f"- `{row['key']}` — `{row['size_mb']}` MB — `{row['last_modified']}`"
        for row in rows
    ) or "- No objects found"
    return (
        f"### Live S3 Inspection\n\n"
        f"- **Bucket:** `{inspection.get('bucket', 'n/a')}`\n"
        f"- **Prefix:** `{inspection.get('prefix', 'n/a')}`\n"
        f"- **Objects shown:** `{len(rows)}`\n\n"
        f"{preview_lines}"
    )