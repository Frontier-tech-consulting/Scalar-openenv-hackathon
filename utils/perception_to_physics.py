"""Real-to-Sim transfer widget and rollout test harness.

As documented in ``transition_real_to_general_meeting.md``.

This module demonstrates the bridge between 2D perception (bounding boxes, depth,
hand joints) and 3D contact-rich physics in MuJoCo. It dynamically constructs a
workbench and a target object, maps 2.5D optical/depth coordinates to the 3D
physics space, and uses kinematic retargeting plus a PD controller to drive a
simulated arm toward the object.

Usage:
    python utils/perception_to_physics.py --headless --steps 300
    python utils/perception_to_physics.py --steps 500
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, UTC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None


DEFAULT_INTRINSICS = {"fx": 137.98, "fy": 138.23, "cx": 232.17, "cy": 125.37}
ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
WORKBENCH_TOP_Z = 0.75
OBJECT_HALF_HEIGHT = 0.01
DEFAULT_OUTPUT_DIR = Path("outputs/perception_to_physics")
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "comparison_report.html"


@dataclass
class RolloutResult:
    """Summary metrics for a short headless MuJoCo rollout."""

    target_pos: np.ndarray
    target_joints: np.ndarray
    initial_distance: float
    final_distance: float
    min_distance: float
    object_displacement: float
    object_height: float
    contact_count: int
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_pos": self.target_pos.tolist(),
            "target_joints": self.target_joints.tolist(),
            "initial_distance": self.initial_distance,
            "final_distance": self.final_distance,
            "min_distance": self.min_distance,
            "object_displacement": self.object_displacement,
            "object_height": self.object_height,
            "contact_count": self.contact_count,
            "steps": self.steps,
        }


def _require_mujoco() -> Any:
    if mujoco is None:
        raise ImportError("Please install mujoco: pip install mujoco")
    return mujoco


def _require_imageio() -> Any:
    if imageio is None:
        raise ImportError("Please install imageio: pip install imageio")
    return imageio


def _pad_frame_for_video(frame: np.ndarray, block_size: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width = ((width + block_size - 1) // block_size) * block_size
    if padded_height == height and padded_width == width:
        return frame
    canvas = np.zeros((padded_height, padded_width, frame.shape[2]), dtype=frame.dtype)
    canvas[:height, :width] = frame
    return canvas


def _make_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "timestamped_metrics": run_dir / "metrics.json",
        "timestamped_video": run_dir / "rollout.mp4",
        "latest_metrics": output_dir / "latest_metrics.json",
        "latest_video": output_dir / "latest_rollout.mp4",
        "history": output_dir / "runs.jsonl",
    }


def _load_runs(history_path: Path) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if not history_path.exists():
                return runs
        for raw_line in history_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                        continue
                try:
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                                runs.append(payload)
                except json.JSONDecodeError:
                        continue
        runs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return runs


def _video_src_for_report(video_path: Path, report_path: Path) -> str:
        """Return a browser-friendly video source path for HTML report embedding."""
        try:
                rel_path = video_path.resolve().relative_to(report_path.parent.resolve())
                return rel_path.as_posix()
        except ValueError:
                return video_path.resolve().as_uri()


def generate_comparison_report(
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        output_path: Path | str | None = None,
) -> Path:
        """Generate an HTML page listing all runs with side-by-side embedded videos."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = Path(output_path) if output_path is not None else DEFAULT_REPORT_PATH
        if not report_path.is_absolute():
                report_path = (output_dir / report_path).resolve() if report_path.parent == Path(".") else report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)

        runs = _load_runs(output_dir / "runs.jsonl")
        table_rows: list[str] = []
        video_cards: list[str] = []

        for run in runs:
                run_id = str(run.get("run_id", "unknown"))
                created_at = str(run.get("created_at", ""))
                steps = int(run.get("steps", 0) or 0)
                start_distance = float(run.get("initial_distance", float("nan")))
                end_distance = float(run.get("final_distance", float("nan")))
                min_distance = float(run.get("min_distance", float("nan")))
                contacts = int(run.get("contact_count", 0) or 0)
                video_path = Path(str(run.get("video_path", "")))
                has_video = video_path.exists()
                if has_video:
                        video_src = _video_src_for_report(video_path, report_path)
                        video_embed = (
                                f"<video controls preload=\"metadata\" width=\"640\" height=\"480\">"
                                f"<source src=\"{html.escape(video_src)}\" type=\"video/mp4\" />"
                                "Your browser does not support embedded videos."
                                "</video>"
                        )
                else:
                        video_embed = "<div class=\"missing\">Video not found on disk for this run.</div>"

                table_rows.append(
                        "<tr>"
                        f"<th scope=\"row\">{html.escape(run_id)}</th>"
                        f"<td>{html.escape(created_at)}</td>"
                        f"<td>{steps}</td>"
                        f"<td>{start_distance:.4f}</td>"
                        f"<td>{end_distance:.4f}</td>"
                        f"<td>{min_distance:.4f}</td>"
                        f"<td>{contacts}</td>"
                        f"<td>{'yes' if has_video else 'no'}</td>"
                        "</tr>"
                )

                video_cards.append(
                        "<article class=\"card\">"
                        f"<h3>{html.escape(run_id)}</h3>"
                        f"<p class=\"meta\">{html.escape(created_at)} · steps={steps} · contacts={contacts}</p>"
                        f"<p class=\"meta\">distance: start={start_distance:.4f}, end={end_distance:.4f}, min={min_distance:.4f}</p>"
                        f"{video_embed}"
                        "</article>"
                )

        if not runs:
                table_rows.append(
                        "<tr><th scope=\"row\">n/a</th><td colspan=\"7\">No runs found in runs.jsonl. Generate at least one rollout with --render-video.</td></tr>"
                )
                video_cards.append("<article class=\"card\"><h3>No videos yet</h3><p class=\"meta\">Run with --render-video first.</p></article>")

        html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Perception-to-Physics Run Comparison</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0 auto; max-width: 1400px; padding: 24px; background: #f6f8fc; color: #1b2233; }}
        h1, h2 {{ margin: 0 0 12px; }}
        .summary {{ background: #ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 6px 20px rgba(26, 38, 72, 0.08); margin-bottom: 16px; }}
        .table-wrap {{ overflow-x: auto; background: #ffffff; border-radius: 12px; box-shadow: 0 6px 20px rgba(26, 38, 72, 0.08); margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; border-spacing: 0; }}
        caption {{ caption-side: top; text-align: left; font-weight: 700; padding: 14px 16px 8px; }}
        th, td {{ border: 1px solid #dde3f0; padding: 8px 10px; font-size: 0.92rem; }}
        thead > tr {{ background: #edf3ff; }}
        tbody > tr:nth-of-type(even) {{ background: #fafcff; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }}
        .card {{ background: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 6px 20px rgba(26, 38, 72, 0.08); }}
        .meta {{ margin: 6px 0; color: #415071; font-size: 0.88rem; }}
        video {{ width: 100%; height: auto; border-radius: 10px; background: #e8edf7; display: block; }}
        .missing {{ padding: 16px; background: #fff3cd; color: #7f5f00; border: 1px solid #ffe69c; border-radius: 8px; }}
    </style>
</head>
<body>
    <section class=\"summary\">
        <h1>Perception-to-Physics Comparison</h1>
        <p>Runs loaded from <strong>{html.escape(str((output_dir / 'runs.jsonl').resolve()))}</strong></p>
        <p>Total runs: <strong>{len(runs)}</strong></p>
    </section>

    <section class=\"table-wrap\">
        <table>
            <caption>Rollout Metrics</caption>
            <thead>
                <tr>
                    <th scope=\"col\">Run ID</th>
                    <th scope=\"col\">Created At</th>
                    <th scope=\"col\">Steps</th>
                    <th scope=\"col\">Initial Dist</th>
                    <th scope=\"col\">Final Dist</th>
                    <th scope=\"col\">Min Dist</th>
                    <th scope=\"col\">Contacts</th>
                    <th scope=\"col\">Video</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </section>

    <section class=\"grid\">
        {''.join(video_cards)}
    </section>
</body>
</html>
"""

        report_path.write_text(html_doc, encoding="utf-8")
        return report_path

# ── 1. Perception Mocks (Simulating YOLO + Depth Anything V2 + HaMeR) ───────

def mock_yolo_detection() -> dict[str, Any]:
    """Simulates 2D bounding box detection for a PCB component on the workbench."""
    return {
        "class": "pcb_component",
        "bbox_2d": [180.0, 150.0, 220.0, 170.0],  # x1, y1, x2, y2 in pixels (456x256 image)
        "confidence": 0.98
    }

def mock_depth_estimation(bbox_2d: list[float]) -> float:
    """Simulates Depth Anything V2 assigning a metrical depth to the 2D crop."""
    # Assuming the camera is 0.8m away from the workbench
    return 0.85

def unproject_2d_to_3d(x: float, y: float, depth: float, intrinsics: dict) -> np.ndarray:
    """Classic pinhole camera unprojection from 2.5D pixel space to 3D world space."""
    # Intrinsics matching our Egocentric-100K defaults
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    
    # 3D coordinates relative to camera frame
    x_c = (x - cx) * depth / fx
    y_c = (y - cy) * depth / fy
    z_c = depth
    
    # Transform camera frame to MuJoCo world frame (assuming camera is at Z=1.5, looking down pitch=-45 deg)
    theta = math.radians(-45)
    R_cam_to_world = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)]
    ])
    cam_pos_world = np.array([0.0, -0.5, 1.5])
    
    point_world = cam_pos_world + R_cam_to_world @ np.array([x_c, y_c, z_c])
    return point_world

def infer_target_position(intrinsics: dict[str, float] | None = None) -> np.ndarray:
    """Run the mocked perception stack and return a workbench-aligned 3D target."""
    intrinsics = intrinsics or DEFAULT_INTRINSICS
    yolo_data = mock_yolo_detection()
    center_x = (yolo_data["bbox_2d"][0] + yolo_data["bbox_2d"][2]) / 2.0
    center_y = (yolo_data["bbox_2d"][1] + yolo_data["bbox_2d"][3]) / 2.0
    depth = mock_depth_estimation(yolo_data["bbox_2d"])
    target_3d = unproject_2d_to_3d(center_x, center_y, depth, intrinsics)
    target_3d[2] = WORKBENCH_TOP_Z + OBJECT_HALF_HEIGHT
    return target_3d


# ── 2. Scene Reconstruction (Spawning Digital Twins) ──────────────────────

def generate_mujoco_xml(target_pos: np.ndarray) -> str:
    """Generates a MuJoCo scene XML with a basic Humanoid, a workbench, and a target object."""
    tx, ty, tz = target_pos
    return f"""
    <mujoco model="Real_to_Sim_Widget">
      <compiler angle="degree" coordinate="local"/>
      <option timestep="0.01" gravity="0 0 -9.81"/>
      <asset>
        <texture type="skybox" builtin="gradient" width="128" height="128" rgb1=".4 .6 .8" rgb2="0 0 0"/>
        <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="127" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 .2" width="512" height="512"/>
        <material name="MatPlane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
      <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -1"/>
        <geom name="floor" type="plane" material="MatPlane" size="5 5 0.1"/>
        
        <!-- The Workbench -->
        <body name="workbench" pos="0 0.5 0.7">
            <geom type="box" size="0.6 0.4 0.05" rgba="0.2 0.6 0.3 1" mass="100"/>
        </body>
        
        <!-- The Reconstructed PCB Component -->
        <body name="pcb_target" pos="{tx:.3f} {ty:.3f} {tz:.3f}">
            <freejoint/>
            <geom type="box" size="0.04 0.02 0.01" rgba="0.8 0.2 0.2 1" mass="0.1" friction="1 0.005 0.0001"/>
        </body>

        <!-- Abstract Robotic/Humanoid Arm (Simplified for Retargeting demo) -->
        <body name="base_link" pos="0 0 0.9">
            <geom type="cylinder" size="0.05 0.2" rgba="0.7 0.7 0.7 1"/>
            <body name="shoulder" pos="0 0 0.2">
                <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom type="sphere" size="0.06"/>
                <body name="upper_arm" pos="0 0.1 0">
                    <joint name="shoulder_lift" type="hinge" axis="1 0 0" range="-90 90"/>
                    <geom type="capsule" size="0.04 0.15" pos="0 0.15 0" euler="90 0 0"/>
                    <body name="elbow" pos="0 0.3 0">
                        <joint name="elbow_flex" type="hinge" axis="1 0 0" range="-150 0"/>
                        <geom type="sphere" size="0.05"/>
                        <body name="forearm" pos="0 0.15 0">
                            <geom type="capsule" size="0.035 0.15" pos="0 0.15 0" euler="90 0 0"/>
                            <body name="wrist" pos="0 0.3 0">
                                <joint name="wrist_flex" type="hinge" axis="1 0 0" range="-90 90"/>
                                <!-- The Hand / End Effector -->
                                <body name="end_effector" pos="0 0.05 0">
                                    <geom type="sphere" size="0.04" rgba="0.9 0.8 0.6 1"/>
                                    <site name="grasp_site" pos="0 0.05 0" size="0.01" rgba="1 1 0 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
      </worldbody>
      
      <actuator>
        <motor name="shoulder_pan_motor" joint="shoulder_pan" ctrlrange="-50 50"/>
        <motor name="shoulder_lift_motor" joint="shoulder_lift" ctrlrange="-50 50"/>
        <motor name="elbow_flex_motor" joint="elbow_flex" ctrlrange="-50 50"/>
        <motor name="wrist_flex_motor" joint="wrist_flex" ctrlrange="-20 20"/>
      </actuator>
    </mujoco>
    """

def build_simulation(target_pos: np.ndarray):
    """Construct a MuJoCo model/data pair for the inferred task scene."""
    mj = _require_mujoco()
    xml_str = generate_mujoco_xml(target_pos)
    model = mj.MjModel.from_xml_string(xml_str)
    data = mj.MjData(model)
    return model, data


def get_arm_joint_addresses(model) -> tuple[np.ndarray, np.ndarray]:
    """Return qpos and qvel addresses for the controllable arm joints."""
    mj = _require_mujoco()
    joint_ids = []
    qpos_addresses = []
    qvel_addresses = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        joint_ids.append(joint_id)
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        qvel_addresses.append(model.jnt_dofadr[joint_id])
    return (
        np.asarray(qpos_addresses, dtype=int),
        np.asarray(qvel_addresses, dtype=int),
        np.asarray(joint_ids, dtype=int),
    )


def get_site_position(model, data, site_name: str) -> np.ndarray:
    mj = _require_mujoco()
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


# ── 3. Kinematic Retargeting (Inverse Kinematics) ─────────────────────────

def calculate_ik(model, data, site_name: str, target_pos: np.ndarray, max_steps: int = 100):
    """
    Translates the 3D target coordinate into joint angles (Kinematic mimicry).
    This mimics what Pinocchio/dex-retargeting would do.
    """
    mj = _require_mujoco()
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    arm_qpos_addrs, arm_qvel_addrs, joint_ids = get_arm_joint_addresses(model)
    
    for _ in range(max_steps):
        mj.mj_forward(model, data)
        current_pos = data.site_xpos[site_id]
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < 0.01:
            break
            
        mj.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, arm_qvel_addrs]
        
        # Damped Least Squares IK (pseudo-inverse)
        lambda_val = 0.1
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_val ** 2 * np.eye(3))
        delta_q = J_pinv @ error
        
        data.qpos[arm_qpos_addrs] += delta_q * 0.1
        lower_limits = np.deg2rad(model.jnt_range[joint_ids, 0])
        upper_limits = np.deg2rad(model.jnt_range[joint_ids, 1])
        data.qpos[arm_qpos_addrs] = np.clip(data.qpos[arm_qpos_addrs], lower_limits, upper_limits)
        mj.mj_kinematics(model, data)

    return data.qpos[arm_qpos_addrs].copy()


def compute_distance_to_target(model, data, target_pos: np.ndarray, site_name: str = "grasp_site") -> float:
    """Euclidean distance between end effector site and object target."""
    site_pos = get_site_position(model, data, site_name)
    return float(np.linalg.norm(target_pos - site_pos))


def run_headless_rollout(steps: int = 300, kp: float = 50.0, kd: float = 5.0) -> RolloutResult:
    """Run a short rollout without a viewer and return metrics for testing."""
    mj = _require_mujoco()
    target_pos = infer_target_position()
    model, data = build_simulation(target_pos)
    target_joints = calculate_ik(model, data, site_name="grasp_site", target_pos=target_pos)
    arm_qpos_addrs, arm_qvel_addrs, _ = get_arm_joint_addresses(model)
    initial_distance = compute_distance_to_target(model, data, target_pos)
    min_distance = initial_distance
    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pcb_target")
    object_start_pos = data.xpos[object_body_id].copy()
    contact_count = 0

    for _ in range(steps):
        current_q = data.qpos[arm_qpos_addrs]
        current_v = data.qvel[arm_qvel_addrs]
        data.ctrl[:] = kp * (target_joints - current_q) - kd * current_v
        mj.mj_step(model, data)
        min_distance = min(min_distance, compute_distance_to_target(model, data, target_pos))
        if data.ncon > 0:
            contact_count += int(data.ncon)

    final_distance = compute_distance_to_target(model, data, target_pos)
    object_end_pos = data.xpos[object_body_id].copy()
    return RolloutResult(
        target_pos=target_pos,
        target_joints=target_joints,
        initial_distance=initial_distance,
        final_distance=final_distance,
        min_distance=min_distance,
        object_displacement=float(np.linalg.norm(object_end_pos - object_start_pos)),
        object_height=float(object_end_pos[2]),
        contact_count=contact_count,
        steps=steps,
    )


def render_rollout_video(
    steps: int = 300,
    output_path: Path | str = DEFAULT_OUTPUT_DIR / "latest_rollout.mp4",
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    kp: float = 50.0,
    kd: float = 5.0,
) -> RolloutResult:
    """Render a headless MuJoCo rollout to MP4 and return metrics."""
    mj = _require_mujoco()
    io = _require_imageio()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_pos = infer_target_position()
    model, data = build_simulation(target_pos)
    target_joints = calculate_ik(model, data, site_name="grasp_site", target_pos=target_pos)
    arm_qpos_addrs, arm_qvel_addrs, _ = get_arm_joint_addresses(model)
    initial_distance = compute_distance_to_target(model, data, target_pos)
    min_distance = initial_distance
    object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pcb_target")
    object_start_pos = data.xpos[object_body_id].copy()
    contact_count = 0
    renderer = mj.Renderer(model, height=height, width=width)

    try:
        with io.get_writer(str(output_path), fps=fps) as writer:
            for _ in range(steps):
                current_q = data.qpos[arm_qpos_addrs]
                current_v = data.qvel[arm_qvel_addrs]
                data.ctrl[:] = kp * (target_joints - current_q) - kd * current_v
                mj.mj_step(model, data)
                min_distance = min(min_distance, compute_distance_to_target(model, data, target_pos))
                if data.ncon > 0:
                    contact_count += int(data.ncon)
                renderer.update_scene(data)
                frame = renderer.render()
                writer.append_data(_pad_frame_for_video(frame))
    finally:
        renderer.close()

    final_distance = compute_distance_to_target(model, data, target_pos)
    object_end_pos = data.xpos[object_body_id].copy()
    return RolloutResult(
        target_pos=target_pos,
        target_joints=target_joints,
        initial_distance=initial_distance,
        final_distance=final_distance,
        min_distance=min_distance,
        object_displacement=float(np.linalg.norm(object_end_pos - object_start_pos)),
        object_height=float(object_end_pos[2]),
        contact_count=contact_count,
        steps=steps,
    )


def save_rollout_artifacts(
    result: RolloutResult,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    video_path: Path | str | None = None,
    artifact_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Persist rollout metrics and keep latest aliases for comparison across runs."""
    output_dir = Path(output_dir)
    paths = artifact_paths or _make_output_paths(output_dir)
    payload = {
        "run_id": paths["run_dir"].name,
        "created_at": datetime.now(UTC).isoformat(),
        "video_path": str((Path(video_path) if video_path is not None else paths["timestamped_video"]).resolve()),
        **result.to_dict(),
    }
    metrics_blob = json.dumps(payload, indent=2)
    history_blob = json.dumps(payload)
    paths["timestamped_metrics"].write_text(metrics_blob + "\n", encoding="utf-8")
    paths["latest_metrics"].write_text(metrics_blob + "\n", encoding="utf-8")
    with paths["history"].open("a", encoding="utf-8") as history_file:
        history_file.write(history_blob + "\n")
    return paths

# ── 4. Main Simulation Loop ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perception-to-physics rollout demo")
    parser.add_argument("--steps", type=int, default=300, help="Number of rollout steps to simulate")
    parser.add_argument("--headless", action="store_true", help="Run a non-visual rollout and print metrics")
    parser.add_argument("--render-video", action="store_true", help="Render a headless MP4 rollout into outputs")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML comparison page from runs.jsonl")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH, help="Output HTML path for comparison report")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/MP4 rollout artifacts")
    return parser.parse_args()


def main() -> None:
    mj = _require_mujoco()
    args = parse_args()
    print("=== Perception to Physics: Real-to-Sim Widget ===")
    target_3d = infer_target_position()
    print(f"[1] Inferred task target at: {target_3d}")

    if args.generate_report:
        report_path = generate_comparison_report(output_dir=args.output_dir, output_path=args.report_output)
        print(f"[2] Comparison report generated: {report_path}")
        return

    if args.render_video:
        artifact_paths = _make_output_paths(args.output_dir)
        result = render_rollout_video(steps=args.steps, output_path=artifact_paths["timestamped_video"])
        save_rollout_artifacts(
            result,
            output_dir=args.output_dir,
            video_path=artifact_paths["timestamped_video"],
            artifact_paths=artifact_paths,
        )
        artifact_paths["latest_video"].write_bytes(artifact_paths["timestamped_video"].read_bytes())
        print(f"[2] Rendered rollout video: {artifact_paths['timestamped_video']}")
        print(f"[3] Latest rollout video: {artifact_paths['latest_video']}")
        print(f"[4] Metrics JSON: {artifact_paths['latest_metrics']}")
        print(f"[5] Distance to object: start={result.initial_distance:.4f} end={result.final_distance:.4f} min={result.min_distance:.4f}")
        return

    if args.headless:
        result = run_headless_rollout(steps=args.steps)
        print(f"[2] Headless rollout steps: {result.steps}")
        print(f"[3] Distance to object: start={result.initial_distance:.4f} end={result.final_distance:.4f} min={result.min_distance:.4f}")
        print(f"[4] Object displacement: {result.object_displacement:.4f}m, contacts observed: {result.contact_count}")
        return

    model, data = build_simulation(target_3d)
    target_joints = calculate_ik(model, data, site_name="grasp_site", target_pos=target_3d)
    arm_qpos_addrs, arm_qvel_addrs, _ = get_arm_joint_addresses(model)
    kp = 50.0
    kd = 5.0

    print("[2] Launching Contact-Rich Simulation Physics Engine...")
    print("    Targeting grasp on the PCB component. Close the viewer to exit.")
    with mj.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_q = data.qpos[arm_qpos_addrs]
            current_v = data.qvel[arm_qvel_addrs]
            data.ctrl[:] = kp * (target_joints - current_q) - kd * current_v
            mj.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
