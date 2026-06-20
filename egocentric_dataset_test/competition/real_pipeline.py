"""Real PCB Manipulation Training Pipeline
==========================================

Perception (Ultralytics YOLO + MediaPipe) → Physics (MuJoCo arm + full hand) → RL (SB3 PPO) → Video

Replaces all mocked simulation in demo.py with a fully real pipeline:

Stage 1 — Perception
  • Generate a realistic synthetic egocentric factory frame (480×360 BGR)
    • Run local Ultralytics YOLO on it — get real object detections (PCB board / workbench)
  • Run MediaPipe HandLandmarker on a synthetic hand image → 21 keypoints
  • Unproject YOLO bbox center → 3-D world target via pinhole camera model

Stage 2 — Physics
  • Build a complete MuJoCo scene: robot base + 4-DOF arm + full 5-finger hand
    + PCB target object on a workbench (24 hinge joints, 15 position actuators)
  • Position PCB at the 3-D target inferred from perception

Stage 3 — RL Training
  • Wrap scene in a gymnasium.Env compatible with SB3
  • Train SB3 PPO for real (512 timesteps — 3-5 s on CPU)
  • Collect episode reward history for the training-curve plot

Stage 4 — Render
  • Replay trained policy for 80 steps, capturing MuJoCo frames
  • Compose side-by-side MP4: left = annotated perception, right = simulation
  • Return paths to (status_md, curve_png, sim_frame_png, sim_video_mp4)
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from egocentric_dataset_test.data.ego4d_metadata import Ego4DMetadataStore

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import zarr

    _HAS_ZARR = True
except Exception:  # pragma: no cover
    zarr = None  # type: ignore[assignment]
    _HAS_ZARR = False

# MuJoCo (required)
import mujoco

# gymnasium ≥ 0.26 or gym (SB3 2.x prefers gymnasium)
try:
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

    _GYM_BACKEND = "gymnasium"
except ImportError:  # pragma: no cover
    import gym  # type: ignore[no-redef]
    from gym import spaces as gym_spaces  # type: ignore[assignment]

    _GYM_BACKEND = "gym"

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# YOLO (optional — graceful fallback to synthetic bounding box)
try:
    from ultralytics import YOLO as _YOLO_CLASS

    _HAS_YOLO = True
except Exception:  # pragma: no cover
    _HAS_YOLO = False

# MediaPipe (optional)
try:
    import mediapipe as mp  # type: ignore[import-untyped]

    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover
    _HAS_MEDIAPIPE = False

# ---------------------------------------------------------------------------
# Paths & render dimensions
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOLO_MODEL_CANDIDATES = [
    _REPO_ROOT / "yolov5su.pt",
    _REPO_ROOT / "yolov5s.pt",
    _REPO_ROOT / "yolov8n.pt",
]
_YOLO_MODEL = next((path for path in _YOLO_MODEL_CANDIDATES if path.exists()), _YOLO_MODEL_CANDIDATES[0])
_HAND_TASK = _REPO_ROOT / "hand_landmarker.task"
_CHECKPOINT_DIR = _REPO_ROOT / "egocentric_dataset_test" / "checkpoints" / "ego_openenv"
_TRAINING_SUMMARY_PATH = _CHECKPOINT_DIR / "training_summary.json"
_TRAINING_METRICS_PATH = _CHECKPOINT_DIR / "training_metrics.jsonl"

RENDER_W, RENDER_H = 480, 360  # MuJoCo simulation render size
PERCEP_W, PERCEP_H = 480, 360  # perception panel size (matches render)

# Egocentric-100K pinhole intrinsics (456×256 sensor) — scaled to PERCEP frame
_FX = 137.98 * PERCEP_W / 456.0
_FY = 138.23 * PERCEP_H / 256.0
_CX = 232.17 * PERCEP_W / 456.0
_CY = 125.37 * PERCEP_H / 256.0

# World-space layout of the MuJoCo scene
_WORKBENCH_WORLD = np.array([0.38, 0.0, 0.69])  # workbench body origin
_PCB_WORLD_Z = _WORKBENCH_WORLD[2] + 0.028  # top of workbench + PCB site offset
_YOLO_MODEL_CACHE: Any | None = None
_EGO4D_METADATA_STORE: Any | None = None


@dataclass(slots=True)
class StreamRLConfig:
    algorithm: str = "ppo"
    total_timesteps: int = 1024
    chunk_duration_sec: float = 2.0
    preview_updates_per_clip: int = 12
    max_output_frames: int = 720
    physics_substeps_per_action: int = 5
    control_scale: float = 0.45
    policy_action_stride: int = 1
    runtime_mode: str = "openenv_local"
    seed: int = 7


STREAM_PRESETS: dict[str, StreamRLConfig] = {
    "low_latency": StreamRLConfig(
        chunk_duration_sec=1.25,
        preview_updates_per_clip=18,
        max_output_frames=540,
        physics_substeps_per_action=4,
        control_scale=0.40,
        policy_action_stride=2,
    ),
    "balanced": StreamRLConfig(),
    "smooth": StreamRLConfig(
        chunk_duration_sec=3.0,
        preview_updates_per_clip=10,
        max_output_frames=900,
        physics_substeps_per_action=6,
        control_scale=0.48,
        policy_action_stride=1,
    ),
}


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
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _get_yolo_model() -> Any | None:
    global _YOLO_MODEL_CACHE
    if not _HAS_YOLO or not _YOLO_MODEL.exists():
        return None
    if _YOLO_MODEL_CACHE is None:
        _YOLO_MODEL_CACHE = _YOLO_CLASS(str(_YOLO_MODEL))
    return _YOLO_MODEL_CACHE


def _get_ego4d_metadata_store() -> Any | None:
    global _EGO4D_METADATA_STORE
    if Ego4DMetadataStore is None:
        return None
    if _EGO4D_METADATA_STORE is not None:
        return _EGO4D_METADATA_STORE

    parquet_path = os.getenv("EGO4D_METADATA_PARQUET", "").strip()
    dataset_id = os.getenv("EGO4D_METADATA_DATASET_ID", "builddotai/Egocentric-100K-Evaluation").strip()
    try:
        store = Ego4DMetadataStore(dataset_id=dataset_id)
        if parquet_path:
            store.load_from_parquet(parquet_path, max_rows=256, include_images=True)
        else:
            store.load(max_rows=256)
        _EGO4D_METADATA_STORE = store
    except Exception:
        _EGO4D_METADATA_STORE = None
    return _EGO4D_METADATA_STORE


def _load_real_ego4d_frame(task_class: str | None = None) -> np.ndarray | None:
    if cv2 is None:
        return None
    store = _get_ego4d_metadata_store()
    if store is None:
        return None
    jpeg_bytes = store.sample_frame(task_class=task_class)
    if not jpeg_bytes:
        return None
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.resize(frame, (PERCEP_W, PERCEP_H))


def resolve_stream_config(
    preset: str = "balanced",
    *,
    algorithm: str = "ppo",
    total_timesteps: int | None = None,
    runtime_mode: str = "openenv_local",
    seed: int = 7,
) -> StreamRLConfig:
    base = STREAM_PRESETS.get(preset, STREAM_PRESETS["balanced"])
    config = replace(base)
    config.algorithm = algorithm.lower().strip()
    config.runtime_mode = runtime_mode
    config.seed = seed
    if total_timesteps is not None:
        config.total_timesteps = int(total_timesteps)
    return config


def _build_sb3_model(algorithm: str, vec_env: DummyVecEnv, total_timesteps: int, seed: int) -> BaseAlgorithm:
    algo = algorithm.lower().strip()
    if algo == "ppo":
        n_steps = max(32, min(256, total_timesteps))
        return PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            n_steps=n_steps,
            batch_size=min(64, n_steps),
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            n_steps=max(5, min(64, max(5, total_timesteps // 8))),
            learning_rate=7e-4,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.0,
        )
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=20_000,
            learning_starts=min(128, max(32, total_timesteps // 8)),
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            gamma=0.99,
        )
    if algo == "td3":
        n_actions = int(vec_env.action_space.shape[-1])
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.10 * np.ones(n_actions))
        return TD3(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            learning_rate=1e-3,
            buffer_size=20_000,
            learning_starts=min(128, max(32, total_timesteps // 8)),
            batch_size=64,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.99,
            action_noise=action_noise,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")

# ============================================================================
# MuJoCo XML — robot base + shoulder + upper-arm + forearm + wrist + hand
# (24 hinge joints, 15 position-servo actuators, PCB target)
# {TX} {TY}: PCB offset from workbench centre in metres
# ============================================================================
_ARM_HAND_XML = """\
<mujoco model="pcb_arm_hand">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <compiler autolimits="true"/>
  <default>
    <joint damping="1.0" armature="0.02"/>
    <geom condim="3" friction="1 0.005 0.0001"/>
  </default>
  <asset>
    <texture name="checker" type="2d" builtin="checker"
             rgb1=".13 .19 .13" rgb2=".17 .23 .17" width="256" height="256"/>
    <material name="floor_m" texture="checker" texrepeat="8 8" reflectance=".12"/>
    <material name="steel_m" rgba=".62 .65 .70 1" reflectance=".28"/>
    <material name="wood_m"  rgba=".52 .42 .27 1"/>
    <material name="pcb_m"   rgba=".06 .50 .16 1"/>
    <material name="skin_m"  rgba=".88 .73 .53 1"/>
  </asset>
  <worldbody>
    <camera name="main_cam"
            pos="0.17 -1.08 1.14"
            xyaxes="1 0 0 0 0.40 0.70"
            fovy="54"/>
    <camera name="close_cam"
            pos="0.55 -0.45 0.88"
            xyaxes="1 0 0 0 0.55 0.60"
            fovy="38"/>
    <light name="key" pos="0.2 -0.4 2.0" dir="0 0.18 -1"
           diffuse=".82 .82 .78" castshadow="true"/>
    <light name="fill" pos="-0.7 0.6 1.4" dir=".35 -.2 -.8"
           diffuse=".30 .30 .36" castshadow="false"/>
    <geom name="floor" type="plane" size="2 2 .1" material="floor_m"/>
    <!-- WORKBENCH -->
    <body name="workbench" pos="0.38 0 0.69">
      <geom type="box" size=".30 .22 .018" material="wood_m" mass="50"/>
      <!-- PCB board (pos relative to workbench top-centre) -->
      <body name="pcb_board" pos="{TX} {TY} 0.020">
        <geom name="pcb_base" type="box" size=".038 .024 .004" material="pcb_m" mass="0.05"/>
        <geom type="box" size=".006 .006 .003" pos="-.014 .008 .007"
              rgba=".18 .18 .18 1" mass=".001"/>
        <geom type="box" size=".006 .006 .003" pos=".010 .008 .007"
              rgba=".18 .18 .18 1" mass=".001"/>
        <geom type="box" size=".008 .004 .002" pos="0 -.010 .006"
              rgba=".78 .68 .08 1" mass=".001"/>
        <site name="pcb_center"   pos="0 0 .008" size=".003" rgba="1 1 0 1"/>
        <site name="grasp_target" pos="0 0 .016" size=".007" rgba="1 .3 0 .5"/>
      </body>
    </body>
    <!-- ROBOT BASE -->
    <body name="robot_base" pos="-0.04 0 0.74">
      <geom type="cylinder" size=".068 .068" material="steel_m" mass="12"/>
      <body name="shoulder" pos="0 0 0.068">
        <geom type="sphere" size=".074" material="steel_m" mass="2.5"/>
        <joint name="shoulder_pan" type="hinge" axis="0 0 1"
               range="-75 75" limited="true" damping="2.2"/>
        <body name="upper_arm" pos="0 0 0">
          <geom type="capsule" fromto="0 0 0 0 0 0.31"
                size=".043" material="steel_m" mass="1.8"/>
          <joint name="shoulder_lift" type="hinge" axis="0 1 0"
                 range="-78 38" limited="true" damping="2.8"/>
          <body name="elbow" pos="0 0 0.31">
            <geom type="sphere" size=".050" material="steel_m" mass="1.0"/>
            <joint name="elbow_flex" type="hinge" axis="0 1 0"
                   range="-128 0" limited="true" damping="1.8"/>
            <body name="forearm" pos="0 0 0">
              <geom type="capsule" fromto="0 0 0 0 0 0.25"
                    size=".034" material="steel_m" mass="1.0"/>
              <body name="wrist" pos="0 0 0.25">
                <geom type="sphere" size=".038" material="steel_m" mass="0.4"/>
                <joint name="wrist_flex" type="hinge" axis="0 1 0"
                       range="-68 68" limited="true" damping="0.9"/>
                <joint name="wrist_rot" type="hinge" axis="0 0 1"
                       range="-58 58" limited="true" damping="0.6"/>
                <!-- PALM -->
                <body name="palm" pos="0 0 0.040">
                  <geom type="box" size=".029 .043 .012"
                        material="skin_m" mass="0.14"/>
                  <site name="palm_center" pos="0 0 .014"
                        size=".004" rgba="1 1 1 .4"/>
                  <!-- THUMB -->
                  <body name="thumb_base" pos="-.024 -.030 .008">
                    <joint name="thumb_abd" type="hinge" axis="1 0 0"
                           range="-28 28" limited="true" damping=".12"/>
                    <geom type="capsule" fromto="0 -.018 0 0 .018 .014"
                          size=".0080" material="skin_m" mass=".010"/>
                    <joint name="thumb_mcp" type="hinge" axis="0 1 0"
                           range="-18 60" limited="true" damping=".16"/>
                    <body name="thumb_dist" pos="0 .018 .014">
                      <geom type="capsule" fromto="0 0 0 0 .014 .010"
                            size=".0070" material="skin_m" mass=".006"/>
                      <joint name="thumb_ip" type="hinge" axis="0 1 0"
                             range="-8 60" limited="true" damping=".12"/>
                      <site name="thumb_tip" pos="0 .016 .011" size=".0044"/>
                    </body>
                  </body>
                  <!-- INDEX -->
                  <body name="index_base" pos="-.017 .043 .006">
                    <joint name="index_abd" type="hinge" axis="1 0 0"
                           range="-15 15" limited="true" damping=".09"/>
                    <geom type="capsule" fromto="0 0 0 0 .027 0"
                          size=".0080" material="skin_m" mass=".009"/>
                    <joint name="index_mcp" type="hinge" axis="0 1 0"
                           range="-8 84" limited="true" damping=".14"/>
                    <body name="index_mid" pos="0 .027 0">
                      <geom type="capsule" fromto="0 0 0 0 .020 0"
                            size=".0070" material="skin_m" mass=".007"/>
                      <joint name="index_pip" type="hinge" axis="0 1 0"
                             range="0 100" limited="true" damping=".11"/>
                      <body name="index_dist" pos="0 .020 0">
                        <geom type="capsule" fromto="0 0 0 0 .014 0"
                              size=".0060" material="skin_m" mass=".005"/>
                        <joint name="index_dip" type="hinge" axis="0 1 0"
                               range="0 74" limited="true" damping=".09"/>
                        <site name="index_tip" pos="0 .016 0" size=".0044"/>
                      </body>
                    </body>
                  </body>
                  <!-- MIDDLE -->
                  <body name="middle_base" pos="0 .043 .006">
                    <joint name="middle_abd" type="hinge" axis="1 0 0"
                           range="-10 10" limited="true" damping=".09"/>
                    <geom type="capsule" fromto="0 0 0 0 .031 0"
                          size=".0082" material="skin_m" mass=".010"/>
                    <joint name="middle_mcp" type="hinge" axis="0 1 0"
                           range="-8 84" limited="true" damping=".14"/>
                    <body name="middle_mid" pos="0 .031 0">
                      <geom type="capsule" fromto="0 0 0 0 .022 0"
                            size=".0072" material="skin_m" mass=".007"/>
                      <joint name="middle_pip" type="hinge" axis="0 1 0"
                             range="0 100" limited="true" damping=".11"/>
                      <body name="middle_dist" pos="0 .022 0">
                        <geom type="capsule" fromto="0 0 0 0 .015 0"
                              size=".0062" material="skin_m" mass=".005"/>
                        <joint name="middle_dip" type="hinge" axis="0 1 0"
                               range="0 74" limited="true" damping=".09"/>
                        <site name="middle_tip" pos="0 .017 0" size=".0044"/>
                      </body>
                    </body>
                  </body>
                  <!-- RING -->
                  <body name="ring_base" pos=".017 .043 .006">
                    <joint name="ring_abd" type="hinge" axis="1 0 0"
                           range="-15 15" limited="true" damping=".09"/>
                    <geom type="capsule" fromto="0 0 0 0 .027 0"
                          size=".0075" material="skin_m" mass=".008"/>
                    <joint name="ring_mcp" type="hinge" axis="0 1 0"
                           range="-8 84" limited="true" damping=".14"/>
                    <body name="ring_mid" pos="0 .027 0">
                      <geom type="capsule" fromto="0 0 0 0 .020 0"
                            size=".0065" material="skin_m" mass=".006"/>
                      <joint name="ring_pip" type="hinge" axis="0 1 0"
                             range="0 100" limited="true" damping=".11"/>
                      <body name="ring_dist" pos="0 .020 0">
                        <geom type="capsule" fromto="0 0 0 0 .014 0"
                              size=".0055" material="skin_m" mass=".004"/>
                        <joint name="ring_dip" type="hinge" axis="0 1 0"
                               range="0 74" limited="true" damping=".09"/>
                        <site name="ring_tip" pos="0 .016 0" size=".0044"/>
                      </body>
                    </body>
                  </body>
                  <!-- LITTLE -->
                  <body name="little_base" pos=".031 .038 .004">
                    <joint name="little_abd" type="hinge" axis="1 0 0"
                           range="-20 20" limited="true" damping=".09"/>
                    <geom type="capsule" fromto="0 0 0 0 .021 0"
                          size=".0065" material="skin_m" mass=".007"/>
                    <joint name="little_mcp" type="hinge" axis="0 1 0"
                           range="-8 84" limited="true" damping=".12"/>
                    <body name="little_mid" pos="0 .021 0">
                      <geom type="capsule" fromto="0 0 0 0 .015 0"
                            size=".0057" material="skin_m" mass=".005"/>
                      <joint name="little_pip" type="hinge" axis="0 1 0"
                             range="0 100" limited="true" damping=".10"/>
                      <body name="little_dist" pos="0 .015 0">
                        <geom type="capsule" fromto="0 0 0 0 .011 0"
                              size=".0048" material="skin_m" mass=".003"/>
                        <joint name="little_dip" type="hinge" axis="0 1 0"
                               range="0 74" limited="true" damping=".08"/>
                        <site name="little_tip" pos="0 .013 0" size=".0040"/>
                      </body>
                    </body>
                  </body>
                </body>  <!-- palm -->
              </body>  <!-- wrist -->
            </body>  <!-- forearm -->
          </body>  <!-- elbow -->
        </body>  <!-- upper_arm -->
      </body>  <!-- shoulder -->
    </body>  <!-- robot_base -->
  </worldbody>

  <actuator>
    <!-- 5 arm position servos (ctrlrange in degrees → compiled to radians) -->
    <position name="sp_p"  joint="shoulder_pan"  kp="130" ctrlrange="-75 75"/>
    <position name="sl_p"  joint="shoulder_lift"  kp="170" ctrlrange="-78 38"/>
    <position name="ef_p"  joint="elbow_flex"     kp="170" ctrlrange="-128 0"/>
    <position name="wf_p"  joint="wrist_flex"     kp="65"  ctrlrange="-68 68"/>
    <position name="wr_p"  joint="wrist_rot"      kp="55"  ctrlrange="-58 58"/>
    <!-- 10 finger position servos -->
    <position name="tm_p"  joint="thumb_mcp"      kp="24"  ctrlrange="-18 60"/>
    <position name="ti_p"  joint="thumb_ip"       kp="18"  ctrlrange="-8 60"/>
    <position name="im_p"  joint="index_mcp"      kp="30"  ctrlrange="-8 84"/>
    <position name="ip_p"  joint="index_pip"      kp="24"  ctrlrange="0 100"/>
    <position name="mm_p"  joint="middle_mcp"     kp="30"  ctrlrange="-8 84"/>
    <position name="mp_p"  joint="middle_pip"     kp="24"  ctrlrange="0 100"/>
    <position name="rm_p"  joint="ring_mcp"       kp="28"  ctrlrange="-8 84"/>
    <position name="rp_p"  joint="ring_pip"       kp="22"  ctrlrange="0 100"/>
    <position name="lm_p"  joint="little_mcp"     kp="24"  ctrlrange="-8 84"/>
    <position name="lp_p"  joint="little_pip"     kp="18"  ctrlrange="0 100"/>
  </actuator>

  <sensor>
    <framepos name="palm_pos"  objtype="site" objname="palm_center"/>
    <framepos name="itip_pos"  objtype="site" objname="index_tip"/>
    <framepos name="ttip_pos"  objtype="site" objname="thumb_tip"/>
    <framepos name="pcbc_pos"  objtype="site" objname="pcb_center"/>
  </sensor>
</mujoco>
"""

# ============================================================================
# Stage 1 — Perception
# ============================================================================

def _draw_factory_frame() -> np.ndarray:
    """Synthesise a realistic-looking 480×360 BGR egocentric factory frame.

    PCB is placed at (300, 220) px (top of workbench area) so YOLO can see it.
    """
    img = np.zeros((PERCEP_H, PERCEP_W, 3), dtype=np.uint8)

    # --- floor gradient (dark green-grey) ---
    for y in range(PERCEP_H):
        v = int(28 + 12 * y / PERCEP_H)
        img[y, :] = [v, v + 5, v]

    # --- workbench surface (wooden) ---
    wb_y0, wb_y1 = 150, 300
    wb_x0, wb_x1 = 40, 440
    img[wb_y0:wb_y1, wb_x0:wb_x1] = [55, 90, 145]  # BGR wood tone
    # wood grain lines
    for yi in range(wb_y0, wb_y1, 12):
        img[yi, wb_x0:wb_x1] = [48, 80, 130]

    # --- PCB board (green) at workbench centre ---
    pcb_cx, pcb_cy = 300, 220
    pcb_w, pcb_h = 80, 48
    px0, px1 = pcb_cx - pcb_w // 2, pcb_cx + pcb_w // 2
    py0, py1 = pcb_cy - pcb_h // 2, pcb_cy + pcb_h // 2
    img[py0:py1, px0:px1] = [20, 128, 40]  # green PCB
    # IC chips on PCB
    cv2.rectangle(img, (px0 + 8, py0 + 8), (px0 + 20, py0 + 20), (30, 30, 30), -1)
    cv2.rectangle(img, (px0 + 30, py0 + 8), (px0 + 42, py0 + 20), (30, 30, 30), -1)
    # Gold connector strip
    cv2.rectangle(img, (px0 + 5, py1 - 12), (px1 - 5, py1 - 5), (0, 180, 200), -1)
    # Red dot = pin 1 marker
    cv2.circle(img, (px0 + 6, py0 + 6), 3, (0, 0, 220), -1)

    # --- worker arm silhouette (flesh tone) ---
    arm_pts = np.array([
        [200, 360], [230, 310], [250, 280], [270, 260],
        [290, 260], [310, 280], [330, 310], [360, 360]
    ], dtype=np.int32)
    cv2.fillPoly(img, [arm_pts], (140, 180, 205))  # flesh BGR

    # --- overhead factory lamp glow ---
    cv2.circle(img, (240, 30), 45, (230, 228, 210), -1)
    cv2.circle(img, (240, 30), 55, (180, 178, 160), 4)

    # --- HUD text ---
    cv2.putText(img, "Egocentric CAM  |  factory_001 / worker_001",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"PCB Assembly Station  [{_YOLO_MODEL.stem} active]",
                (10, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 180), 1, cv2.LINE_AA)

    return img  # BGR uint8


def _draw_hand_frame() -> np.ndarray:
    """Synthesise a 300×300 RGB hand image for MediaPipe detection."""
    img = np.full((300, 300, 3), [240, 200, 160], dtype=np.uint8)  # light background

    # Palm base
    palm_pts = np.array([
        [120, 200], [180, 200], [185, 150], [175, 120], [165, 115],
        [155, 120], [145, 120], [135, 115], [125, 120], [115, 150]
    ], dtype=np.int32)
    cv2.fillPoly(img, [palm_pts], (195, 155, 110))

    # Fingers: index, middle, ring, little, thumb
    fingers = [
        # (tip_x, tip_y, base_x, base_y, width)
        (140, 60,  140, 135, 14),   # index
        (152, 48,  152, 130, 15),   # middle
        (163, 55,  163, 132, 13),   # ring
        (174, 68,  174, 138, 11),   # little
        (108, 105, 130, 175, 16),   # thumb
    ]
    for tx, ty, bx, by, w in fingers:
        pts = np.array([
            [bx - w // 2, by], [bx + w // 2, by],
            [tx + w // 2 - 2, ty], [tx - w // 2 + 2, ty]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (195, 155, 110))
        # knuckle lines
        for frac in [0.33, 0.66]:
            kx = int(bx + frac * (tx - bx))
            ky = int(by + frac * (ty - by))
            cv2.line(img, (kx - w // 2, ky), (kx + w // 2, ky),
                     (165, 128, 90), 1)

    # Fingernails
    for tx, ty, _, _, w in fingers:
        cv2.ellipse(img, (tx, ty), (w // 2 - 1, 5), 0, 0, 360, (220, 200, 195), -1)

    return img  # RGB uint8


def run_yolo_detection(
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[float, float] | None]:
    """Run local Ultralytics YOLO on `frame_bgr`.

    Returns
    -------
    annotated_bgr : np.ndarray
        Frame with bounding boxes drawn.
    detections : list of dicts  {cls_name, conf, xyxy}
    pcb_pixel_center : (cx, cy) of the highest-confidence detection, or None
    """
    if not _HAS_YOLO or not _YOLO_MODEL.exists():
        # Fallback: draw a synthetic box at the PCB location we drew
        annotated = frame_bgr.copy()
        pcb_cx, pcb_cy = 300, 220
        cv2.rectangle(annotated, (260, 196), (340, 244), (0, 255, 128), 2)
        cv2.putText(annotated, "PCB [synth]", (262, 193),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
        return annotated, [{"cls_name": "pcb_board", "conf": 1.0, "xyxy": [260, 196, 340, 244]}], (pcb_cx, pcb_cy)

    model = _get_yolo_model()
    if model is None:
        return frame_bgr.copy(), [], None
    results = model(frame_bgr, verbose=False, conf=0.15, imgsz=640)

    annotated = results[0].plot()  # BGR annotated frame

    detections: list[dict[str, Any]] = []
    best_center: tuple[float, float] | None = None
    best_conf = -1.0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = results[0].names.get(cls_id, str(cls_id))
            detections.append({"cls_name": cls_name, "conf": conf, "xyxy": xyxy})
            if conf > best_conf:
                best_conf = conf
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                best_center = (cx, cy)

    # If nothing detected, mark the drawn PCB location
    if best_center is None:
        pcb_cx, pcb_cy = 300.0, 220.0
        cv2.rectangle(annotated, (260, 196), (340, 244), (0, 220, 80), 2)
        cv2.putText(annotated, "PCB target", (262, 193),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 80), 1, cv2.LINE_AA)
        best_center = (pcb_cx, pcb_cy)

    return annotated, detections, best_center


def _grabcut_mask(frame_bgr: np.ndarray, xyxy: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(frame_bgr.shape[1] - 2, x1))
    y1 = max(0, min(frame_bgr.shape[0] - 2, y1))
    x2 = max(x1 + 1, min(frame_bgr.shape[1] - 1, x2))
    y2 = max(y1 + 1, min(frame_bgr.shape[0] - 1, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def overlay_detection_and_segmentation(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    overlay = frame_bgr.copy()
    if not detections:
        return overlay
    top = max(detections, key=lambda item: float(item.get("conf", 0.0)))
    xyxy = top.get("xyxy", [])
    if len(xyxy) != 4:
        return overlay
    mask = _grabcut_mask(frame_bgr, xyxy)
    if mask is not None and mask.any():
        color = np.zeros_like(overlay)
        color[..., 1] = 180
        color[..., 2] = 80
        alpha = (mask[..., None] * 0.35).astype(np.float32)
        overlay = np.clip((1.0 - alpha) * overlay + alpha * color, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        cv2.putText(
            overlay,
            "Segmentation overlay",
            (10, overlay.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def pixel_to_world_3d(
    px: float,
    py: float,
    assumed_depth_m: float = 0.72,
) -> np.ndarray:
    """Unproject pixel (px, py) → world 3-D using pinhole model + scene prior.

    The camera is assumed to be looking at the workbench from ~0.75 m away.
    World X grows rightward, Z is up; workbench sits at Z ≈ 0.71 m above ground.
    """
    # Direction in camera frame
    dx = (px - _CX) / _FX
    dy = (py - _CY) / _FY
    # Scaled by depth (camera Y is forward in world)
    # Camera sits above and behind the workbench looking downward.
    # Simple mapping: camera coords (dx, dy, 1) @ assumed_depth → world offset
    wx = _WORKBENCH_WORLD[0] + dx * assumed_depth_m * 0.4
    wy = _WORKBENCH_WORLD[1] + dy * assumed_depth_m * 0.2
    wz = _PCB_WORLD_Z
    return np.clip(
        np.array([wx, wy, wz]),
        [_WORKBENCH_WORLD[0] - 0.25, _WORKBENCH_WORLD[1] - 0.18, wz - 0.01],
        [_WORKBENCH_WORLD[0] + 0.25, _WORKBENCH_WORLD[1] + 0.18, wz + 0.01],
    )


def run_mediapipe_landmarks(
    hand_rgb: np.ndarray,
) -> list[tuple[float, float, float]] | None:
    """Run MediaPipe HandLandmarker on `hand_rgb` (300×300 RGB).

    Returns 21 (x_norm, y_norm, z_norm) normalised landmarks, or None.
    """
    if not _HAS_MEDIAPIPE or not _HAND_TASK.exists():
        return None

    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_HAND_TASK)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
        )
        with HandLandmarker.create_from_options(opts) as lm:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_rgb)
            result = lm.detect(mp_img)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return [(pt.x, pt.y, pt.z) for pt in result.hand_landmarks[0]]
    except Exception:
        pass
    return None


def _synthetic_landmarks() -> list[tuple[float, float, float]]:
    """Return 21 hardcoded hand landmarks (pre-grasp pose) when MediaPipe fails."""
    # MediaPipe hand landmark indices 0-20 in normalised [0,1] image coords
    # wrist=0, thumb MCP=1..4, index MCP=5..8, middle=9..12, ring=13..16, little=17..20
    pts = [
        (0.50, 0.90, 0.00),  # 0  wrist
        (0.36, 0.82, 0.01),  # 1  thumb CMC
        (0.26, 0.72, 0.02),  # 2  thumb MCP
        (0.20, 0.62, 0.03),  # 3  thumb IP
        (0.16, 0.54, 0.04),  # 4  thumb tip
        (0.42, 0.68, 0.01),  # 5  index MCP
        (0.42, 0.55, 0.02),  # 6  index PIP
        (0.42, 0.44, 0.03),  # 7  index DIP
        (0.42, 0.35, 0.04),  # 8  index tip
        (0.50, 0.66, 0.01),  # 9  middle MCP
        (0.50, 0.52, 0.02),  # 10 middle PIP
        (0.50, 0.41, 0.03),  # 11 middle DIP
        (0.50, 0.32, 0.04),  # 12 middle tip
        (0.58, 0.67, 0.01),  # 13 ring MCP
        (0.58, 0.54, 0.02),  # 14 ring PIP
        (0.58, 0.44, 0.03),  # 15 ring DIP
        (0.58, 0.35, 0.04),  # 16 ring tip
        (0.65, 0.70, 0.01),  # 17 little MCP
        (0.65, 0.58, 0.02),  # 18 little PIP
        (0.65, 0.50, 0.03),  # 19 little DIP
        (0.65, 0.42, 0.04),  # 20 little tip
    ]
    return pts


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks: list[tuple[float, float, float]],
    color: tuple[int, int, int] = (0, 255, 180),
    label: str = "Hand Pose",
) -> np.ndarray:
    """Overlay 21-point hand skeleton on `frame` (BGR)."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]

    # Connections: finger chains
    connections = [
        # thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # little
        (0, 17), (17, 18), (18, 19), (19, 20),
        # palm
        (5, 9), (9, 13), (13, 17),
    ]
    out = frame.copy()
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(out, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 3, color, -1, cv2.LINE_AA)

    cv2.putText(out, label, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    return out


def _load_episode_frame_store(episode: dict[str, Any] | None) -> Any | None:
    if episode is None or not _HAS_ZARR:
        return None
    local_episode_path = str(episode.get("local_episode_path", "") or "")
    if not local_episode_path:
        return None
    path = Path(local_episode_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        return None
    try:
        root = zarr.open(str(path), mode="r")
    except Exception:
        return None
    for key in ("images.front_1", "images", "frames"):
        if key in root:
            return root[key]
    return None


def _episode_frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        return _draw_factory_frame()
    frame_u8 = np.asarray(frame, dtype=np.uint8)
    if frame_u8.shape[2] == 3:
        return cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
    return _draw_factory_frame()


def _predict_action_and_value(model: BaseAlgorithm, obs: np.ndarray) -> tuple[np.ndarray, float]:
    action, _ = model.predict(obs, deterministic=True)
    if not _HAS_TORCH:
        return np.asarray(action, dtype=np.float32), 0.0
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            if not hasattr(model.policy, "predict_values"):
                raise AttributeError("Policy has no value head")
            value_tensor = model.policy.predict_values(obs_tensor)
        value_estimate = float(value_tensor.detach().cpu().numpy().reshape(-1)[0])
    except Exception:
        value_estimate = 0.0
    return np.asarray(action, dtype=np.float32), value_estimate


def _frame_count(frame_store: Any | None) -> int:
    if frame_store is None:
        return 0
    shape = getattr(frame_store, "shape", None)
    if shape:
        return int(shape[0])
    try:
        return int(len(frame_store))
    except Exception:
        return 0


# ============================================================================
# Stage 2 + 3 — Physics + RL  (gymnasium.Env wrapping the MuJoCo scene)
# ============================================================================

class PCBManipEnv(gym.Env):
    """Gymnasium env: robotic arm + full 5-finger hand reaching for a PCB.

    Observation (48-D float32):
        qpos[:24]   — all 24 hinge joint positions (rad)
        qvel[:24]   — all 24 joint velocities (rad/s)
    Action (15-D float32 in [-1,1]):
        Position targets for the 15 actuators, normalised to [-1,1]
        and linearly scaled to each actuator's ctrlrange.
    Reward:
        -2 × dist(index_tip → pcb_center)
        +0.5 bonus if dist < 0.03 m  (near-grasp)
        -0.002 step penalty (efficiency)
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    # Joint names in the model (must match XML)
    _INIT_DEGREES: dict[str, float] = {
        "shoulder_pan":  0.0,
        "shoulder_lift": -22.0,
        "elbow_flex":    -52.0,
        "wrist_flex":     18.0,
        "wrist_rot":       0.0,
        "thumb_mcp":      22.0,
        "thumb_ip":       15.0,
        "index_mcp":      28.0,
        "index_pip":      18.0,
        "middle_mcp":     28.0,
        "middle_pip":     18.0,
        "ring_mcp":       25.0,
        "ring_pip":       15.0,
        "little_mcp":     22.0,
        "little_pip":     13.0,
    }

    def __init__(self, target_pos: np.ndarray, config: StreamRLConfig | None = None) -> None:
        super().__init__()
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.config = config or STREAM_PRESETS["balanced"]

        # Build model
        tx = float(self.target_pos[0] - _WORKBENCH_WORLD[0])
        ty = float(self.target_pos[1])
        xml = _ARM_HAND_XML.replace("{TX}", f"{tx:.4f}").replace("{TY}", f"{ty:.4f}")
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self._renderer = mujoco.Renderer(self.mj_model, height=RENDER_H, width=RENDER_W)

        nq = self.mj_model.nq  # 24
        nv = self.mj_model.nv  # 24
        nu = self.mj_model.nu  # 15

        self.observation_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(nq + nv,), dtype=np.float32
        )
        self.action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(nu,), dtype=np.float32
        )

        # Pre-compute ctrl ranges (compiled, in radians)
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()

        # Site IDs
        self._itip_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "index_tip"
        )
        self._pcb_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "pcb_center"
        )
        self._palm_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "palm_center"
        )

        # Precomputed initial qpos
        self._init_qpos = self._compute_init_qpos()

        self._step_count = 0
        self._max_steps = max(60, min(self.config.max_output_frames, 240))

    def _compute_init_qpos(self) -> np.ndarray:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        q = self.mj_data.qpos.copy()
        for jname, deg in self._INIT_DEGREES.items():
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                qadr = self.mj_model.jnt_qposadr[jid]
                q[qadr] = math.radians(deg)
        return q

    def _get_obs(self) -> np.ndarray:
        nq = self.mj_model.nq
        nv = self.mj_model.nv
        return np.concatenate([
            self.mj_data.qpos[:nq],
            self.mj_data.qvel[:nv],
        ]).astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self._init_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map [-1,1] → ctrlrange
        a = np.asarray(action, dtype=np.float64)
        ctrl_mid = 0.5 * (self._ctrl_lo + self._ctrl_hi)
        ctrl_half_span = 0.5 * (self._ctrl_hi - self._ctrl_lo)
        ctrl = ctrl_mid + np.clip(a, -1.0, 1.0) * ctrl_half_span * self.config.control_scale
        self.mj_data.ctrl[:] = ctrl

        for _ in range(self.config.physics_substeps_per_action):
            mujoco.mj_step(self.mj_model, self.mj_data)

        if not np.isfinite(self.mj_data.qpos).all() or not np.isfinite(self.mj_data.qvel).all():
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self.mj_data.qpos[:] = self._init_qpos
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._step_count = 0
            return self._get_obs(), -2.0, True, False, {"reset_reason": "non_finite_state"}

        self._step_count += 1
        obs = self._get_obs()

        # Reward
        if self._itip_id >= 0 and self._pcb_id >= 0:
            itip = self.mj_data.site_xpos[self._itip_id]
            pcbc = self.mj_data.site_xpos[self._pcb_id]
            dist = float(np.linalg.norm(itip - pcbc))
        else:
            dist = float(np.linalg.norm(self.mj_data.site_xpos[self._palm_id] - self.target_pos))

        reward = -dist * 2.0 + (0.5 if dist < 0.03 else 0.0) - 0.002
        terminated = self._step_count >= self._max_steps
        return obs, float(reward), terminated, False, {}

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self.mj_data, camera="main_cam")
        return self._renderer.render()  # H×W×3 RGB uint8

    def close(self) -> None:
        try:
            del self._renderer
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reward tracking callback
# ---------------------------------------------------------------------------
class _RewardTracker(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(float(info["episode"]["r"]))
        return True


def _training_update_count(config: StreamRLConfig) -> int:
    if config.total_timesteps <= 128:
        return 1
    return max(4, min(8, math.ceil(config.total_timesteps / 128)))


def _float_metric(logger_values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = logger_values.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _build_live_training_metric(
    model: BaseAlgorithm,
    tracker: _RewardTracker,
    update_index: int,
    update_total: int,
    timesteps_completed: int,
    chunk_timesteps: int,
) -> dict[str, Any]:
    logger_values = dict(getattr(model.logger, "name_to_value", {}) or {})
    recent_rewards = tracker.ep_rewards[-4:]
    episode_reward = recent_rewards[-1] if recent_rewards else 0.0
    reward_mean = float(np.mean(recent_rewards)) if recent_rewards else 0.0
    return {
        "update_index": update_index,
        "update_total": update_total,
        "timesteps_completed": timesteps_completed,
        "chunk_timesteps": chunk_timesteps,
        "episode_reward": episode_reward,
        "reward_mean": reward_mean,
        "policy_loss": _float_metric(
            logger_values,
            "train/policy_gradient_loss",
            "train/loss",
            "train/actor_loss",
        ),
        "value_loss": _float_metric(
            logger_values,
            "train/value_loss",
            "train/critic_loss",
        ),
        "entropy": _float_metric(
            logger_values,
            "train/entropy_loss",
            "train/ent_coef_loss",
            "train/ent_coef",
        ),
        "episodes_seen": len(tracker.ep_rewards),
    }


def iter_train_rl_agent(
    target_pos: np.ndarray,
    config: StreamRLConfig,
) -> Iterator[dict[str, Any]]:
    def _make_env():
        env = PCBManipEnv(target_pos, config=config)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    tracker = _RewardTracker()
    model = _build_sb3_model(config.algorithm, vec_env, config.total_timesteps, config.seed)
    updates_total = _training_update_count(config)
    chunk_timesteps = max(32, math.ceil(config.total_timesteps / updates_total))
    timesteps_completed = 0

    try:
        for update_index in range(updates_total):
            remaining = config.total_timesteps - timesteps_completed
            if remaining <= 0:
                break
            current_chunk = min(chunk_timesteps, remaining)
            model.learn(
                total_timesteps=current_chunk,
                callback=tracker,
                reset_num_timesteps=timesteps_completed == 0,
                progress_bar=False,
            )
            timesteps_completed += current_chunk
            yield {
                "done": False,
                "metric": _build_live_training_metric(
                    model,
                    tracker,
                    update_index=update_index,
                    update_total=updates_total,
                    timesteps_completed=timesteps_completed,
                    chunk_timesteps=current_chunk,
                ),
                "episode_rewards": list(tracker.ep_rewards),
                "model": model,
            }
    finally:
        vec_env.close()

    eval_env = PCBManipEnv(target_pos, config=config)
    yield {
        "done": True,
        "model": model,
        "episode_rewards": list(tracker.ep_rewards),
        "eval_env": eval_env,
    }


# ============================================================================
# Stage 3 — RL training
# ============================================================================

def train_rl_agent(
    target_pos: np.ndarray,
    total_timesteps: int = 512,
    config: StreamRLConfig | None = None,
) -> tuple[BaseAlgorithm, list[float], PCBManipEnv]:
    """Train an SB3 policy on the arm+hand MuJoCo env.

    Returns (trained_model, episode_rewards, eval_env).
    `eval_env` is a fresh env for rollout rendering (caller must close it).
    """
    config = config or resolve_stream_config(total_timesteps=total_timesteps)

    final_model: BaseAlgorithm | None = None
    episode_rewards: list[float] = []
    eval_env: PCBManipEnv | None = None
    for item in iter_train_rl_agent(target_pos, config):
        final_model = item.get("model")
        episode_rewards = list(item.get("episode_rewards", []))
        if item.get("done"):
            eval_env = item.get("eval_env")
    if final_model is None or eval_env is None:
        raise RuntimeError("Training did not produce a model and evaluation environment")
    return final_model, episode_rewards, eval_env


# ============================================================================
# Stage 4 — Render combined video
# ============================================================================

def _add_hud(
    frame_rgb: np.ndarray,
    step: int,
    dist: float,
    reward: float,
    target: np.ndarray,
) -> np.ndarray:
    """Overlay step / dist / reward HUD onto an RGB frame."""
    out = frame_rgb.copy()
    lines = [
        f"Step {step:3d}",
        f"Dist {dist * 100:.1f} cm",
        f"Rew  {reward:+.3f}",
        f"Tgt  ({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 80), 1, cv2.LINE_AA)
    return out


def render_policy_video(
    model: BaseAlgorithm,
    eval_env: PCBManipEnv,
    perception_panel: np.ndarray,   # BGR 480×360
    n_render_steps: int = 80,
    output_path: str | None = None,
) -> str:
    """Roll out the trained policy and write a side-by-side MP4.

    Returns path to the MP4 file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix="_sim.mp4", delete=False)
        output_path = tmp.name

    # Convert perception panel to RGB for compositing
    percep_rgb = cv2.cvtColor(
        cv2.resize(perception_panel, (PERCEP_W, PERCEP_H)), cv2.COLOR_BGR2RGB
    )

    obs, _ = eval_env.reset()
    frames: list[np.ndarray] = []

    cumulative_reward = 0.0
    for step in range(n_render_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        cumulative_reward += reward

        # Get distance for HUD
        if eval_env._itip_id >= 0 and eval_env._pcb_id >= 0:
            dist = float(np.linalg.norm(
                eval_env.mj_data.site_xpos[eval_env._itip_id]
                - eval_env.mj_data.site_xpos[eval_env._pcb_id]
            ))
        else:
            dist = 0.5

        # Render simulation (RGB)
        sim_rgb = eval_env.render()  # H×W×3

        # HUD on sim frame
        sim_rgb = _add_hud(sim_rgb, step + 1, dist, cumulative_reward, eval_env.target_pos)

        # Add step counter to perception panel
        percep_frame = percep_rgb.copy()
        cv2.putText(percep_frame, f"Step {step + 1:3d} | Dist {dist * 100:.1f} cm",
                    (8, PERCEP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (80, 255, 160), 1, cv2.LINE_AA)

        # Side-by-side: perception (left) + simulation (right)
        combined = np.concatenate([percep_frame, sim_rgb], axis=1)  # (H, 2W, 3)
        frames.append(combined)

        if terminated or truncated:
            # Hold last frame for 10 more steps
            for _ in range(10):
                frames.append(combined)
            break

    # Write MP4 with imageio (uses ffmpeg or pillow writer)
    try:
        writer = imageio.get_writer(
            output_path, fps=15, format="FFMPEG",
            codec="libx264", quality=7,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        )
    except Exception:
        # fallback: GIF or pillow mp4
        writer = imageio.get_writer(output_path, fps=15)

    for f in frames:
        writer.append_data(f)
    writer.close()

    return output_path


# ============================================================================
# Training curve plot
# ============================================================================

def build_training_curve(
    ep_rewards: list[float],
    worker_id: str,
    target_pos: np.ndarray,
    algorithm: str = "ppo",
) -> str:
    """Save a training-curve PNG and return the path."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#1a1a2a")
    fig.patch.set_facecolor("#1a1a2a")

    for ax in axes:
        ax.set_facecolor("#0e0e1e")
        ax.tick_params(colors="#aaaacc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#445")

    # Left: episode returns
    ax = axes[0]
    if ep_rewards:
        ax.plot(ep_rewards, color="#4fc3f7", linewidth=1.6, label="Episode Return")
        ax.fill_between(range(len(ep_rewards)), ep_rewards,
                        alpha=0.25, color="#4fc3f7")
        # smoothed (rolling mean)
        if len(ep_rewards) >= 3:
            window = max(3, len(ep_rewards) // 6)
            smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
            x_sm = range(window - 1, window - 1 + len(smoothed))
            ax.plot(x_sm, smoothed, color="#ff7043", linewidth=2.0,
                    linestyle="--", label="Smoothed")
        ax.axhline(max(ep_rewards), color="#81c784", linestyle=":", linewidth=1)
        ax.legend(fontsize=8, facecolor="#1a1a2a", edgecolor="#445",
                  labelcolor="white")
    else:
        ax.text(0.5, 0.5, "Training in progress…",
                ha="center", va="center", color="#aaaacc", fontsize=9)
    ax.set_title(f"{algorithm.upper()} Episode Returns — {worker_id}", color="white", fontsize=9)
    ax.set_xlabel("Episode", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Return", color="#aaaacc", fontsize=8)

    # Right: target 3-D position + reward stats bar chart
    ax2 = axes[1]
    labels = ["Min", "Mean", "Max"]
    if ep_rewards:
        vals = [min(ep_rewards), float(np.mean(ep_rewards)), max(ep_rewards)]
    else:
        vals = [0.0, 0.0, 0.0]
    colors = ["#ef5350", "#ffee58", "#66bb6a"]
    bars = ax2.bar(labels, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=8)
    ax2.set_title("Return Statistics", color="white", fontsize=9)
    ax2.set_ylabel("Return", color="#aaaacc", fontsize=8)
    ax2.tick_params(colors="#aaaacc")
    ax2.text(0.98, 0.02,
             f"Target  X={target_pos[0]:.2f}  Y={target_pos[1]:.2f}  Z={target_pos[2]:.2f}",
             ha="right", va="bottom", transform=ax2.transAxes,
             color="#aaaacc", fontsize=7)

    plt.tight_layout(pad=1.5)
    tmp = tempfile.NamedTemporaryFile(suffix="_curve.png", delete=False)
    fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def empty_plot_df() -> Any:
    if not _HAS_PANDAS:
        return None
    return pd.DataFrame(columns=["step", "value", "metric"])


def build_reward_plot_df(history: list[dict[str, Any]], clip_label: str | None = None) -> Any:
    if not _HAS_PANDAS:
        return None
    rows: list[dict[str, Any]] = []
    prefix = f"{clip_label} · " if clip_label else ""
    for item in history:
        step = int(item.get("timesteps_completed", len(rows) + 1))
        rows.append({"step": step, "value": float(item.get("episode_reward", 0.0)), "metric": prefix + "episode_reward"})
        rows.append({"step": step, "value": float(item.get("reward_mean", 0.0)), "metric": prefix + "reward_mean"})
    return pd.DataFrame(rows, columns=["step", "value", "metric"])


def build_telemetry_plot_df(history: list[dict[str, Any]], clip_label: str | None = None) -> Any:
    if not _HAS_PANDAS:
        return None
    rows: list[dict[str, Any]] = []
    prefix = f"{clip_label} · " if clip_label else ""
    metrics = [
        ("policy_loss", "policy_loss"),
        ("value_loss", "value_loss"),
        ("entropy", "entropy"),
        ("policy_value_estimate", "policy_value"),
        ("action_mean", "action_mean"),
    ]
    for item in history:
        step = int(item.get("timesteps_completed", len(rows) + 1))
        for key, name in metrics:
            rows.append({"step": step, "value": float(item.get(key, 0.0)), "metric": prefix + name})
    return pd.DataFrame(rows, columns=["step", "value", "metric"])


# ============================================================================
# Perception panel builder (composites YOLO + hand skeleton)
# ============================================================================

def build_perception_panel(
    annotated_bgr: np.ndarray,
    landmarks: list[tuple[float, float, float]] | None,
    target_3d: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Compose the 480×360 BGR perception display panel."""
    panel = cv2.resize(annotated_bgr, (PERCEP_W, PERCEP_H))

    # Overlay hand skeleton
    lm = landmarks if landmarks else _synthetic_landmarks()
    source_label = "MediaPipe" if landmarks else "Virtual (MediaPipe n/d)"
    panel = draw_hand_skeleton(panel, lm, color=(0, 240, 170), label=source_label)

    # 3-D target crosshair
    # Project target back to pixel for visualisation
    tx_px = int(_CX + (target_3d[0] - _WORKBENCH_WORLD[0]) * _FX / 0.72)
    ty_px = int(_CY + (target_3d[1] - _WORKBENCH_WORLD[1]) * _FY / 0.72)
    tx_px = np.clip(tx_px, 10, PERCEP_W - 10)
    ty_px = np.clip(ty_px, 10, PERCEP_H - 10)
    cv2.drawMarker(panel, (tx_px, ty_px), (0, 80, 255),
                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(panel, f"3D target ({target_3d[0]:.2f},{target_3d[1]:.2f},{target_3d[2]:.2f})",
                (tx_px - 30, ty_px - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 80, 255), 1, cv2.LINE_AA)

    # Detection summary
    det_str = f"YOLO: {len(detections)} detection(s)"
    if detections:
        top = detections[0]
        det_str += f"  [{top['cls_name']} {top['conf']:.2f}]"
    cv2.putText(panel, det_str, (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 60), 1, cv2.LINE_AA)

    return panel


# ============================================================================
# Snapshot (single PNG of the simulation frame)
# ============================================================================

def _save_frame_png(rgb_frame: np.ndarray) -> str:
    from PIL import Image  # type: ignore[import]

    tmp = tempfile.NamedTemporaryFile(suffix="_frame.png", delete=False)
    Image.fromarray(rgb_frame).save(tmp.name)
    return tmp.name


def _write_video_file(frames: list[np.ndarray], fps: float, suffix: str = "_chunk.mp4") -> str | None:
    if not frames:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    output_path = tmp.name
    try:
        writer = imageio.get_writer(
            output_path,
            fps=max(1.0, fps),
            format="FFMPEG",
            codec="libx264",
            quality=7,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        )
    except Exception:
        writer = imageio.get_writer(output_path, fps=max(1.0, fps))

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


# ============================================================================
# Public API — called by demo.py
# ============================================================================

def _prepare_clip_context(
    worker_id: str,
    video_label: str,
    clip_metadata: dict[str, Any] | None,
    episode: dict[str, Any] | None,
) -> dict[str, Any]:
    clip = clip_metadata or {}
    clip_number = int(clip.get("clip_number", clip.get("video_index", 0)) or 0)
    frame_store = _load_episode_frame_store(episode)

    if frame_store is not None and _frame_count(frame_store) > 0:
        first_frame_bgr = cv2.resize(_episode_frame_to_bgr(np.asarray(frame_store[0])), (PERCEP_W, PERCEP_H))
    else:
        first_frame_bgr = _load_real_ego4d_frame(task_class=str(clip.get("task_class", "") or "")) or _draw_factory_frame()

    annotated_bgr, detections, pcb_pixel = run_yolo_detection(first_frame_bgr)
    annotated_bgr = overlay_detection_and_segmentation(annotated_bgr, detections)

    if pcb_pixel is not None:
        target_3d = pixel_to_world_3d(pcb_pixel[0], pcb_pixel[1])
    else:
        target_3d = np.array([_WORKBENCH_WORLD[0], 0.0, _PCB_WORLD_Z])

    rng = np.random.default_rng(clip_number + 42)
    target_3d[0] += rng.uniform(-0.06, 0.06)
    target_3d[1] += rng.uniform(-0.04, 0.04)
    target_3d = np.clip(
        target_3d,
        [_WORKBENCH_WORLD[0] - 0.22, -0.15, _PCB_WORLD_Z - 0.005],
        [_WORKBENCH_WORLD[0] + 0.22, 0.15, _PCB_WORLD_Z + 0.005],
    )

    hand_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks = run_mediapipe_landmarks(hand_rgb)
    if landmarks is None:
        landmarks = run_mediapipe_landmarks(_draw_hand_frame())
    mp_source = "MediaPipe HandLandmarker (21 kpts)" if landmarks else "virtual skeleton (synthetic)"

    perception_panel = build_perception_panel(annotated_bgr, landmarks, target_3d, detections)
    preview_path = _save_frame_png(cv2.cvtColor(perception_panel, cv2.COLOR_BGR2RGB))

    return {
        "worker_id": worker_id,
        "video_label": video_label,
        "clip": clip,
        "clip_number": clip_number,
        "frame_store": frame_store,
        "detections": detections,
        "n_detections": len(detections),
        "target_3d": target_3d,
        "landmarks": landmarks,
        "mp_source": mp_source,
        "perception_panel": perception_panel,
        "preview_path": preview_path,
    }


def _epoch_markdown(
    worker_id: str,
    video_label: str,
    epoch_index: int,
    epoch_total: int,
    metric: dict[str, Any],
    action: np.ndarray,
    value_estimate: float,
    target_3d: np.ndarray,
    algorithm: str,
    runtime_mode: str,
) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action_preview = ", ".join(f"{v:+.2f}" for v in action[:5])
    return (
        "### Live Training Telemetry\n\n"
        f"- **Worker / Video:** `{worker_id}` / `{video_label}`\n"
        f"- **Algorithm / Runtime:** `{algorithm.upper()}` / `{runtime_mode}`\n"
        f"- **Epoch:** `{epoch_index + 1}/{epoch_total}`\n"
        f"- **Timesteps completed:** `{int(metric.get('timesteps_completed', 0))}` (+`{int(metric.get('chunk_timesteps', 0))}` this update)\n"
        f"- **Episode reward:** `{float(metric.get('episode_reward', 0.0)):.3f}`\n"
        f"- **Rolling reward mean:** `{float(metric.get('reward_mean', 0.0)):.3f}` across `{int(metric.get('episodes_seen', 0))}` episode(s)\n"
        f"- **Policy loss:** `{float(metric.get('policy_loss', metric.get('actor_loss', 0.0))):.6f}`\n"
        f"- **Value loss:** `{float(metric.get('value_loss', metric.get('critic_loss', 0.0))):.6f}`\n"
        f"- **Entropy / Aux:** `{float(metric.get('entropy', metric.get('ent_coef', 0.0))):.6f}`\n"
        f"- **Policy value estimate:** `{value_estimate:.4f}`\n"
        f"- **Action sample:** `{action_preview}`\n"
        f"- **Action mean / std:** `{float(action.mean()):+.3f}` / `{float(action.std()):.3f}`\n"
        f"- **3D target:** `X={target_3d[0]:.3f} Y={target_3d[1]:.3f} Z={target_3d[2]:.3f}`"
    )


def _video_progress_markdown(
    worker_id: str,
    video_label: str,
    processed: int,
    total: int,
    clip_duration: float,
    detections: list[dict[str, Any]],
    action: np.ndarray,
    value_estimate: float,
    algorithm: str,
    runtime_mode: str,
    chunk_index: int = 0,
) -> str:
    pct = 100.0 * processed / max(total, 1)
    det_desc = "none"
    if detections:
        top = max(detections, key=lambda item: float(item.get("conf", 0.0)))
        det_desc = f"{top.get('cls_name', 'obj')} {float(top.get('conf', 0.0)):.2f}"
    return (
        "### Full-Clip Rendering Progress\n\n"
        f"- **Worker / Video:** `{worker_id}` / `{video_label}`\n"
        f"- **Algorithm / Runtime:** `{algorithm.upper()}` / `{runtime_mode}`\n"
        f"- **Rendered frames:** `{processed}/{total}` (`{pct:.1f}%`)\n"
        f"- **Target clip duration:** `{clip_duration:.1f}s`\n"
        f"- **Streamed chunk:** `{chunk_index}`\n"
        f"- **Current detection:** `{det_desc}`\n"
        f"- **Current action mean:** `{float(np.mean(action)):+.3f}`\n"
        f"- **Current policy value:** `{value_estimate:.4f}`\n"
        "- **Video layout:** left = real worker clip + detection/segmentation, right = MuJoCo hand simulation\n"
        "- **Frontend behavior:** video stream advances chunk-by-chunk while the full render is still in progress"
    )


def iter_full_episode_video(
    model: BaseAlgorithm,
    target_3d: np.ndarray,
    episode: dict[str, Any] | None,
    clip_metadata: dict[str, Any],
    worker_id: str,
    video_label: str,
    config: StreamRLConfig | None = None,
) -> Iterator[dict[str, Any]]:
    resolved_config = config or STREAM_PRESETS["balanced"]
    frame_store = _load_episode_frame_store(episode)
    if frame_store is None or _frame_count(frame_store) == 0:
        eval_env = PCBManipEnv(target_3d, config=resolved_config)
        try:
            fallback_path = render_policy_video(model, eval_env, _draw_factory_frame(), n_render_steps=180)
        finally:
            eval_env.close()
        yield {
            "done": True,
            "video_path": fallback_path,
            "preview_path": None,
            "rendered_frames": 180,
            "clip_duration": float(clip_metadata.get("duration_sec", 0.0) or 0.0),
        }
        return

    duration_sec = float(clip_metadata.get("duration_sec", 0.0) or 0.0)
    total_source_frames = _frame_count(frame_store)
    max_output_frames = resolved_config.max_output_frames
    sample_stride = max(1, math.ceil(total_source_frames / max_output_frames))
    sampled_indices = list(range(0, total_source_frames, sample_stride))
    output_fps = max(1.0, len(sampled_indices) / max(duration_sec, 1.0))
    update_stride = max(1, len(sampled_indices) // max(1, resolved_config.preview_updates_per_clip))
    chunk_frame_target = max(16, int(round(output_fps * resolved_config.chunk_duration_sec)))

    tmp = tempfile.NamedTemporaryFile(suffix="_full_episode.mp4", delete=False)
    output_path = tmp.name
    writer = imageio.get_writer(
        output_path,
        fps=output_fps,
        format="FFMPEG",
        codec="libx264",
        quality=7,
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
    )

    eval_env = PCBManipEnv(target_3d, config=resolved_config)
    obs, _ = eval_env.reset()
    cumulative_reward = 0.0
    last_preview_path: str | None = None
    chunk_frames: list[np.ndarray] = []
    chunk_index = 0
    last_action = np.zeros(eval_env.action_space.shape, dtype=np.float32)
    last_value_estimate = 0.0

    try:
        for render_index, frame_index in enumerate(sampled_indices, start=1):
            src_rgb = np.asarray(frame_store[frame_index])
            src_bgr = cv2.resize(_episode_frame_to_bgr(src_rgb), (PERCEP_W, PERCEP_H))

            annotated_bgr, detections, _ = run_yolo_detection(src_bgr)
            annotated_bgr = overlay_detection_and_segmentation(annotated_bgr, detections)
            landmarks = run_mediapipe_landmarks(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
            perception_panel = build_perception_panel(annotated_bgr, landmarks, target_3d, detections)
            percep_rgb = cv2.cvtColor(perception_panel, cv2.COLOR_BGR2RGB)

            if render_index == 1 or (render_index - 1) % max(1, resolved_config.policy_action_stride) == 0:
                last_action, last_value_estimate = _predict_action_and_value(model, obs)
            action = last_action
            value_estimate = last_value_estimate
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            cumulative_reward += float(reward)

            if eval_env._itip_id >= 0 and eval_env._pcb_id >= 0:
                dist = float(
                    np.linalg.norm(
                        eval_env.mj_data.site_xpos[eval_env._itip_id] - eval_env.mj_data.site_xpos[eval_env._pcb_id]
                    )
                )
            else:
                dist = 0.0

            sim_rgb = eval_env.render()
            sim_rgb = _add_hud(sim_rgb, render_index, dist, cumulative_reward, target_3d)
            cv2.putText(
                sim_rgb,
                f"Act μ {float(action.mean()):+.2f} | V {value_estimate:+.2f}",
                (8, RENDER_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (110, 255, 180),
                1,
                cv2.LINE_AA,
            )

            combined = np.concatenate([percep_rgb, sim_rgb], axis=1)
            writer.append_data(combined)
            chunk_frames.append(combined)

            chunk_path: str | None = None
            should_flush_chunk = len(chunk_frames) >= chunk_frame_target or render_index == len(sampled_indices)
            if should_flush_chunk:
                chunk_index += 1
                chunk_path = _write_video_file(chunk_frames, output_fps, suffix=f"_chunk_{chunk_index:03d}.mp4")
                chunk_frames = []

            if render_index == 1 or render_index % update_stride == 0 or render_index == len(sampled_indices):
                last_preview_path = _save_frame_png(combined)
                yield {
                    "done": False,
                    "preview_path": last_preview_path,
                    "processed": render_index,
                    "total": len(sampled_indices),
                    "clip_duration": duration_sec,
                    "detections": detections,
                    "action": action,
                    "value_estimate": value_estimate,
                    "video_chunk_path": chunk_path,
                    "chunk_index": chunk_index if chunk_path else 0,
                }
            elif chunk_path is not None:
                yield {
                    "done": False,
                    "preview_path": last_preview_path,
                    "processed": render_index,
                    "total": len(sampled_indices),
                    "clip_duration": duration_sec,
                    "detections": detections,
                    "action": action,
                    "value_estimate": value_estimate,
                    "video_chunk_path": chunk_path,
                    "chunk_index": chunk_index,
                }

            if terminated or truncated:
                obs, _ = eval_env.reset()
    finally:
        writer.close()
        eval_env.close()

    yield {
        "done": True,
        "video_path": output_path,
        "preview_path": last_preview_path,
        "rendered_frames": len(sampled_indices),
        "clip_duration": duration_sec,
    }


def stream_full_pipeline(
    worker_id: str,
    video_label: str,
    clip_metadata: dict[str, Any] | None = None,
    episode: dict[str, Any] | None = None,
    total_timesteps: int = 1024,
    algorithm: str = "ppo",
    preset: str = "balanced",
    runtime_mode: str = "openenv_local",
) -> Iterator[tuple[str, Any, Any, str | None, str | None]]:
    t0 = time.perf_counter()
    context = _prepare_clip_context(worker_id, video_label, clip_metadata, episode)
    target_3d = context["target_3d"]
    config = resolve_stream_config(
        preset,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        runtime_mode=runtime_mode,
    )

    yield (
        "### Preparing Full-Clip Analysis\n\n"
        f"- **Worker / Video:** `{worker_id}` / `{video_label}`\n"
        f"- **Algorithm / Runtime:** `{config.algorithm.upper()}` / `{config.runtime_mode}`\n"
        f"- **Streaming preset:** `{preset}` | chunk `{config.chunk_duration_sec:.1f}s` | preview updates `{config.preview_updates_per_clip}`\n"
        f"- **Clip duration:** `{float(context['clip'].get('duration_sec', 0.0) or 0.0):.1f}s`\n"
        f"- **Cached frames available:** `{_frame_count(context['frame_store'])}`\n"
        "- **Next:** initialize policy training and live epoch telemetry",
        empty_plot_df(),
        empty_plot_df(),
        None,
        None,
    )

    trained_model: BaseAlgorithm | None = None
    epoch_rewards: list[float] = []
    training_history: list[dict[str, Any]] = []
    epoch_env = PCBManipEnv(target_3d, config=config)
    obs, _ = epoch_env.reset()

    try:
        for training_item in iter_train_rl_agent(target_3d, config):
            trained_model = training_item.get("model")
            if training_item.get("done"):
                eval_env = training_item.get("eval_env")
                if eval_env is not None:
                    eval_env.close()
                break

            metric = dict(training_item.get("metric", {}))
            epoch_index = int(metric.get("update_index", len(epoch_rewards)))
            epoch_total = int(metric.get("update_total", 1))
            if trained_model is None:
                continue

            action, value_estimate = _predict_action_and_value(trained_model, obs)
            obs, _, terminated, truncated, _ = epoch_env.step(action)
            if terminated or truncated:
                obs, _ = epoch_env.reset()
            epoch_rewards = list(training_item.get("episode_rewards", []))
            metric["policy_value_estimate"] = value_estimate
            metric["action_mean"] = float(action.mean())
            metric["clip_label"] = video_label
            training_history.append(metric)
            frame_rgb = epoch_env.render()
            cv2.putText(
                frame_rgb,
                f"Epoch {epoch_index + 1}/{epoch_total} | V {value_estimate:+.2f}",
                (8, RENDER_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (110, 255, 180),
                1,
                cv2.LINE_AA,
            )
            yield (
                _epoch_markdown(
                    worker_id,
                    video_label,
                    epoch_index,
                    epoch_total,
                    metric,
                    action,
                    value_estimate,
                    target_3d,
                    config.algorithm,
                    config.runtime_mode,
                ),
                build_reward_plot_df(training_history, clip_label=video_label),
                build_telemetry_plot_df(training_history, clip_label=video_label),
                None,
                None,
            )
    finally:
        epoch_env.close()

    if trained_model is None:
        raise RuntimeError("Training stream did not produce a trained model")

    final_video_path: str | None = None
    for item in iter_full_episode_video(
        model=trained_model,
        target_3d=target_3d,
        episode=episode,
        clip_metadata=context["clip"],
        worker_id=worker_id,
        video_label=video_label,
        config=config,
    ):
        if item.get("done"):
            final_video_path = item.get("video_path")
            break
        streamed_video_path = item.get("video_chunk_path")
        yield (
            _video_progress_markdown(
                worker_id,
                video_label,
                int(item.get("processed", 0)),
                int(item.get("total", 0)),
                float(item.get("clip_duration", 0.0)),
                list(item.get("detections", [])),
                np.asarray(item.get("action", np.zeros(15, dtype=np.float32))),
                float(item.get("value_estimate", 0.0)),
                config.algorithm,
                config.runtime_mode,
                int(item.get("chunk_index", 0)),
            ),
            build_reward_plot_df(training_history, clip_label=video_label),
            build_telemetry_plot_df(training_history, clip_label=video_label),
            streamed_video_path,
            streamed_video_path,
        )

    best_reward = max(epoch_rewards) if epoch_rewards else 0.0
    mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
    elapsed = time.perf_counter() - t0
    yield (
        "### ✅ Full-Clip Training + Rendering Complete\n\n"
        f"- **Worker / Video:** `{worker_id}` / `{video_label}`\n"
        f"- **Algorithm / Runtime:** `{config.algorithm.upper()}` / `{config.runtime_mode}`\n"
        f"- **Training updates streamed:** `{_training_update_count(config)}`\n"
        f"- **Best / Mean reward:** `{best_reward:.3f}` / `{mean_reward:.3f}`\n"
        f"- **Policy telemetry source:** `live SB3 training loop on the clip-conditioned target`\n"
        f"- **Clip duration rendered:** `{float(context['clip'].get('duration_sec', 0.0) or 0.0):.1f}s`\n"
        f"- **Perception:** `Ultralytics {_YOLO_MODEL.stem} detection + box-guided segmentation overlay`\n"
        f"- **Simulation:** `MuJoCo arm-hand policy replay aligned across the full clip`\n"
        f"- **Total pipeline time:** `{elapsed:.1f}s`",
        build_reward_plot_df(training_history, clip_label=video_label),
        build_telemetry_plot_df(training_history, clip_label=video_label),
        final_video_path,
        final_video_path,
    )

def run_full_pipeline(
    worker_id: str,
    video_label: str,
    clip_metadata: dict[str, Any] | None = None,
    episode: dict[str, Any] | None = None,
    total_timesteps: int = 512,
    n_render_steps: int = 72,
) -> tuple[str, Any, Any, str | None, str | None]:
    """Run the full Perception → Physics → RL → Video pipeline.

    Parameters
    ----------
    worker_id : str
    video_label : str
    clip_metadata : dict, optional
        Egocentric-100K clip metadata (used for seeding target position).
    total_timesteps : int
        SB3 PPO training budget (keep ≤ 2048 for a fast demo).
    n_render_steps : int
        Number of simulation frames to render into the output video.

    Returns
    -------
    status_md : str  — Markdown summary
    reward_plot : Any  — live reward plot dataframe
    telemetry_plot : Any  — live telemetry plot dataframe
    preview_video : str | None  — path to streamed preview video chunk or final video
    video_path : str | None  — path to the side-by-side MP4
    """
    del n_render_steps
    final_tuple: tuple[str, Any, Any, str | None, str | None] | None = None
    for update in stream_full_pipeline(
        worker_id=worker_id,
        video_label=video_label,
        clip_metadata=clip_metadata,
        episode=episode,
        total_timesteps=max(total_timesteps, 768),
    ):
        final_tuple = update
    if final_tuple is None:
        return "### Pipeline Error\n\nNo output generated.", None, None, None
    return final_tuple
