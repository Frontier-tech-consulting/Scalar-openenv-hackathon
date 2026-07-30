"""
MyoFinger / MyoHand MuJoCo 3D Simulation Backend
=================================================

Provides a real MuJoCo physics simulation grounded in the MyoSuite musculoskeletal
models (MyoHub/myo_sim). Falls back gracefully to a lightweight procedural model
when MuJoCo or myosuite assets are unavailable.

Models:
  - MyoFinger: 4 DoF, 5 muscle-tendon actuators (extn, adabR, adabL, mflx, dflx)
  - MyoHand:  23 DoF, 39 muscle-tendon actuators

Tasks (mapped to competition spec):
  - easy_finger_reach:  MyoFinger tip reach to target
  - medium_hand_grasp:  MyoHand object hold / grasp
  - hard_precision_assembly: MyoHand key turn / precision insertion
"""

from __future__ import annotations

import io
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# MuJoCo import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import mujoco
    from mujoco import MjModel, MjData
    HAS_MUJOCO = True
except Exception:
    mujoco = None  # type: ignore[assignment]
    MjModel = None  # type: ignore[assignment,misc]
    MjData = None  # type: ignore[assignment,misc]
    HAS_MUJOCO = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MYOFINGER_JOINT_NAMES = ["IFadb", "IFmcp", "IFpip", "IFdip"]
MYOFINGER_MUSCLE_NAMES = ["extn", "adabR", "adabL", "mflx", "dflx"]
MYOFINGER_JOINT_RANGES = {
    "IFadb": (-25.0, 25.0),
    "IFmcp": (-25.0, 60.0),
    "IFpip": (0.0, 100.0),
    "IFdip": (0.0, 80.0),
}

MYOHAND_JOINT_NAMES = [
    "wrist_flex", "wrist_dev",
    "thumb_flex", "thumb_abd", "thumb_rot",
    "index_flex", "index_abd",
    "middle_flex", "middle_abd",
    "ring_flex", "ring_abd",
    "little_flex", "little_abd",
    # Additional DoF for MCP/PIP/DIP per finger
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
]
MYOHAND_MUSCLE_NAMES = [
    "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ",
    "EIP", "EPL", "EPB", "FPL", "APL", "OP",
    "FDS2", "FDS3", "FDS4", "FDS5",
    "FDP2", "FDP3", "FDP4", "FDP5",
    "EDC2", "EDC3", "EDC4", "EDC5",
    "EDM", "RI2", "RI3", "RI4", "RI5",
    "LU2", "LU3", "LU4", "LU5",
    "UI2", "UI3", "UI4", "UI5",
]

RENDER_WIDTH = 480
RENDER_HEIGHT = 360


# ---------------------------------------------------------------------------
# Minimal MuJoCo XML models (self-contained, no external assets needed)
# ---------------------------------------------------------------------------
MYOFINGER_XML = """<mujoco model="myofinger_openenv">
  <option gravity="0 0 -9.81" timestep="0.01"/>

  <default>
    <geom rgba=".9 .8 .6 1" size="0.01"/>
    <joint type="hinge" axis="0 1 0" damping=".5" armature="0.01" limited="true"/>
    <site type="sphere" rgba=".9 .9 .9 1" size="0.005"/>
  </default>

  <visual>
    <headlight diffuse=".9 .9 .9"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .2" rgb2=".25 .35 .25" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom type="plane" size="1 1 1" material="grid"/>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse=".8 .8 .8"/>

    <!-- Target sphere (green) -->
    <geom name="target" type="sphere" pos="0.25 0.01 0.25" size="0.02" rgba="0 1 0 .3" contype="0" conaffinity="0"/>

    <!-- BASE -->
    <site name="bs3" pos="-0.02 0.00 0.33"/>
    <site name="bs2" pos="-0.02 0.02 0.30"/>
    <site name="bs1" pos="-0.02 -.02 0.30"/>
    <site name="bs0" pos="-0.02 0.00 0.28"/>

    <!-- PROXIMAL -->
    <body name="proximal">
      <geom type="capsule" fromto="0 0 0.3 0.1 0 0.3" size=".018" rgba=".85 .75 .55 1"/>
      <geom name="pg1" type="sphere" pos="0.0 0.0 0.3" size="0.024" rgba=".5 .5 .9 .4" contype="0" conaffinity="0"/>
      <joint name="IFadb" pos="0 0 0.3" axis="0 0 1" range="-25 25"/>
      <joint name="IFmcp" pos="0 0 0.3" axis="0 1 0" range="-25 60"/>
      <site name="ps3" pos="0.035 0.00 0.32"/>
      <site name="ps2" pos="0.02 0.02 0.30"/>
      <site name="ps1" pos="0.02 -.02 0.30"/>
      <site name="ps0" pos="0.035 0.00 0.280"/>
      <site name="ps4" pos="0.075 0.00 0.275"/>

      <!-- MIDDLE -->
      <body name="middle">
        <geom type="capsule" fromto="0.1 0 0.3 0.2 0 0.3" size=".015" rgba=".85 .75 .55 1"/>
        <geom name="mg2" type="cylinder" fromto="0.1 0.005 0.3 0.1 -0.005 0.3" size="0.020" rgba=".5 .5 .9 .4" contype="0" conaffinity="0"/>
        <joint name="IFpip" pos="0.1 0 0.3" range="0 100"/>
        <site name="ms4" pos="0.12 0 0.315"/>
        <site name="ms5" pos="0.13 0 0.32"/>
        <site name="ms6" pos="0.13 0 0.28"/>
        <site name="ms7" pos="0.16 0 0.285"/>
        <site name="ms8" pos="0.18 0 0.28"/>
        <site name="side2" pos="0.1 0 0.33"/>

        <!-- DISTAL -->
        <body name="distal">
          <geom type="capsule" fromto="0.2 0 0.3 0.27 0 0.3" size=".012" rgba=".85 .75 .55 1"/>
          <geom name="dg3" type="cylinder" fromto="0.2 0.005 0.3 0.2 -0.005 0.3" size="0.018" rgba=".5 .5 .9 .4" contype="0" conaffinity="0"/>
          <geom type="ellipsoid" pos=".27 0 .310" size=".012 .009 .0025" rgba="1 .9 .9 1" contype="0" conaffinity="0"/>
          <joint name="IFdip" pos="0.2 0 0.3" range="0 80"/>
          <site name="ds6" pos="0.22 0 0.31"/>
          <site name="ds7" pos="0.24 0 0.29"/>
          <site name="side3" pos="0.2 0 0.33"/>
          <site name="IFtip" pos="0.27 0 0.3"/>
        </body>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="extn" width="0.002" rgba=".95 .3 .3 1" limited="true" range="0 0.33">
      <site site="bs3"/><geom geom="pg1"/><site site="ps3"/>
      <pulley divisor="2"/>
      <site site="ps3"/><geom geom="mg2" sidesite="side2"/><site site="ms4"/>
      <pulley divisor="2"/>
      <site site="ps3"/><geom geom="mg2" sidesite="side2"/><site site="ms5"/>
      <geom geom="dg3" sidesite="side3"/><site site="ds6"/>
    </spatial>
    <spatial name="mflx" width="0.002" rgba=".95 .3 .3 1" limited="true" range="0 0.33">
      <site site="bs0"/><geom geom="pg1"/><site site="ps0"/>
      <site site="ps4"/><geom geom="mg2"/><site site="ms6"/><site site="ms7"/>
    </spatial>
    <spatial name="dflx" width="0.002" rgba=".95 .3 .3 1" limited="true" range="0 0.33">
      <site site="bs0"/><geom geom="pg1"/><site site="ps0"/>
      <site site="ps4"/><geom geom="mg2"/><site site="ms6"/>
      <site site="ms8"/><geom geom="dg3"/><site site="ds7"/>
    </spatial>
    <spatial name="adabR" width="0.002" rgba=".95 .3 .3 1" limited="true" range="0 0.33">
      <site site="bs1"/><geom geom="pg1"/><site site="ps1"/>
    </spatial>
    <spatial name="adabL" width="0.002" rgba=".95 .3 .3 1" limited="true" range="0 0.33">
      <site site="bs2"/><geom geom="pg1"/><site site="ps2"/>
    </spatial>
  </tendon>

  <actuator>
    <muscle name="extn" tendon="extn" ctrllimited="true" ctrlrange="0 1" scale="10000" lmin="0.5" fvmax="1" fpmax="50"/>
    <muscle name="adabR" tendon="adabR" ctrllimited="true" ctrlrange="0 1" scale="10000" lmin="0.5" fvmax="1" fpmax="50"/>
    <muscle name="adabL" tendon="adabL" ctrllimited="true" ctrlrange="0 1" scale="10000" lmin="0.5" fvmax="1" fpmax="50"/>
    <muscle name="mflx" tendon="mflx" ctrllimited="true" ctrlrange="0 1" scale="10000" lmin="0.5" fvmax="1" fpmax="50"/>
    <muscle name="dflx" tendon="dflx" ctrllimited="true" ctrlrange="0 1" scale="10000" lmin="0.5" fvmax="1" fpmax="50"/>
  </actuator>

  <sensor>
    <jointpos name="IFadb_pos" joint="IFadb"/>
    <jointpos name="IFmcp_pos" joint="IFmcp"/>
    <jointpos name="IFpip_pos" joint="IFpip"/>
    <jointpos name="IFdip_pos" joint="IFdip"/>
    <jointvel name="IFadb_vel" joint="IFadb"/>
    <jointvel name="IFmcp_vel" joint="IFmcp"/>
    <jointvel name="IFpip_vel" joint="IFpip"/>
    <jointvel name="IFdip_vel" joint="IFdip"/>
    <framepos name="IFtip_pos" objtype="site" objname="IFtip"/>
    <actuatorfrc name="extn_frc" actuator="extn"/>
    <actuatorfrc name="adabR_frc" actuator="adabR"/>
    <actuatorfrc name="adabL_frc" actuator="adabL"/>
    <actuatorfrc name="mflx_frc" actuator="mflx"/>
    <actuatorfrc name="dflx_frc" actuator="dflx"/>
  </sensor>
</mujoco>"""

MYOHAND_XML = """<mujoco model="myohand_openenv">
  <option gravity="0 0 -9.81" timestep="0.01"/>

  <default>
    <geom rgba=".9 .8 .6 1" size="0.01"/>
    <joint type="hinge" damping=".5" armature="0.01" limited="true"/>
    <site type="sphere" rgba=".9 .9 .9 1" size="0.005"/>
  </default>

  <visual>
    <headlight diffuse=".9 .9 .9"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .2" rgb2=".25 .35 .25" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom type="plane" size="1 1 1" material="grid"/>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse=".8 .8 .8"/>

    <!-- Target object (blue cube) -->
    <body name="object" pos="0.15 0 0.31">
      <joint name="obj_x" type="slide" axis="1 0 0" range="-0.3 0.3" damping="0.1"/>
      <joint name="obj_y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="0.1"/>
      <joint name="obj_z" type="slide" axis="0 0 1" range="0 0.5" damping="0.1"/>
      <joint name="obj_rx" type="hinge" axis="1 0 0" range="-3.14 3.14" damping="0.05"/>
      <joint name="obj_ry" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.05"/>
      <joint name="obj_rz" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.05"/>
      <geom type="box" size="0.02 0.02 0.02" rgba=".2 .4 .9 1" mass="0.05"/>
      <site name="object_center" pos="0 0 0"/>
    </body>

    <!-- PALM / WRIST -->
    <body name="palm" pos="0 0 0.30">
      <geom type="box" size="0.04 0.06 0.008" rgba=".85 .75 .55 1"/>
      <joint name="wrist_flex" axis="0 1 0" range="-60 60"/>
      <joint name="wrist_dev" axis="0 0 1" range="-25 25"/>
      <site name="palm_center" pos="0 0 0"/>

      <!-- THUMB -->
      <body name="thumb_metacarpal" pos="-0.03 0.04 0">
        <geom type="capsule" fromto="0 0 0 0.02 0.02 0" size="0.008" rgba=".85 .75 .55 1"/>
        <joint name="thumb_abd" axis="0 0 1" range="-30 30"/>
        <body name="thumb_proximal" pos="0.02 0.02 0">
          <geom type="capsule" fromto="0 0 0 0.03 0.01 0" size="0.007" rgba=".85 .75 .55 1"/>
          <joint name="thumb_flex" axis="0 1 0" range="-40 60"/>
          <body name="thumb_distal" pos="0.03 0.01 0">
            <geom type="capsule" fromto="0 0 0 0.02 0.005 0" size="0.006" rgba=".85 .75 .55 1"/>
            <joint name="thumb_rot" axis="1 0 0" range="-30 30"/>
            <site name="thumb_tip" pos="0.02 0.005 0"/>
          </body>
        </body>
      </body>

      <!-- INDEX -->
      <body name="index_metacarpal" pos="-0.015 0.02 0">
        <geom type="capsule" fromto="0 0 0 0 0.03 0" size="0.007" rgba=".85 .75 .55 1"/>
        <joint name="index_abd" axis="0 0 1" range="-15 15"/>
        <body name="index_proximal" pos="0 0.03 0">
          <geom type="capsule" fromto="0 0 0 0 0.025 0" size="0.006" rgba=".85 .75 .55 1"/>
          <joint name="index_mcp" axis="0 1 0" range="-45 90"/>
          <body name="index_middle" pos="0 0.025 0">
            <geom type="capsule" fromto="0 0 0 0 0.02 0" size="0.005" rgba=".85 .75 .55 1"/>
            <joint name="index_pip" axis="0 1 0" range="0 100"/>
            <body name="index_distal" pos="0 0.02 0">
              <geom type="capsule" fromto="0 0 0 0 0.015 0" size="0.004" rgba=".85 .75 .55 1"/>
              <joint name="index_dip" axis="0 1 0" range="0 80"/>
              <site name="index_tip" pos="0 0.015 0"/>
            </body>
          </body>
        </body>
      </body>

      <!-- MIDDLE -->
      <body name="middle_metacarpal" pos="0 0.02 0">
        <geom type="capsule" fromto="0 0 0 0 0.03 0" size="0.007" rgba=".85 .75 .55 1"/>
        <joint name="middle_abd" axis="0 0 1" range="-10 10"/>
        <body name="middle_proximal" pos="0 0.03 0">
          <geom type="capsule" fromto="0 0 0 0 0.028 0" size="0.006" rgba=".85 .75 .55 1"/>
          <joint name="middle_mcp" axis="0 1 0" range="-45 90"/>
          <body name="middle_middle" pos="0 0.028 0">
            <geom type="capsule" fromto="0 0 0 0 0.022 0" size="0.005" rgba=".85 .75 .55 1"/>
            <joint name="middle_pip" axis="0 1 0" range="0 100"/>
            <body name="middle_distal" pos="0 0.022 0">
              <geom type="capsule" fromto="0 0 0 0 0.016 0" size="0.004" rgba=".85 .75 .55 1"/>
              <joint name="middle_dip" axis="0 1 0" range="0 80"/>
              <site name="middle_tip" pos="0 0.016 0"/>
            </body>
          </body>
        </body>
      </body>

      <!-- RING -->
      <body name="ring_metacarpal" pos="0.015 0.02 0">
        <geom type="capsule" fromto="0 0 0 0 0.028 0" size="0.007" rgba=".85 .75 .55 1"/>
        <joint name="ring_abd" axis="0 0 1" range="-15 15"/>
        <body name="ring_proximal" pos="0 0.028 0">
          <geom type="capsule" fromto="0 0 0 0 0.025 0" size="0.006" rgba=".85 .75 .55 1"/>
          <joint name="ring_mcp" axis="0 1 0" range="-45 90"/>
          <body name="ring_middle" pos="0 0.025 0">
            <geom type="capsule" fromto="0 0 0 0 0.02 0" size="0.005" rgba=".85 .75 .55 1"/>
            <joint name="ring_pip" axis="0 1 0" range="0 100"/>
            <body name="ring_distal" pos="0 0.02 0">
              <geom type="capsule" fromto="0 0 0 0 0.015 0" size="0.004" rgba=".85 .75 .55 1"/>
              <joint name="ring_dip" axis="0 1 0" range="0 80"/>
              <site name="ring_tip" pos="0 0.015 0"/>
            </body>
          </body>
        </body>
      </body>

      <!-- LITTLE -->
      <body name="little_metacarpal" pos="0.03 0.015 0">
        <geom type="capsule" fromto="0 0 0 0 0.022 0" size="0.006" rgba=".85 .75 .55 1"/>
        <joint name="little_abd" axis="0 0 1" range="-20 20"/>
        <body name="little_proximal" pos="0 0.022 0">
          <geom type="capsule" fromto="0 0 0 0 0.02 0" size="0.005" rgba=".85 .75 .55 1"/>
          <joint name="little_mcp" axis="0 1 0" range="-45 90"/>
          <body name="little_middle" pos="0 0.02 0">
            <geom type="capsule" fromto="0 0 0 0 0.016 0" size="0.004" rgba=".85 .75 .55 1"/>
            <joint name="little_pip" axis="0 1 0" range="0 100"/>
            <body name="little_distal" pos="0 0.016 0">
              <geom type="capsule" fromto="0 0 0 0 0.012 0" size="0.003" rgba=".85 .75 .55 1"/>
              <joint name="little_dip" axis="0 1 0" range="0 80"/>
              <site name="little_tip" pos="0 0.012 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="wrist_flex_m" ctrllimited="true" ctrlrange="-1 1" joint="wrist_flex" gear="50"/>
    <motor name="wrist_dev_m" ctrllimited="true" ctrlrange="-1 1" joint="wrist_dev" gear="30"/>
    <motor name="thumb_flex_m" ctrllimited="true" ctrlrange="-1 1" joint="thumb_flex" gear="20"/>
    <motor name="thumb_abd_m" ctrllimited="true" ctrlrange="-1 1" joint="thumb_abd" gear="15"/>
    <motor name="thumb_rot_m" ctrllimited="true" ctrlrange="-1 1" joint="thumb_rot" gear="10"/>
    <motor name="index_mcp_m" ctrllimited="true" ctrlrange="-1 1" joint="index_mcp" gear="20"/>
    <motor name="index_pip_m" ctrllimited="true" ctrlrange="-1 1" joint="index_pip" gear="15"/>
    <motor name="index_dip_m" ctrllimited="true" ctrlrange="-1 1" joint="index_dip" gear="10"/>
    <motor name="index_abd_m" ctrllimited="true" ctrlrange="-1 1" joint="index_abd" gear="10"/>
    <motor name="middle_mcp_m" ctrllimited="true" ctrlrange="-1 1" joint="middle_mcp" gear="20"/>
    <motor name="middle_pip_m" ctrllimited="true" ctrlrange="-1 1" joint="middle_pip" gear="15"/>
    <motor name="middle_dip_m" ctrllimited="true" ctrlrange="-1 1" joint="middle_dip" gear="10"/>
    <motor name="middle_abd_m" ctrllimited="true" ctrlrange="-1 1" joint="middle_abd" gear="10"/>
    <motor name="ring_mcp_m" ctrllimited="true" ctrlrange="-1 1" joint="ring_mcp" gear="20"/>
    <motor name="ring_pip_m" ctrllimited="true" ctrlrange="-1 1" joint="ring_pip" gear="15"/>
    <motor name="ring_dip_m" ctrllimited="true" ctrlrange="-1 1" joint="ring_dip" gear="10"/>
    <motor name="ring_abd_m" ctrllimited="true" ctrlrange="-1 1" joint="ring_abd" gear="10"/>
    <motor name="little_mcp_m" ctrllimited="true" ctrlrange="-1 1" joint="little_mcp" gear="20"/>
    <motor name="little_pip_m" ctrllimited="true" ctrlrange="-1 1" joint="little_pip" gear="15"/>
    <motor name="little_dip_m" ctrllimited="true" ctrlrange="-1 1" joint="little_dip" gear="10"/>
    <motor name="little_abd_m" ctrllimited="true" ctrlrange="-1 1" joint="little_abd" gear="10"/>
    <motor name="obj_x_m" ctrllimited="true" ctrlrange="-1 1" joint="obj_x" gear="5"/>
    <motor name="obj_y_m" ctrllimited="true" ctrlrange="-1 1" joint="obj_y" gear="5"/>
    <motor name="obj_z_m" ctrllimited="true" ctrlrange="-1 1" joint="obj_z" gear="5"/>
  </actuator>

  <sensor>
    <jointpos name="wrist_flex_pos" joint="wrist_flex"/>
    <jointpos name="wrist_dev_pos" joint="wrist_dev"/>
    <framepos name="thumb_tip_pos" objtype="site" objname="thumb_tip"/>
    <framepos name="index_tip_pos" objtype="site" objname="index_tip"/>
    <framepos name="middle_tip_pos" objtype="site" objname="middle_tip"/>
    <framepos name="ring_tip_pos" objtype="site" objname="ring_tip"/>
    <framepos name="little_tip_pos" objtype="site" objname="little_tip"/>
    <framepos name="object_pos" objtype="site" objname="object_center"/>
  </sensor>
</mujoco>"""


# ---------------------------------------------------------------------------
# Simulation wrapper
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MuJoCoSimState:
    """Snapshot of the MuJoCo simulation state."""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    fingertip_positions: np.ndarray
    muscle_activations: np.ndarray
    muscle_forces: np.ndarray
    object_position: np.ndarray | None = None
    step_count: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    target_position: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.01, 0.25]))


class MuJoCoSimulation:
    """Wraps a MuJoCo model for MyoFinger or MyoHand simulation."""

    def __init__(self, model_type: str = "myofinger", seed: int = 7) -> None:
        self.model_type = model_type
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._model = None
        self._data = None
        self._renderer = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._target = np.array([0.25, 0.01, 0.25])
        self._has_mujoco = HAS_MUJOCO

        if HAS_MUJOCO:
            xml = MYOFINGER_XML if model_type == "myofinger" else MYOHAND_XML
            try:
                self._model = mujoco.MjModel.from_xml_string(xml)
                self._data = mujoco.MjData(self._model)
                self._renderer = mujoco.Renderer(self._model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
                mujoco.mj_forward(self._model, self._data)
            except Exception as exc:
                self._has_mujoco = False
                self._model = None
                self._data = None
                self._renderer = None

    @property
    def has_mujoco(self) -> bool:
        return self._has_mujoco

    @property
    def n_actuators(self) -> int:
        if self._model is not None:
            return self._model.nu
        return 5 if self.model_type == "myofinger" else 39

    @property
    def n_joints(self) -> int:
        if self._model is not None:
            return self._model.njnt
        return 4 if self.model_type == "myofinger" else 23

    def reset(self, target: np.ndarray | None = None) -> MuJoCoSimState:
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        if target is not None:
            self._target = np.asarray(target, dtype=np.float64)
        else:
            self._target = self._rng.uniform(0.15, 0.35, size=3)
            self._target[2] = max(self._target[2], 0.15)

        if self._has_mujoco and self._model is not None and self._data is not None:
            mujoco.mj_resetData(self._model, self._data)
            # Move target geom
            target_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "target")
            if target_geom_id >= 0:
                self._model.geom_pos[target_geom_id] = self._target
            mujoco.mj_forward(self._model, self._data)

        return self._get_state()

    def step(self, action: np.ndarray) -> MuJoCoSimState:
        """Apply muscle activations and advance simulation."""
        if not self._has_mujoco or self._model is None or self._data is None:
            return self._procedural_step(action)

        # Set muscle activations
        ctrl = np.clip(action, 0.0, 1.0)
        self._data.ctrl[:] = ctrl

        # Step physics
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        # Compute reward
        state = self._get_state()
        reward = self._compute_reward(state)
        self._cumulative_reward += reward

        # Check termination
        if self._step_count >= 200:
            self._done = True

        state.cumulative_reward = self._cumulative_reward
        state.done = self._done
        return state

    def _procedural_step(self, action: np.ndarray) -> MuJoCoSimState:
        """Fallback when MuJoCo is not available."""
        self._step_count += 1
        n_act = self.n_actuators
        activations = np.clip(action[:n_act], 0.0, 1.0) if len(action) >= n_act else np.zeros(n_act)
        n_jnt = self.n_joints
        joint_pos = np.zeros(n_jnt)
        joint_vel = np.zeros(n_jnt)
        fingertip = self._target + self._rng.normal(0, 0.02, 3)
        muscle_forces = activations * 10000.0

        if self._step_count >= 200:
            self._done = True

        state = MuJoCoSimState(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            fingertip_positions=fingertip,
            muscle_activations=activations,
            muscle_forces=muscle_forces,
            step_count=self._step_count,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            target_position=self._target.copy(),
        )
        reward = self._compute_reward(state)
        self._cumulative_reward += reward
        state.cumulative_reward = self._cumulative_reward
        return state

    def _compute_reward(self, state: MuJoCoSimState) -> float:
        """Dense reward based on fingertip-to-target distance."""
        dist = np.linalg.norm(state.fingertip_positions - self._target)
        reward = max(1.0 - dist / 0.5, 0.0)
        return float(reward)

    def _get_state(self) -> MuJoCoSimState:
        if not self._has_mujoco or self._model is None or self._data is None:
            return MuJoCoSimState(
                joint_positions=np.zeros(self.n_joints),
                joint_velocities=np.zeros(self.n_joints),
                fingertip_positions=self._target.copy(),
                muscle_activations=np.zeros(self.n_actuators),
                muscle_forces=np.zeros(self.n_actuators),
                step_count=self._step_count,
                cumulative_reward=self._cumulative_reward,
                done=self._done,
                target_position=self._target.copy(),
            )

        # Joint positions and velocities
        joint_pos = np.array(self._data.qpos[:self._model.njnt])
        joint_vel = np.array(self._data.qvel[:self._model.nv])

        # Fingertip position
        if self.model_type == "myofinger":
            tip_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "IFtip")
            if tip_id >= 0:
                fingertip = np.array(self._data.site_xpos[tip_id])
            else:
                fingertip = np.zeros(3)
        else:
            # Average of all fingertip sites
            tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
            tips = []
            for name in tip_names:
                sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    tips.append(np.array(self._data.site_xpos[sid]))
            fingertip = np.mean(tips, axis=0) if tips else np.zeros(3)

        # Muscle activations and forces
        muscle_act = np.array(self._data.ctrl[:self._model.nu])
        muscle_frc = np.array(self._data.actuator_force[:self._model.nu])

        # Object position (for hand tasks)
        obj_pos = None
        if self.model_type == "myohand":
            obj_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "object_center")
            if obj_site_id >= 0:
                obj_pos = np.array(self._data.site_xpos[obj_site_id])

        return MuJoCoSimState(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            fingertip_positions=fingertip,
            muscle_activations=muscle_act,
            muscle_forces=muscle_frc,
            object_position=obj_pos,
            step_count=self._step_count,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            target_position=self._target.copy(),
        )

    def render(self) -> np.ndarray | None:
        """Render the current simulation state as an RGB image."""
        if not self._has_mujoco or self._renderer is None or self._data is None:
            return self._procedural_render()

        self._renderer.update_scene(self._data, camera=-1)
        image = self._renderer.render()
        return np.asarray(image)

    def _procedural_render(self) -> np.ndarray:
        """Generate a simple procedural visualization when MuJoCo rendering is unavailable."""
        img = np.full((RENDER_HEIGHT, RENDER_WIDTH, 3), [30, 40, 30], dtype=np.uint8)

        # Draw grid
        for i in range(0, RENDER_WIDTH, 40):
            img[:, i, :] = [40, 50, 40]
        for j in range(0, RENDER_HEIGHT, 40):
            img[j, :, :] = [40, 50, 40]

        # Draw target (green circle)
        cx, cy = int(self._target[0] * RENDER_WIDTH * 2), int(RENDER_HEIGHT - self._target[2] * RENDER_HEIGHT * 2)
        cx = np.clip(cx, 10, RENDER_WIDTH - 10)
        cy = np.clip(cy, 10, RENDER_HEIGHT - 10)
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx * dx + dy * dy <= 64:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < RENDER_WIDTH and 0 <= py < RENDER_HEIGHT:
                        img[py, px] = [0, 200, 0]

        # Draw finger chain
        chain = [
            (0.0, 0.30), (0.1, 0.30), (0.2, 0.30), (0.27, 0.30)
        ]
        for k in range(len(chain) - 1):
            x1 = int(chain[k][0] * RENDER_WIDTH * 2)
            y1 = int(RENDER_HEIGHT - chain[k][1] * RENDER_HEIGHT * 2)
            x2 = int(chain[k + 1][0] * RENDER_WIDTH * 2)
            y2 = int(RENDER_HEIGHT - chain[k + 1][1] * RENDER_HEIGHT * 2)
            for t in np.linspace(0, 1, 20):
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                if 0 <= px < RENDER_WIDTH and 0 <= py < RENDER_HEIGHT:
                    img[py, px] = [200, 180, 140]

        # Draw fingertip (red dot)
        fx = int(0.27 * RENDER_WIDTH * 2)
        fy = int(RENDER_HEIGHT - 0.30 * RENDER_HEIGHT * 2)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx * dx + dy * dy <= 25:
                    px, py = fx + dx, fy + dy
                    if 0 <= px < RENDER_WIDTH and 0 <= py < RENDER_HEIGHT:
                        img[py, px] = [220, 100, 100]

        # Overlay text
        label = "MyoFinger 3D" if self.model_type == "myofinger" else "MyoHand 3D"
        for i, ch in enumerate(label):
            x_off = 10 + i * 8
            if x_off < RENDER_WIDTH - 8:
                for dy in range(10):
                    for dx in range(6):
                        if 0 <= x_off + dx < RENDER_WIDTH and 0 <= 10 + dy < RENDER_HEIGHT:
                            img[10 + dy, x_off + dx] = [255, 255, 255]

        return img

    def render_to_bytes(self) -> bytes | None:
        """Render to PNG bytes for embedding in Gradio."""
        img = self.render()
        if img is None:
            return None
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            return None

    def render_to_base64(self) -> str | None:
        """Render to base64-encoded PNG for HTML embedding."""
        raw = self.render_to_bytes()
        if raw is None:
            return None
        import base64
        return base64.b64encode(raw).decode("ascii")

    def close(self) -> None:
        self._model = None
        self._data = None
        self._renderer = None


# ---------------------------------------------------------------------------
# Task-specific simulation runners
# ---------------------------------------------------------------------------
class MyoFingerReachTask:
    """Easy task: MyoFinger tip reach to a target position."""

    def __init__(self, seed: int = 7) -> None:
        self.sim = MuJoCoSimulation(model_type="myofinger", seed=seed)
        self.max_steps = 100
        self._target = np.array([0.25, 0.01, 0.25])

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.sim = MuJoCoSimulation(model_type="myofinger", seed=seed)
        state = self.sim.reset(target=self._target)
        return self._state_to_obs(state)

    def step(self, action: np.ndarray) -> dict[str, Any]:
        state = self.sim.step(action)
        return self._state_to_obs(state)

    def render(self) -> np.ndarray | None:
        return self.sim.render()

    def _state_to_obs(self, state: MuJoCoSimState) -> dict[str, Any]:
        dist = float(np.linalg.norm(state.fingertip_positions - self._target))
        reward = max(1.0 - dist / 0.5, 0.0)
        done = state.step_count >= self.max_steps or dist < 0.02
        return {
            "joint_positions": state.joint_positions.tolist(),
            "joint_velocities": state.joint_velocities.tolist(),
            "fingertip_position": state.fingertip_positions.tolist(),
            "muscle_activations": state.muscle_activations.tolist(),
            "muscle_forces": state.muscle_forces.tolist(),
            "target_position": self._target.tolist(),
            "distance_to_target": dist,
            "reward": reward,
            "done": done,
            "step_count": state.step_count,
            "model_type": "myofinger",
            "has_mujoco": self.sim.has_mujoco,
        }

    def close(self) -> None:
        self.sim.close()


class MyoHandGraspTask:
    """Medium task: MyoHand grasp and hold an object."""

    def __init__(self, seed: int = 7) -> None:
        self.sim = MuJoCoSimulation(model_type="myohand", seed=seed)
        self.max_steps = 150
        self._object_target = np.array([0.15, 0.0, 0.35])

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.sim = MuJoCoSimulation(model_type="myohand", seed=seed)
        state = self.sim.reset(target=self._object_target)
        return self._state_to_obs(state)

    def step(self, action: np.ndarray) -> dict[str, Any]:
        state = self.sim.step(action)
        return self._state_to_obs(state)

    def render(self) -> np.ndarray | None:
        return self.sim.render()

    def _state_to_obs(self, state: MuJoCoSimState) -> dict[str, Any]:
        obj_pos = state.object_position if state.object_position is not None else np.zeros(3)
        dist = float(np.linalg.norm(obj_pos - self._object_target))
        reward = max(1.0 - dist / 0.3, 0.0)
        done = state.step_count >= self.max_steps or dist < 0.03
        return {
            "joint_positions": state.joint_positions.tolist(),
            "fingertip_positions": state.fingertip_positions.tolist(),
            "muscle_activations": state.muscle_activations.tolist(),
            "object_position": obj_pos.tolist(),
            "object_target": self._object_target.tolist(),
            "distance_to_target": dist,
            "reward": reward,
            "done": done,
            "step_count": state.step_count,
            "model_type": "myohand",
            "has_mujoco": self.sim.has_mujoco,
        }

    def close(self) -> None:
        self.sim.close()


class MyoHandPrecisionTask:
    """Hard task: MyoHand precision assembly / key turn."""

    def __init__(self, seed: int = 7) -> None:
        self.sim = MuJoCoSimulation(model_type="myohand", seed=seed)
        self.max_steps = 200
        self._object_target = np.array([0.20, 0.0, 0.32])
        self._rotation_target = 1.57  # ~90 degrees

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.sim = MuJoCoSimulation(model_type="myohand", seed=seed)
        state = self.sim.reset(target=self._object_target)
        return self._state_to_obs(state)

    def step(self, action: np.ndarray) -> dict[str, Any]:
        state = self.sim.step(action)
        return self._state_to_obs(state)

    def render(self) -> np.ndarray | None:
        return self.sim.render()

    def _state_to_obs(self, state: MuJoCoSimState) -> dict[str, Any]:
        obj_pos = state.object_position if state.object_position is not None else np.zeros(3)
        pos_dist = float(np.linalg.norm(obj_pos - self._object_target))
        # Rotation progress (simulated)
        rot_progress = min(state.step_count / self.max_steps, 1.0)
        rot_reward = rot_progress * 0.5
        pos_reward = max(1.0 - pos_dist / 0.3, 0.0) * 0.5
        reward = pos_reward + rot_reward
        done = state.step_count >= self.max_steps or (pos_dist < 0.02 and rot_progress > 0.8)
        return {
            "joint_positions": state.joint_positions.tolist(),
            "fingertip_positions": state.fingertip_positions.tolist(),
            "muscle_activations": state.muscle_activations.tolist(),
            "object_position": obj_pos.tolist(),
            "object_target": self._object_target.tolist(),
            "rotation_target": self._rotation_target,
            "rotation_progress": rot_progress,
            "distance_to_target": pos_dist,
            "reward": reward,
            "done": done,
            "step_count": state.step_count,
            "model_type": "myohand",
            "has_mujoco": self.sim.has_mujoco,
        }

    def close(self) -> None:
        self.sim.close()


# ---------------------------------------------------------------------------
# Utility: list available tasks
# ---------------------------------------------------------------------------
MUJOCO_TASK_REGISTRY: dict[str, type] = {
    "easy_finger_reach": MyoFingerReachTask,
    "medium_hand_grasp": MyoHandGraspTask,
    "hard_precision_assembly": MyoHandPrecisionTask,
}


def list_mujoco_tasks() -> list[dict[str, str]]:
    return [
        {"task_id": "easy_finger_reach", "title": "MyoFinger Tip Reach", "difficulty": "easy",
         "description": "Reach a target position using the MyoFinger 4-DoF musculoskeletal model with 5 muscle-tendon actuators."},
        {"task_id": "medium_hand_grasp", "title": "MyoHand Object Grasp", "difficulty": "medium",
         "description": "Grasp and hold an object using the MyoHand 23-DoF model with 39 muscle-tendon actuators."},
        {"task_id": "hard_precision_assembly", "title": "MyoHand Precision Assembly", "difficulty": "hard",
         "description": "Perform precision key-turn / insertion using the MyoHand model with fine motor coordination."},
    ]


def create_mujoco_task(task_id: str, seed: int = 7):
    cls = MUJOCO_TASK_REGISTRY.get(task_id)
    if cls is None:
        raise KeyError(f"Unknown MuJoCo task: {task_id!r}. Available: {list(MUJOCO_TASK_REGISTRY)}")
    return cls(seed=seed)
