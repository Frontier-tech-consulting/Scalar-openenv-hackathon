from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

from egocentric_dataset_test.components.pose_extractor import BodyPoseRecord


@dataclass(slots=True)
class WorkerConfig:
    hand_orientation: str = "both"
    task_class: str = "assembly"
    difficulty: str = "easy"
    body_pose_init: BodyPoseRecord | None = None
    factory_id: str = "factory_001"
    worker_id: str = "worker_001"


class SupplyChainSceneBuilder:
    def generate_mjcf(self, config: WorkerConfig) -> str:
        return f"""
<mujoco model=\"supply_chain_worker\">
  <compiler angle=\"degree\" coordinate=\"local\"/>
  <option timestep=\"0.002\" integrator=\"implicitfast\" gravity=\"0 0 -9.81\"/>
  <worldbody>
    <light pos=\"0 0 3\" dir=\"0 0 -1\"/>
    <geom name=\"floor\" type=\"plane\" size=\"4 4 0.1\" rgba=\"0.2 0.2 0.22 1\"/>
    <body name=\"workbench\" pos=\"0.38 0.0 0.45\">
      <geom type=\"box\" size=\"0.6 0.3 0.45\" rgba=\"0.55 0.42 0.28 1\"/>
    </body>
    <body name=\"conveyor\" pos=\"1.3 0.0 0.35\">
      <geom type=\"box\" size=\"1.0 0.2 0.05\" rgba=\"0.15 0.15 0.18 1\"/>
    </body>
    <body name=\"bin\" pos=\"0.15 -0.35 0.08\">
      <geom type=\"box\" size=\"0.15 0.1 0.08\" rgba=\"0.2 0.25 0.7 1\"/>
    </body>
    <body name=\"pcb\" pos=\"0.5 0.0 0.92\">
      <geom type=\"box\" size=\"0.075 0.05 0.0015\" rgba=\"0.1 0.55 0.1 1\" mass=\"0.05\"/>
    </body>
    <camera name=\"ego\" pos=\"0.0 0.0 1.65\" xyaxes=\"1 0 0 0 1 0\" fovy=\"90\"/>
    <camera name=\"side\" pos=\"1.8 -1.5 1.2\" xyaxes=\"0.7 0.7 0 -0.3 0.3 0.9\"/>
    <camera name=\"overhead\" pos=\"0.5 0 2.5\" xyaxes=\"1 0 0 0 1 0\"/>
  </worldbody>
  <custom>
    <text name=\"task_class\" data=\"{config.task_class}\"/>
    <text name=\"hand_orientation\" data=\"{config.hand_orientation}\"/>
    <text name=\"factory_id\" data=\"{config.factory_id}\"/>
    <text name=\"worker_id\" data=\"{config.worker_id}\"/>
  </custom>
</mujoco>
""".strip()

    def build_scene(self, config: WorkerConfig) -> Any:
        xml = self.generate_mjcf(config)
        if mujoco is None:
            return xml
        return mujoco.MjModel.from_xml_string(xml)
