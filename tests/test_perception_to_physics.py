from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "perception_to_physics.py"
SPEC = importlib.util.spec_from_file_location("perception_to_physics", MODULE_PATH)
perception_to_physics = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = perception_to_physics
SPEC.loader.exec_module(perception_to_physics)


@unittest.skipIf(perception_to_physics.mujoco is None, "mujoco is not installed")
class PerceptionToPhysicsTests(unittest.TestCase):
    def test_infer_target_position_is_workbench_aligned(self) -> None:
        target_pos = perception_to_physics.infer_target_position()

        self.assertEqual(target_pos.shape, (3,))
        self.assertTrue(np.isfinite(target_pos).all())
        self.assertAlmostEqual(
            float(target_pos[2]),
            perception_to_physics.WORKBENCH_TOP_Z + perception_to_physics.OBJECT_HALF_HEIGHT,
        )
        self.assertGreater(target_pos[0], -1.0)
        self.assertLess(target_pos[0], 1.0)
        self.assertGreater(target_pos[1], -1.0)
        self.assertLess(target_pos[1], 2.0)

    def test_headless_rollout_moves_arm_toward_object(self) -> None:
        result = perception_to_physics.run_headless_rollout(steps=200)

        self.assertTrue(np.isfinite(result.target_joints).all())
        self.assertLess(result.final_distance, result.initial_distance)
        self.assertLess(result.final_distance, result.initial_distance * 0.7)
        self.assertLess(result.min_distance, 0.30)
        self.assertGreater(result.object_height, 0.65)
        self.assertGreater(result.contact_count, 0)
        self.assertEqual(result.steps, 200)

    def test_save_rollout_artifacts_writes_latest_metrics(self) -> None:
        result = perception_to_physics.run_headless_rollout(steps=20)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_video = temp_path / "fake_rollout.mp4"
            fake_video.write_bytes(b"mp4")
            artifact_paths = perception_to_physics._make_output_paths(temp_path)
            paths = perception_to_physics.save_rollout_artifacts(
                result,
                output_dir=temp_path,
                video_path=fake_video,
                artifact_paths=artifact_paths,
            )

            self.assertTrue(paths["timestamped_metrics"].exists())
            self.assertTrue(paths["latest_metrics"].exists())
            self.assertTrue(paths["history"].exists())

            latest_payload = json.loads(paths["latest_metrics"].read_text(encoding="utf-8"))
            self.assertEqual(latest_payload["steps"], 20)
            self.assertEqual(latest_payload["video_path"], str(fake_video.resolve()))
            self.assertEqual(latest_payload["run_id"], artifact_paths["run_dir"].name)

    def test_generate_comparison_report_from_runs_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "20260101T000000Z"
            run_dir.mkdir(parents=True, exist_ok=True)
            video_path = run_dir / "rollout.mp4"
            video_path.write_bytes(b"mp4")
            history = temp_path / "runs.jsonl"
            history.write_text(
                json.dumps(
                    {
                        "run_id": "20260101T000000Z",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "video_path": str(video_path.resolve()),
                        "initial_distance": 0.9,
                        "final_distance": 0.3,
                        "min_distance": 0.2,
                        "contact_count": 11,
                        "steps": 120,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report_path = perception_to_physics.generate_comparison_report(
                output_dir=temp_path,
                output_path=temp_path / "comparison_report.html",
            )

            report_html = report_path.read_text(encoding="utf-8")
            self.assertIn("Perception-to-Physics Comparison", report_html)
            self.assertIn("20260101T000000Z", report_html)
            self.assertIn("<video controls preload=\"metadata\"", report_html)
            self.assertIn("rollout.mp4", report_html)


if __name__ == "__main__":
    unittest.main()
