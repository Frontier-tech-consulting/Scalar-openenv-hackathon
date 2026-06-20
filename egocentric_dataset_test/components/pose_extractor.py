from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    mp = None  # type: ignore[assignment]


@dataclass(slots=True)
class BodyPoseRecord:
    body_landmarks: np.ndarray
    left_hand_kps: np.ndarray
    right_hand_kps: np.ndarray
    wrist_velocity: np.ndarray
    confidence: float
    frame_index: int
    timestamp_s: float


@dataclass(slots=True)
class PoseSequence:
    records: list[BodyPoseRecord]

    def to_mjcf_keyframes(self) -> str:
        lines = []
        for record in self.records:
            wrist = record.wrist_velocity.reshape(-1).tolist()
            lines.append(
                f'<key name="frame_{record.frame_index}" qpos="{' '.join(f'{value:.6f}' for value in wrist)}"/>'
            )
        return "\n".join(lines)

    def dominant_hand(self) -> str:
        left_energy = sum(float(np.linalg.norm(record.left_hand_kps)) for record in self.records)
        right_energy = sum(float(np.linalg.norm(record.right_hand_kps)) for record in self.records)
        if left_energy <= 1e-6 and right_energy <= 1e-6:
            return "none"
        if left_energy > 1e-6 and right_energy > 1e-6:
            return "both"
        return "left" if left_energy > right_energy else "right"

    def mean_wrist_speed(self) -> float:
        if not self.records:
            return 0.0
        values = [float(np.linalg.norm(record.wrist_velocity, axis=1).mean()) for record in self.records]
        return float(np.mean(values)) if values else 0.0

    def active_frames_pct(self) -> float:
        if not self.records:
            return 0.0
        active = [record for record in self.records if float(np.linalg.norm(record.wrist_velocity)) > 0.05]
        return len(active) / len(self.records)


class PoseExtractor:
    def __init__(self) -> None:
        self._holistic = None
        if mp is not None:
            try:
                self._holistic = mp.solutions.holistic.Holistic(static_image_mode=True)
            except Exception:
                self._holistic = None

    def extract_from_frame(self, jpeg_bytes: bytes, frame_index: int = 0, timestamp_s: float = 0.0) -> BodyPoseRecord:
        zero_body = np.zeros((33, 3), dtype=np.float32)
        zero_hand = np.zeros((21, 3), dtype=np.float32)
        zero_velocity = np.zeros((2, 3), dtype=np.float32)
        if not jpeg_bytes or cv2 is None or self._holistic is None:
            return BodyPoseRecord(zero_body, zero_hand, zero_hand.copy(), zero_velocity, 0.0, frame_index, timestamp_s)

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return BodyPoseRecord(zero_body, zero_hand, zero_hand.copy(), zero_velocity, 0.0, frame_index, timestamp_s)
        result = self._holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        def _landmarks_to_array(landmarks, expected: int) -> np.ndarray:
            array = np.zeros((expected, 3), dtype=np.float32)
            if landmarks is None:
                return array
            for idx, landmark in enumerate(getattr(landmarks, "landmark", [])[:expected]):
                array[idx] = [float(landmark.x), float(landmark.y), float(landmark.z)]
            return array

        body = _landmarks_to_array(getattr(result, "pose_landmarks", None), 33)
        left = _landmarks_to_array(getattr(result, "left_hand_landmarks", None), 21)
        right = _landmarks_to_array(getattr(result, "right_hand_landmarks", None), 21)
        confidence = 0.0
        if getattr(result, "pose_landmarks", None) is not None:
            visibilities = [float(getattr(lm, "visibility", 0.0)) for lm in result.pose_landmarks.landmark]
            if visibilities:
                confidence = float(min(visibilities))
        return BodyPoseRecord(body, left, right, zero_velocity, confidence, frame_index, timestamp_s)

    def extract_from_clip(self, frames: list[bytes], fps: float = 30.0) -> PoseSequence:
        records: list[BodyPoseRecord] = []
        previous_left = None
        previous_right = None
        dt = 1.0 / max(fps, 1e-6)
        for idx, frame in enumerate(frames):
            record = self.extract_from_frame(frame, frame_index=idx, timestamp_s=idx * dt)
            left_wrist = record.left_hand_kps[0] if len(record.left_hand_kps) else np.zeros(3, dtype=np.float32)
            right_wrist = record.right_hand_kps[0] if len(record.right_hand_kps) else np.zeros(3, dtype=np.float32)
            left_velocity = np.zeros(3, dtype=np.float32) if previous_left is None else (left_wrist - previous_left) / dt
            right_velocity = np.zeros(3, dtype=np.float32) if previous_right is None else (right_wrist - previous_right) / dt
            record.wrist_velocity[:] = np.stack([left_velocity, right_velocity], axis=0)
            records.append(record)
            previous_left = left_wrist
            previous_right = right_wrist
        return PoseSequence(records)

    def unproject_2d_to_3d(self, kp_2d, depth_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        x_px, y_px = kp_2d
        x = (float(x_px) - cx) * depth_m / fx
        y = (float(y_px) - cy) * depth_m / fy
        return np.asarray([x, y, depth_m], dtype=np.float32)
