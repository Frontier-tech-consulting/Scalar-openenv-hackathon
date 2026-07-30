"""Quick validation script for the real worker-level pipeline."""
import os
import sys
from itertools import islice

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import numpy as np

from egocentric_dataset_test.competition.demo import _start_worker_training_stream, _worker_choices
from egocentric_dataset_test.competition.real_pipeline import (
    _ARM_HAND_XML,
    _YOLO_MODEL,
    PCBManipEnv,
    _draw_factory_frame,
    _draw_hand_frame,
    run_mediapipe_landmarks,
    run_yolo_detection,
    train_rl_agent,
)

print("imports OK")

# 1. MuJoCo XML compile
xml = _ARM_HAND_XML.replace("{TX}", "0.00").replace("{TY}", "0.00")
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
print(f"XML OK: nq={m.nq}  nv={m.nv}  nu={m.nu}  njnt={m.njnt}")

# 2. Env smoke-test
env = PCBManipEnv(np.array([0.38, 0.0, 0.718]))
obs, _ = env.reset()
print(f"Env obs shape: {obs.shape}  action shape: {env.action_space.shape}")
a = env.action_space.sample()
obs2, r, term, trunc, info = env.step(a)
print(f"Step OK: reward={r:.4f}  terminated={term}")
rgb = env.render()
print(f"Render OK: {rgb.shape}  dtype={rgb.dtype}")
env.close()

# 3. Factory frame + YOLO
frame = _draw_factory_frame()
print(f"Factory frame: {frame.shape}")
ann, dets, pcb_px = run_yolo_detection(frame)
print(f"YOLO model: {_YOLO_MODEL.name}")
print(f"YOLO done: {len(dets)} detections  PCB pixel: {pcb_px}")

# 4. MediaPipe
hand_rgb = _draw_hand_frame()
lm = run_mediapipe_landmarks(hand_rgb)
print(f"MediaPipe: {'detected ' + str(len(lm)) + ' landmarks' if lm else 'no detection (virtual fallback)'}")

# 5. Short PPO train
ppo, ep_rews, eval_env = train_rl_agent(np.array([0.38, 0.0, 0.718]), total_timesteps=128)
print(f"PPO train OK: {len(ep_rews)} episodes, best={max(ep_rews) if ep_rews else 0:.3f}")
eval_env.close()

# 6. Worker-level stream smoke test
workers = _worker_choices()
worker = workers[0] if workers else "worker_001"
updates = list(islice(_start_worker_training_stream(worker), 2))
print(f"Worker stream OK: worker={worker}  updates={len(updates)}")

print("\nAll checks passed!")
