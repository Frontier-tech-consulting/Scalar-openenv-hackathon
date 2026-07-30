# Refactored Feature Data Pipeline — Statement of Work

**Project:** Egocentric Supply Chain Worker → MuJoCo Simulation  
**Dataset:** `builddotai/Egocentric-100K-Evaluation` + existing zarr episode cache  
**Date:** 2026-04-30  
**Status:** Architecture Specification — Pre-Implementation

---

## 1. Problem Statement

The current codebase (`competition/`) has a working MuJoCo simulation stack and an
OpenEnv-compatible environment, but it is **missing metadata alignment between the
raw worker video episodes and the evaluation labels** available in the
`Egocentric-100K-Evaluation` dataset.

Specifically:

| Gap | Impact |
|-----|--------|
| No hand orientation metadata per episode frame | Sim cannot reproduce laterality-aware grasping |
| No personal task class labels (assembly / inspection / transport) | RL reward signal is task-agnostic |
| Ego4D parquet is not joined to the existing zarr episode keys | Per-worker enrichment is impossible |
| No 3D pose extraction pipeline from video frames | Simulation uses placeholder kinematics |
| No end-to-end path from raw `.mp4` → enriched zarr → MuJoCo replay | Current `real_pipeline.py` generates synthetic data only |

The goal of this refactor is to build that full path in three stages:

1. **Ingest** — fetch and parse `ego4d.parquet`, map clips to zarr worker episodes  
2. **Enrich** — attach hand visibility, hand orientation, task class, and 3D pose to each episode  
3. **Simulate** — drive a realistic MuJoCo supply-chain scene from the enriched data

---

## 2. Dataset Architecture — What We Have

### 2.1 `builddotai/Egocentric-100K-Evaluation`

**Files:**

| File | Size | Content |
|------|------|---------|
| `ego4d.parquet` | 2.03 GB | Ego4D evaluation frames: `mp4` (binary) + `json` (string) |
| `egocentric_100k.parquet` | 2.31 GB | 100K dataset frames (without `frame_id`) |
| `epic_kitchens.parquet` | 1.65 GB | EPIC-KITCHENS evaluation frames |
| `prompts/active_manipulation.txt` | 603 B | Gemini prompt used for active-manipulation labeling |
| `prompts/hand_visibility.txt` | — | Gemini prompt used for hand-count labeling |

**Row schema (confirmed from README metadata):**

```
features:
  - name: mp4    dtype: binary   # raw video frame bytes (JPEG encoded)
  - name: json   dtype: string   # JSON evaluation payload
```

**Evaluation JSON payload (per-frame gemini-2.5-flash output):**

```json
{
  "hand_count": 0 | 1 | 2,
  "answer": "yes" | "no",          // active_manipulation
  "clip_id": "...",                 // Ego4D clip identifier
  "frame_index": 1234,
  "timestamp_s": 42.1
}
```

**Published benchmark figures (Ego4D slice, 10K frames):**

| Metric | Ego4D | Egocentric-100K | EPIC-KITCHENS |
|--------|-------|-----------------|---------------|
| 0 hands | 32.67% | **3.04%** | 9.63% |
| 1+ hands | 67.33% | **96.95%** | 90.37% |
| 2 hands | 36.95% | **79.05%** | 61.05% |
| Active manipulation | 50.07% | **92.76%** | 85.04% |

> Egocentric-100K is the highest-quality egocentric labour dataset; Ego4D is used
> here as the evaluation reference because it has stable clip IDs for alignment.

### 2.2 Existing Zarr Episode Cache

Location: `data/egocentric100k_cache/`

Naming pattern:
```
episode_{factory_id}_{worker_id}_{sequence:05d}.zarr
```

Example keys already on disk:
```
episode_factory_001_worker_001_00001.zarr
episode_factory_002_worker_001_00024.zarr
episode_factory_worker_30563.zarr   ← legacy flat naming
```

The zarr stores contain per-step arrays (joint angles, EE pose, reward, etc.) but
**no annotation metadata** — no hand count, no task label, no clip_id linkage.

### 2.3 Existing Competition Modules

| Module | Purpose | Refactor Impact |
|--------|---------|-----------------|
| `competition/tasks.py` | `CompetitionTaskSpec`, `TaskStage`, `GradeBreakdown` | **Keep as-is** — task taxonomy is sound |
| `competition/environment.py` | `EgocentricFactoryObservation`, OpenEnv `Environment` impl | **Extend** — add enriched metadata fields |
| `competition/mujoco_sim.py` | MyoFinger / MyoHand physics (847 lines) | **Extend** — add MS-Human-700 full-body scene |
| `competition/real_pipeline.py` | Perception → Physics → RL → Video (1985 lines) | **Refactor** — replace synthetic frames with real ego4d frames |
| `competition/shards.py` | `SurrogateMyoShard` — surrogate backend sharding | **Keep as-is** |
| `competition/surrogate_backend.py` | Surrogate inference backend | **Keep as-is** |
| `competition/s3_rl_bridge.py` | S3 ↔ RL training bridge | **Keep as-is** |

---

## 3. Refactored Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1 — INGEST                                    │
│                                                                             │
│  ego4d.parquet (HF streaming)                                               │
│       │                                                                     │
│       ▼                                                                     │
│  data/ego4d_metadata.py                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Ego4DMetadataStore                                                  │   │
│  │    .load(max_rows, streaming=True)    ← no 2GB download needed       │   │
│  │    .load_from_parquet(path)           ← local file fallback          │   │
│  │                                                                      │   │
│  │  FrameEvalRecord                                                     │   │
│  │    clip_id, frame_index, timestamp_s                                 │   │
│  │    hand_count         : int     {0, 1, 2}                            │   │
│  │    active_manipulation: bool                                         │   │
│  │    hand_orientation   : str     {none, left, right, both}            │   │
│  │    task_class         : str     {assembly, inspection, transport,    │   │
│  │                                  handling, idle}                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2 — ENRICH                                    │
│                                                                              │
│  components/metadata_enricher.py                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  EpisodeMetadataEnricher                                             │   │
│  │    .match_clip_to_episode(clip_id)                                   │   │
│  │      → fuzzy match ego4d clip_id → zarr episode key                 │   │
│  │    .enrich_zarr(episode_key, records)                                │   │
│  │      → write .attrs["ego4d_meta"] into zarr                         │   │
│  │    .batch_enrich(store, zarr_dir)                                    │   │
│  │      → enrich all cached episodes in parallel                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  components/pose_extractor.py                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  PoseExtractor                                                       │   │
│  │    .extract_from_frame(jpeg_bytes)                                   │   │
│  │      → MediaPipe Holistic  → 33 body + 21L + 21R hand landmarks     │   │
│  │      → project to 3D world coords via pinhole model                 │   │
│  │    .extract_from_clip(clip_frames)                                   │   │
│  │      → temporal smoothing (OneEuroFilter)                            │   │
│  │      → output: PoseSequence {body_kps, left_hand_kps, right_hand_kps}│   │
│  │                                                                      │   │
│  │  BodyPoseRecord                                                      │   │
│  │    body_landmarks   : ndarray[33, 3]   (x, y, z in metres)          │   │
│  │    left_hand_kps    : ndarray[21, 3]                                 │   │
│  │    right_hand_kps   : ndarray[21, 3]                                 │   │
│  │    wrist_velocity   : ndarray[2, 3]    (m/s, per hand)               │   │
│  │    confidence       : float            (min landmark visibility)     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3 — SIMULATE                                  │
│                                                                              │
│  realistic_sim/supply_chain_mjcf.py                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  SupplyChainSceneBuilder                                             │   │
│  │    .build_scene(worker_config)  → mjModel                           │   │
│  │                                                                      │   │
│  │  Scene composition:                                                  │   │
│  │    • Humanoid body  : MS-Human-700 (157 DoF, Apache-2.0)            │   │
│  │        via google-deepmind/mujoco_menagerie ms_human_700/           │   │
│  │    • Workbench      : box geom (1.2m × 0.6m × 0.9m)                │   │
│  │    • Conveyor belt  : textured plane + velocity actuator            │   │
│  │    • Tool objects   : PCB board, bin, screwdriver (mesh + convex)   │   │
│  │    • Cameras        : ego (head-mounted) + side + overhead          │   │
│  │    • Lighting       : directional + point (factory ceiling)         │   │
│  │                                                                      │   │
│  │  WorkerConfig                                                        │   │
│  │    hand_orientation : "left" | "right" | "both"                     │   │
│  │    task_class       : "assembly" | "inspection" | "transport" | …   │   │
│  │    difficulty       : "easy" | "medium" | "hard"                    │   │
│  │    body_pose_init   : BodyPoseRecord | None                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  realistic_sim/video_to_sim_pipeline.py                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  VideoToSimPipeline                          ← master orchestrator   │   │
│  │                                                                      │   │
│  │  Input : zarr episode key  OR  raw .mp4 path                        │   │
│  │  Output: SimResult {                                                 │   │
│  │    video_path     : Path  ← side-by-side ego + sim .mp4             │   │
│  │    reward_curve   : ndarray                                          │   │
│  │    pose_sequence  : PoseSequence                                     │   │
│  │    eval_metadata  : list[FrameEvalRecord]                            │   │
│  │    mjmodel_xml    : str                                              │   │
│  │  }                                                                   │   │
│  │                                                                      │   │
│  │  Steps:                                                              │   │
│  │    1. load zarr or decode mp4 → frames                               │   │
│  │    2. PoseExtractor.extract_from_clip(frames) → PoseSequence        │   │
│  │    3. Ego4DMetadataStore.lookup_by_clip(clip_id)                    │   │
│  │    4. EpisodeMetadataEnricher.enrich_zarr(...)                      │   │
│  │    5. SupplyChainSceneBuilder.build_scene(worker_config)            │   │
│  │    6. run MuJoCo rollout driven by pose trajectory                  │   │
│  │    7. render + compose output video                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. File-by-File Statement of Work

### 4.1 `data/ego4d_metadata.py` ← NEW

**Purpose:** Streaming loader and parser for `ego4d.parquet`.

**Deliverables:**

- `FrameEvalRecord` dataclass  
  Fields: `clip_id`, `frame_index`, `timestamp_s`, `hand_count`, `active_manipulation`,
  `hand_orientation`, `task_class`, `zarr_episode_key`, `raw_json`

- `Ego4DMetadataStore`  
  - `.load(max_rows, streaming=True)` — HF datasets streaming, no full 2GB download  
  - `.load_from_parquet(path)` — local file fallback via `pyarrow`  
  - `.lookup_by_clip(clip_id)` → `list[FrameEvalRecord]`  
  - `.active_clips()` → clips where ≥50% frames are active  
  - `.filter_by_task(task_class)` → filtered record list  
  - `.summary_stats()` → dict matching the README benchmark table  
  - `.as_dataframe()` → pandas DataFrame export  

- `_infer_hand_orientation(hand_count, raw_json)` — heuristic laterality from count  
- `_infer_task_class(active, hand_count, raw_json)` — maps labels to supply-chain taxonomy  

**Dependencies:** `datasets`, `pyarrow`, `pandas` (all lazy-imported with graceful fallback)

---

### 4.2 `components/metadata_enricher.py` ← NEW

**Purpose:** Join `Ego4DMetadataStore` records onto existing zarr episode files.

**Deliverables:**

- `EpisodeMetadataEnricher(store: Ego4DMetadataStore, zarr_dir: Path)`

  - `.match_clip_to_episode(clip_id: str) → str | None`  
    Match an Ego4D `clip_id` to a zarr episode key using:
    1. Exact match on stored `clip_id` attribute if already written  
    2. Fuzzy match on factory/worker IDs extracted from the clip filename pattern  
    3. Temporal overlap matching if timestamp ranges are available  

  - `.enrich_zarr(episode_key: str, records: list[FrameEvalRecord]) → None`  
    Writes into `zarr.open(episode_key).attrs`:
    ```json
    {
      "ego4d_meta": {
        "clip_id": "...",
        "hand_orientation": "both",
        "task_class": "assembly",
        "pct_active": 0.91,
        "hand_count_mean": 1.84,
        "frame_count": 512
      }
    }
    ```

  - `.batch_enrich(max_workers: int = 4) → EnrichmentReport`  
    Parallel enrichment of all zarr episodes using `ThreadPoolExecutor`.

- `EnrichmentReport` dataclass  
  Fields: `total_episodes`, `enriched`, `skipped_no_match`, `skipped_error`, `elapsed_s`

---

### 4.3 `components/pose_extractor.py` ← NEW

**Purpose:** Extract 3D body and hand pose from egocentric JPEG frames.

**Primary tool:** [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)  
**Fallback:** Ego-Exo4D `EGO4D/ego-exo4d-egopose` hand pose baseline (if MediaPipe confidence < 0.6)

**Deliverables:**

- `BodyPoseRecord` dataclass  
  ```python
  body_landmarks:    ndarray[33, 3]   # COCO-style, world metres
  left_hand_kps:     ndarray[21, 3]   # MediaPipe hand landmarks
  right_hand_kps:    ndarray[21, 3]
  wrist_velocity:    ndarray[2, 3]    # m/s, estimated from δpos/δt
  confidence:        float            # min visibility across visible joints
  frame_index:       int
  timestamp_s:       float
  ```

- `PoseSequence` dataclass  
  Wraps `list[BodyPoseRecord]` with:
  - `.to_mjcf_keyframes()` → MJCF `<keyframe>` XML string  
  - `.dominant_hand()` → `"left" | "right" | "both"`  
  - `.mean_wrist_speed()` → float (m/s)  
  - `.active_frames_pct()` → float (frames with both wrists moving > 0.05 m/s)

- `PoseExtractor`  
  - `.extract_from_frame(jpeg_bytes: bytes) → BodyPoseRecord`  
  - `.extract_from_clip(frames: list[bytes], fps: float = 30.0) → PoseSequence`  
    Includes OneEuroFilter temporal smoothing per landmark  
  - `.unproject_2d_to_3d(kp_2d, depth_m, fx, fy, cx, cy) → ndarray[3]`  
    Pinhole camera unprojection; uses factory floor depth prior (0.9m for hands at workbench)

**Pose landmark → MuJoCo joint mapping** (partial, for MS-Human-700):

| MediaPipe Landmark | MS-Human-700 Body Segment | MuJoCo Joint |
|---|---|---|
| `LEFT_WRIST` (15) | `r_radius` | `elbow_flexion_l` |
| `RIGHT_WRIST` (16) | `l_radius` | `elbow_flexion_r` |
| `LEFT_SHOULDER` (11) | `l_humerus` | `shoulder_plane_l` |
| `RIGHT_SHOULDER` (12) | `r_humerus` | `shoulder_plane_r` |
| `LEFT_ELBOW` (13) | `l_ulna` | `elbow_flexion_l` |
| `LEFT_HIP` (23) | `pelvis` | `hip_flexion_l` |

---

### 4.4 `realistic_sim/supply_chain_mjcf.py` ← NEW

**Purpose:** Generate MJCF XML for a realistic factory/supply-chain simulation scene.

**Humanoid asset:** `ms_human_700` from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ms_human_700)  
- 157 DoF biomechanical model, Apache-2.0 license  
- Grade A+ (system-identified values)  
- Replaces the current MyoFinger/MyoHand approximation for full-body tasks

**Alternative / fallback assets (open-source):**

| Asset | Repo | Use Case |
|-------|------|----------|
| `ms_human_700` | mujoco_menagerie | Full-body biomechanical worker |
| `shadow_hand` | mujoco_menagerie | High-DoF manipulation (24 DoF, Apache-2.0) |
| `robotiq_2f85` | mujoco_menagerie | Gripper for bin-picking tasks |
| SMPL-X MJCF | [vchoutas/smplx](https://github.com/vchoutas/smplx) | Parametric body for pose retargeting |
| MyoSuite `myoHand` | [MyoSuite](https://github.com/facebookresearch/myosuite) | Musculoskeletal hand (39 actuators) |

**Deliverables:**

- `WorkerConfig` dataclass  
  ```python
  hand_orientation: str          # "left" | "right" | "both"
  task_class:       str          # "assembly" | "inspection" | "transport" | "handling" | "idle"
  difficulty:       str          # "easy" | "medium" | "hard"
  body_pose_init:   BodyPoseRecord | None
  factory_id:       str          # e.g. "factory_001"
  worker_id:        str          # e.g. "worker_001"
  ```

- `SupplyChainSceneBuilder`  
  - `.build_scene(config: WorkerConfig) → mujoco.MjModel`  
    Returns compiled mjModel ready for simulation.

  - `.generate_mjcf(config: WorkerConfig) → str`  
    Returns raw MJCF XML string. XML structure:
    ```xml
    <mujoco model="supply_chain_worker">
      <compiler .../>
      <option timestep="0.002" integrator="implicitfast"/>
      <asset>
        <include file="ms_human_700/ms_human_700.xml"/>
        <!-- workbench, conveyor, PCB mesh assets -->
      </asset>
      <worldbody>
        <!-- factory floor, ceiling, lighting -->
        <!-- workbench body -->
        <!-- conveyor belt body + velocity actuator -->
        <!-- PCB board body (task object) -->
        <!-- bin / tray body -->
        <!-- include humanoid at workbench position -->
        <!-- ego camera (head-mounted) -->
        <!-- side camera -->
        <!-- overhead camera -->
      </worldbody>
      <actuator>
        <!-- position actuators for humanoid joints -->
        <!-- conveyor velocity actuator -->
      </actuator>
      <sensor>
        <!-- touch sensors on hand geoms -->
        <!-- force-torque on wrist sites -->
        <!-- accelerometer on head -->
      </sensor>
      <keyframe>
        <!-- initial standing pose at workbench -->
        <!-- task-specific grasp poses -->
      </keyframe>
    </mujoco>
    ```

  Scene object dimensions (supply-chain ergonomic defaults):
  - **Workbench:** 1.2 m × 0.6 m × 0.9 m (H), oak texture  
  - **Conveyor belt:** 2.0 m × 0.4 m, velocity actuator ± 0.5 m/s  
  - **PCB board:** 0.15 m × 0.1 m × 0.003 m, 50g  
  - **Bin:** 0.3 m × 0.2 m × 0.15 m, ABS plastic  
  - **Ego camera:** attached to `head` body, 90° FoV, 640×480  

---

### 4.5 `realistic_sim/video_to_sim_pipeline.py` ← NEW

**Purpose:** End-to-end orchestrator: video/zarr in → enriched simulation + video out.

**Deliverables:**

- `SimResult` dataclass  
  ```python
  video_path:       Path          # side-by-side .mp4 (ego left, sim right)
  reward_curve:     ndarray       # per-step reward
  pose_sequence:    PoseSequence  # extracted 3D pose
  eval_metadata:    list[FrameEvalRecord]
  mjmodel_xml:      str           # generated MJCF
  episode_stats:    dict          # step_count, mean_reward, task_class, etc.
  ```

- `VideoToSimPipeline`  
  - `.from_zarr(episode_key: str) → SimResult`  
  - `.from_mp4(video_path: Path, clip_id: str | None = None) → SimResult`  
  - `.from_jpeg_frames(frames: list[bytes], ...) → SimResult`  

  **Internal step sequence:**
  ```
  1. decode_input()
      zarr → load arrays + metadata attrs
      mp4  → decode frames via imageio-ffmpeg

  2. pose_extraction()
      PoseExtractor.extract_from_clip(frames)
      → PoseSequence

  3. metadata_lookup()
      Ego4DMetadataStore.lookup_by_clip(clip_id)
      → list[FrameEvalRecord]
      → derive WorkerConfig from dominant metadata

  4. zarr_enrichment()
      EpisodeMetadataEnricher.enrich_zarr(episode_key, records)
      write ego4d_meta attrs back to disk

  5. scene_build()
      SupplyChainSceneBuilder.build_scene(worker_config)
      → MjModel

  6. sim_rollout()
      drive humanoid joints from PoseSequence keyframes
      compute per-step reward via existing grade_task_run()
      collect frames via mujoco.Renderer (offscreen)

  7. video_compose()
      left panel: annotated ego frames (pose overlay + hand count)
      right panel: MuJoCo rendered simulation frames
      compose side-by-side via imageio-ffmpeg
      write .mp4 to outputs/

  8. return SimResult
  ```

---

### 4.6 `competition/environment.py` ← EXTEND (non-breaking)

Add the following fields to `EgocentricFactoryObservation`:

```python
# NEW — Ego4D-enriched metadata fields
hand_count:         int   = Field(default=0,     description="Gemini-labeled hand count 0/1/2")
hand_orientation:   str   = Field(default="none",description="Inferred: none|left|right|both")
task_class:         str   = Field(default="idle",description="Supply-chain task taxonomy")
active_manipulation:bool  = Field(default=False, description="Gemini active-manipulation label")
pose_confidence:    float = Field(default=0.0,   description="MediaPipe landmark confidence")
```

This is additive-only; no existing field changes. `EgocentricFactoryAction` unchanged.

---

### 4.7 `competition/real_pipeline.py` ← REFACTOR (Stage 1 only)

**Current state:** Generates a synthetic egocentric factory frame using NumPy + OpenCV,
then runs MediaPipe on it. The perception input is entirely fabricated.

**Required change:** Replace `_generate_synthetic_frame()` with a real ego4d frame
pulled from the `Ego4DMetadataStore`:

```python
# BEFORE
frame_bgr = _generate_synthetic_frame(config)

# AFTER
frame_jpeg = metadata_store.sample_frame(task_class=config.task_class)
frame_bgr  = cv2.imdecode(np.frombuffer(frame_jpeg, np.uint8), cv2.IMREAD_COLOR)
```

All downstream stages (YOLO detection, MediaPipe, MuJoCo rollout, video compose)
remain **unchanged**. This is a one-function swap.

---

## 5. Open-Source 3D Segmentation & Simulation Assets

### 5.1 MuJoCo Menagerie — Human Models

| Model | DoF | License | Recommended Use |
|-------|-----|---------|-----------------|
| **MS-Human-700** | 157 | Apache-2.0 | Full-body supply-chain worker (Grade A+) |
| Shadow Hand | 24 | Apache-2.0 | High-fidelity hand manipulation |
| MyoHand (MyoSuite) | 23 | MIT | Musculoskeletal hand with 39 muscle actuators |
| ALOHA 2 | 16 | BSD-3-Clause | Dual-arm manipulation tasks |

Install via:
```bash
pip install robot_descriptions
```
```python
from robot_descriptions.loaders.mujoco import load_robot_description
model = load_robot_description("ms_human_700_mj_description")
```

### 5.2 3D Pose Estimation Tools

| Tool | Input | Output | Notes |
|------|-------|--------|-------|
| **MediaPipe Holistic** | RGB frame | 33 body + 21×2 hand landmarks (3D) | Real-time, Apache-2.0 |
| **EgoExo4D Hand Pose** | Ego RGB | 21 hand joints (3D) | Best accuracy on egocentric occlusions |
| **EgoEgo** ([lijiaman/egoego](https://github.com/lijiaman/egoego_release)) | Ego video | Full-body SMPL pose | NeurIPS 2024, sparse-IMU-free |
| **UnrealEgo2** | Stereo ego | Full-body 3D pose | Best for crouching/bending postures |
| **SMPL-X** | Any | Parametric body mesh | 10475-vertex mesh → MuJoCo URDF |

Primary choice: **MediaPipe Holistic** for speed + Apache-2.0 license compatibility.  
Secondary: **EgoExo4D Hand Pose baseline** for hands during heavy occlusion.

### 5.3 Scene 3D Segmentation

For identifying workbench surfaces and tool objects in the egocentric frame:

| Tool | Task | Notes |
|------|------|-------|
| **YOLOv11** (Ultralytics) | Object detection + segmentation | Already used in `real_pipeline.py` |
| **Segment Anything 2** (Meta) | Zero-shot instance segmentation | Ideal for novel tool objects |
| **FoundationPose** (NVIDIA) | 6-DoF pose estimation of known objects | Best for PCB board / tool retargeting |
| **Dust3r / MASt3r** | Dense 3D reconstruction from 2 frames | For workbench geometry estimation |

---

## 6. New File Layout

```
egocentric_dataset_test/
├── data/
│   ├── ego4d_metadata.py          ← NEW  (Ego4DMetadataStore, FrameEvalRecord)
│   └── egocentric100k_cache/      ← EXISTING zarr episodes
│
├── components/
│   ├── metadata_enricher.py       ← NEW  (EpisodeMetadataEnricher)
│   └── pose_extractor.py          ← NEW  (PoseExtractor, BodyPoseRecord, PoseSequence)
│
├── realistic_sim/
│   ├── supply_chain_mjcf.py       ← NEW  (SupplyChainSceneBuilder, WorkerConfig)
│   └── video_to_sim_pipeline.py   ← NEW  (VideoToSimPipeline, SimResult)
│
├── competition/
│   ├── environment.py             ← EXTEND (5 new obs fields, additive only)
│   ├── real_pipeline.py           ← REFACTOR (replace synthetic frame source)
│   ├── mujoco_sim.py              ← KEEP (MyoFinger/MyoHand still used by surrogate)
│   ├── tasks.py                   ← KEEP
│   ├── shards.py                  ← KEEP
│   ├── surrogate_backend.py       ← KEEP
│   ├── s3_rl_bridge.py            ← KEEP
│   └── server.py                  ← KEEP
│
└── refactor-feature-data-pipeline.md   ← THIS FILE
```

---

## 7. Dependency Requirements

New dependencies to add to `requirements.txt`:

```
# Pose extraction
mediapipe>=0.10.14

# Dataset ingestion
datasets>=3.0.0
pyarrow>=15.0.0

# MuJoCo asset loading
robot_descriptions>=1.9.0

# Already present (verified)
mujoco>=3.1.0
numpy>=1.26.0
pandas>=2.0.0
imageio[ffmpeg]>=2.34.0
opencv-python-headless>=4.9.0
zarr>=2.17.0
```

---

## 8. Implementation Order

| Step | Module | Effort | Depends On |
|------|--------|--------|------------|
| 1 | `data/ego4d_metadata.py` | 2h | nothing |
| 2 | `components/pose_extractor.py` | 3h | mediapipe |
| 3 | `components/metadata_enricher.py` | 2h | Step 1 |
| 4 | `realistic_sim/supply_chain_mjcf.py` | 4h | mujoco_menagerie ms_human_700 |
| 5 | `realistic_sim/video_to_sim_pipeline.py` | 3h | Steps 1–4 |
| 6 | `competition/environment.py` extend | 30m | nothing |
| 7 | `competition/real_pipeline.py` refactor | 1h | Step 1 |

**Total estimated effort:** ~16h

---

## 9. Acceptance Criteria

| Criterion | Test |
|-----------|------|
| Ego4D parquet streams without downloading >300MB | `Ego4DMetadataStore().load(max_rows=500)` completes in <30s |
| Zarr episodes gain `ego4d_meta` attrs after enrichment | `zarr.open(key).attrs["ego4d_meta"]["hand_orientation"]` returns non-null |
| MediaPipe extracts pose from a real ego4d JPEG frame | `PoseExtractor().extract_from_frame(jpeg)` returns `confidence > 0.5` |
| MJCF scene compiles without errors | `mujoco.MjModel.from_xml_string(xml)` succeeds |
| Full pipeline runs end-to-end on one zarr episode | `VideoToSimPipeline().from_zarr("episode_factory_001_worker_001_00001")` returns `SimResult` |
| Output video is valid MP4 | `imageio.get_reader(result.video_path)` returns reader with `n_frames > 0` |
| `real_pipeline.py` uses real ego4d frame, not synthetic | Perception stage detects ≥1 real hand in ego4d frame |
