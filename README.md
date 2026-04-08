---
title: Egocentric Factory Competition Env
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
startup_duration_timeout: 30m
tags:
  - openenv
  - reinforcement-learning
  - robotics
  - fastapi
---

# Egocentric Factory Competition Environment

Submission-ready OpenEnv environment for real-world egocentric manipulation tasks. The environment simulates factory floor actions a human operator performs during electronics assembly: picking a component, sorting it into a tray, and aligning/inserting it during precision assembly.

This submission is the competition-safe distillation of the broader `egocentric_dataset_test` stack: the original repo contains Egocentric-100K ingestion, MyoSuite/MuJoCo simulation, and long-horizon RL tooling, while the shipped OpenEnv task layer compresses that behavior into a deterministic, CPU-friendly benchmark that fits the competition runtime limits.

The current runtime uses a lightweight `surrogate_myo` backend: instead of launching full MuJoCo + MyoSuite physics during judging, it advances through compact `1m50s`-style shards that approximate reach, grasp, alignment, transfer, and insertion phases while keeping the required OpenEnv contract deterministic and fast.

## Why this environment

- Simulates real-world manual work instead of a game or toy domain.
- Exposes a standard OpenEnv-style `reset()`, `step()`, and `state()` lifecycle.
- Ships with 3 deterministic tasks and programmatic graders.
- Provides shaped rewards with partial progress and penalties for unstable control.
- Runs on a shard-based `surrogate_myo` backend so the Space stays within competition CPU/runtime limits.
- Includes a root `inference.py`, root `openenv.yaml`, and root `Dockerfile` for submission.

## Tasks

| Task ID | Difficulty | Objective |
|---|---|---|
| `easy_bin_pick` | easy | Pick a PCB component from a left-side bin and lift it into an inspection pose. |
| `medium_sort_and_place` | medium | Pick a component, rotate it for sorting, move to the tray, and release cleanly. |
| `hard_precision_assembly` | hard | Pick a connector, align it at the slot, insert it, and hold a seated end pose. |

Each task is decomposed into explicit sub-stages. The grader scores:

- stage completion
- precision against target control poses
- step efficiency
- smoothness of the control trajectory

All final scores are normalized to `[0, 1]`.

## Action and observation spaces

### Action

`joint_targets: [reach_x, reach_y, grip_force, wrist_roll]`

- Continuous 4D control vector
- Each dimension is normalized to `[-1.0, 1.0]`

### Observation

Each step returns typed fields including:

- `task_id`, `difficulty`, `task_prompt`
- `backend_mode`, `shard_id`, `shard_progress`
- `current_stage`, `stage_guidance`, `remaining_stages`
- `progress` in `[0, 1]`
- `state_values` containing current tool state and target deltas
- `action_hint` for the current stage
- `reward`, `done`, `last_action_error`

### State

The `state()` endpoint/object includes:

- `episode_id`, `step_count`, `total_steps`
- `cumulative_reward`
- `task_id`, `current_stage`, `progress`
- `backend_mode`, `shard_id`, `shard_progress`, `simulation_origin`
- `grader_score`, `success`, `last_action_error`
- `grader_breakdown` with stage completion, precision, efficiency, and smoothness

## `surrogate_myo` runtime

The env now keeps the same submission-safe HTTP surface while swapping the internal simulator to a shard-aware surrogate backend:

- each task is broken into compact phase shards mapped onto its sub-stages
- each shard carries a small pose bias, response gain, and contact profile
- rewards still depend on stage completion, precision, efficiency, and smoothness
- the runtime exposes shard metadata in observations/state for debugging and reproducibility

Optional environment variables:

```bash
export COMPETITION_BACKEND=surrogate_myo
export SURROGATE_MYO_MAX_SHARD_SECONDS=110
export SURROGATE_MYO_MANIFEST=/data/surrogate_manifest.json
```

If `SURROGATE_MYO_MANIFEST` is omitted, the env uses built-in deterministic shard manifests for all tasks.

## Strict OpenEnv contract

The root deployment exposes the standard environment lifecycle endpoints required by the competition:

- `POST /reset` → returns `observation`, `reward`, `done`
- `POST /step` → accepts `{"action": {"joint_targets": [...]}}` and returns `observation`, `reward`, `done`
- `GET /state` → returns typed episode state
- `GET /schema` → returns JSON schema for action, observation, and state models

Example calls:

```bash
curl -X POST http://127.0.0.1:7860/reset -H 'Content-Type: application/json' -d '{"task_id":"easy_bin_pick","seed":7}'
curl -X POST http://127.0.0.1:7860/step -H 'Content-Type: application/json' -d '{"action":{"joint_targets":[0.78,-0.52,0.10,0.00]}}'
curl http://127.0.0.1:7860/state
```

## Reward design

The reward is shaped across the full trajectory:

- positive reward for reducing distance to the current stage target
- stage-completion bonus when a subgoal is achieved
- success bonus at full task completion
- penalties for regression, clipping, and oscillatory actions

Per-step rewards are clamped to `[0, 1]`.

## Submission files

- `openenv.yaml` — root OpenEnv manifest
- `Dockerfile` — root container build for Hugging Face Spaces
- `inference.py` — required root baseline inference script using the OpenAI client
- `scripts/validate-submission.sh` — convenience validator wrapper

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-space.txt
```

## Run locally

Start the API server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Quick health check:

```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset -H 'Content-Type: application/json' -d '{}'
```

## Baseline inference

Set the model endpoint variables:

```bash
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=openai/gpt-4.1-mini
```

Then run:

```bash
python inference.py
```

The script emits required structured logs in `[START]`, `[STEP]`, and `[END]` format for each task.

### Reproducible baseline scores

The checked-in baseline uses the current stage's `action_hint` as a deterministic policy fallback when no external model output is available. A local run of `python inference.py` produced:

| Task ID | Steps | Score |
|---|---:|---:|
| `easy_bin_pick` | 5 | `0.968` |
| `medium_sort_and_place` | 9 | `0.970` |
| `hard_precision_assembly` | 9 | `0.974` |

These scores are normalized to `[0, 1]` and are deterministic for `seed=7`.

## Hugging Face Space deployment

This repo is configured as a Docker Space through the YAML block at the top of this README.

Typical push flow:

```bash
git add .
git commit -m "Prepare OpenEnv competition submission"
git push
```

After the Space builds, verify:

```bash
curl -X POST https://<your-space>.hf.space/reset -H 'Content-Type: application/json' -d '{}'
```

## Validation

If `openenv` and Docker are installed locally:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://<your-space>.hf.space .
```

For a fast local pre-check before pushing, also run:

```bash
python -m pytest tests/test_competition_submission.py -q
python inference.py
```

## Repository note

This repository still contains research and training code from the earlier egocentric stack. The submission path is intentionally isolated in `egocentric_dataset_test/competition` with a root `server/app.py` shim, and the Docker build copies only the submission runtime so the Space artifact stays lean and reproducible.