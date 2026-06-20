# Building RL Environments with OpenEnv

A hands-on course for ML engineers, researchers, and hobbyists who want to use and build RL environments for LLM training.

**5 modules · ~45-60 min each · Markdown + Jupyter notebooks**

## Prerequisites

- Basic Python
- Familiarity with the Hugging Face ecosystem
- No RL experience required

## How to Use This Course

Each module has two parts:
1. **README.md** — Concepts, architecture, context. Read this first.
2. **notebook.ipynb** — Hands-on code. Open in Google Colab and run top-to-bottom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Modules

| # | Module | What You'll Learn | Notebook |
|---|--------|-------------------|----------|
| 1 | [Why OpenEnv?](module-1/README.md) | The RL loop, why Gym falls short, OpenEnv architecture | [Open →](module-1/notebook.ipynb) |
| 2 | [Using Existing Environments](module-2/README.md) | Environment Hub, type-safe models, policies, competition | [Open →](module-2/notebook.ipynb) |
| 3 | [Deploying Environments](module-3/README.md) | Local dev, Docker, HF Spaces, `openenv push` | [Open →](module-3/notebook.ipynb) |
| 4 | [Building Your Own Environment](module-4/README.md) | The 3-component pattern, scaffold → deploy | [Open →](module-4/notebook.ipynb) |
| 5 | [Training with OpenEnv + TRL](module-5/README.md) | GRPO, reward functions, Wordle training | [Open →](module-5/notebook.ipynb) |

## Quick Start

```bash
# Install OpenEnv core
pip install openenv-core

# Clone the OpenEnv repo to get typed environment clients
git clone https://github.com/meta-pytorch/OpenEnv.git
```

```python
import sys, os
repo = os.path.abspath('OpenEnv')
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, 'src'))

# Echo environment — uses MCP tool-calling interface
from envs.echo_env import EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as env:
    env.reset()
    response = env.call_tool("echo_message", message="Hello, OpenEnv!")
    print(response)  # Hello, OpenEnv!

# OpenSpiel environments — use standard reset/step interface
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction

with OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space").sync() as env:
    result = env.reset()
    result = env.step(OpenSpielAction(action_id=1, game_name="catch"))
    print(result.observation.legal_actions)
```

Every standard OpenEnv environment uses the same 3-method interface: `reset()`, `step()`, `state()`.

## Local OpenEnv Wrapper

This repo also exposes a local wrapper around the upstream `openenv` CLI through `python -m openenv_cli openenv ...`.

Examples:

```bash
# Scaffold a new environment
python -m openenv_cli openenv init my_env --output-dir /tmp/openenv_envs

# Build and validate a dockerized environment
python -m openenv_cli openenv build /tmp/openenv_envs/my_env
python -m openenv_cli openenv validate /tmp/openenv_envs/my_env --json

# Push to Hugging Face Spaces or a registry
python -m openenv_cli openenv push . --repo-id username/my-env
python -m openenv_cli openenv push . --registry docker.io/username --no-interface

# Fork an existing Space and manage deployment variables
python -m openenv_cli openenv fork owner/source-space --repo-id username/my-fork --set-env OPENENV_BASE_URL=https://example

# Manage AI assistant skills
python -m openenv_cli openenv skills preview
python -m openenv_cli openenv skills add --cursor --force

# Validate and inspect an API endpoint
python -m openenv_cli openenv endpoint --base-url http://127.0.0.1:8000
```

## ARC-AGI-3 Testing Guide

This repo includes an OpenEnv-native ARC-AGI-3 environment for:

- local environment validation
- schema and action-space inspection
- PPO-based RL smoke tests
- swarm rollouts and replay inspection
- OpenEnv endpoint testing for LLM-oriented agent training stacks

### 1. Inspect the ARC schema

Use this first to confirm the typed action/observation contract exposed by the environment:

```bash
python -m openenv_cli env list
python -m openenv_cli env info
python -m openenv_cli env schema
```

The ARC descriptor includes:

- typed `ACTION1`-`ACTION7` + `SUBMIT`
- coordinate-aware paint actions
- structured observations with training examples, cursor state, and similarity
- replay/recording support
- swarm-compatible session semantics

### 2. Run a local ARC simulation

This is the fastest way to verify that the environment resets, steps, records, and serializes correctly:

```bash
python -m openenv_cli simulate --steps 8 --seed 42 --output outputs/arc_simulation.json
python -m openenv_cli replay outputs/arc_simulation.json
```

Use this when you want to inspect:

- current grid state
- training examples embedded in the observation
- legal action flow
- generated replay/recording payloads

### 3. Serve ARC-AGI-3 as an OpenEnv endpoint

Start the OpenEnv-compatible FastAPI/WebSocket server:

```bash
python -m openenv_cli serve --host 0.0.0.0 --port 8000
```

Then validate the runtime endpoint through the upstream OpenEnv CLI wrapper:

```bash
python -m openenv_cli openenv endpoint --base-url http://127.0.0.1:8000
python -m openenv_cli openenv validate . --url http://127.0.0.1:8000 --verbose
```

This verifies the environment as an OpenEnv runtime rather than just a local Python object.

### 4. Run a PPO smoke test on ARC-AGI-3

For a short end-to-end training pass:

```bash
python train_arc_template_v2.py --timesteps 256 --output-dir outputs/arc_v2_smoke
```

Or use the CLI directly:

```bash
python -m openenv_cli train \
    --env-type arc_agi \
    --timesteps 1000 \
    --difficulty easy \
    --max-grid-size 6 \
    --max-steps 50 \
    --output outputs/arc_train
```

Artifacts written to the output directory include:

- checkpoints
- summary metrics
- evaluation stats
- optional diagnostics/replay outputs

### 5. Test swarm rollouts

ARC Prize-style swarm orchestration is supported through parallel local workers:

```bash
# Expose the environment
python -m openenv_cli serve --host 0.0.0.0 --port 8000

# In your training script, point the environment client at the remote URL:
#   from openenv import OpenEnv
#   env = OpenEnv("https://your-space.hf.space")
#
# Or via the upstream CLI for schema/versioning:
python -m openenv_cli openenv validate . --url https://your-space.hf.space --verbose
```

For AI-assistant integration and operator tooling:

```bash
python -m openenv_cli openenv skills preview
python -m openenv_cli openenv skills add --cursor --force
```

### 7. What is currently implemented here

This repository currently gives you a complete OpenEnv-native ARC-AGI-3 testing loop for:

- environment definition via typed actions/observations/states
- local training with PPO
- endpoint validation through the upstream `openenv` CLI
- replay and recordings
- local swarm orchestration
- deployment/build/push/fork/skills delegation through `openenv`

If you want to plug in a larger LLM trainer, use the served OpenEnv endpoint as the environment boundary and keep the training system external to the environment process.

## Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Environment Hub Collection](https://huggingface.co/collections/openenv/environment-hub)
- [TRL Documentation](https://huggingface.co/docs/trl)

---

## Bonus: Scaling OpenEnv

For production workloads beyond a single container, see the scaling appendix below.

### WebSocket vs HTTP

OpenEnv uses WebSocket (`/ws`) for persistent sessions instead of stateless HTTP. Each `step()` call is a lightweight frame (~0.1ms overhead) over an existing connection, vs TCP handshake overhead (~10-50ms) with HTTP.

One container handles many isolated sessions — each WebSocket connection gets its own environment instance server-side.

![WebSocket vs HTTP](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/tutorial/images/websocket.png)

### Single Container Scaling

Before adding containers, maximize a single deployment:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions per worker |

With 8 workers, a single container can handle ~2,048 concurrent sessions for simple text environments.

### Multi-Container with Load Balancing

When a single container isn't enough, deploy multiple containers behind Envoy:

| Setup | Containers | Sessions/container | Total capacity |
|-------|------------|-------------------|----------------|
| Single | 1 | 100 | 100 |
| 4× containers | 4 | 100 | 400 |
| 8× containers | 8 | 100 | 800 |

### Benchmark Results

| Infrastructure | Max Concurrent (WS) | Cores | Sessions/Core |
|----------------|---------------------|-------|---------------|
| HF Spaces (free) | 128 | 2 | 64 |
| Local Uvicorn | 2,048 | 8 | 256 |
| Local Docker | 2,048 | 8 | 256 |
| SLURM multi-node | 16,384 | 96 | 171 |

![Scaling](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/tutorial/images/scaling.png)

For full scaling experiments and code, see [burtenshaw/openenv-scaling](https://github.com/burtenshaw/openenv-scaling).

### Recommendations

- **Development / moderate load (<2K concurrent):** Single Uvicorn or Docker container. Best per-core efficiency (256 sessions/core).
- **Demos and published environments:** HF Spaces free tier, reliable up to 128 concurrent sessions.
- **Large-scale training (>2K concurrent):** Multi-node with Envoy load balancer. See [tutorial/03-scaling.md](https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/03-scaling.md).
