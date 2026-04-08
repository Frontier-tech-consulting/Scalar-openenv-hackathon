"""Lean CLI for the OpenEnv ARC training wrapper.

This module intentionally keeps the command surface small and grounded in the
runtime code that exists in this workspace:

- inspect the local ARC OpenEnv descriptor
- delegate packaging and deployment commands to the upstream ``openenv`` CLI
- train and evaluate PPO policies against local or remote OpenEnv runtimes
- simulate, replay, and swarm ARC episodes using the local ARC environment

The CLI is structured around small service objects so Typer commands stay thin
and the operational logic remains typed, documented, and easy to test.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openenv_cli.arc_openenv_utils import ArcActionCodec, ARCAGIEnvironment, make_arc_env
from openenv_cli.config import EnvironmentType, TrainConfig, TrainingBackend
from openenv_cli.metrics import ConsoleLogger, MetricsLogger
from openenv_cli.model.ppo_v2 import MLPPolicy, PPOTrainer, create_policy_from_env
from openenv_cli.recording import RecordingReplayer, load_recording_from_file
from openenv_cli.runtime_v2 import create_rl_env
from openenv_cli.swarm_v2 import SwarmConfig, SwarmOrchestrator
from openenv_types.arc import build_arc_descriptor
from openenv_types.training import AlgorithmCapabilities, RLTechnique, TrainingTemplate


app = typer.Typer(
    name="openenv-train",
    help="OpenEnv-native ARC training CLI",
    no_args_is_help=True,
    add_completion=False,
)
env_app = typer.Typer(name="env", help="Inspect ARC environment metadata and descriptor output")
openenv_app = typer.Typer(name="openenv", help="Delegate supported operations to the upstream openenv CLI")
skills_app = typer.Typer(name="skills", help="Manage upstream OpenEnv skills through the openenv CLI")
app.add_typer(env_app, name="env")
app.add_typer(openenv_app, name="openenv")
openenv_app.add_typer(skills_app, name="skills")


@dataclass(slots=True)
class OpenEnvCommandRunner:
    """Thin adapter around the upstream ``openenv`` executable."""

    console: Console
    executable: str = "openenv"

    def run(self, args: list[str]) -> None:
        """Run the upstream CLI and stream stdout/stderr back to the user."""

        process = subprocess.run([self.executable, *args], capture_output=True, text=True)
        if process.stdout:
            self.console.print(process.stdout, end="")
        if process.stderr:
            style = "yellow" if process.returncode == 0 else "red"
            self.console.print(process.stderr, style=style, end="")
        if process.returncode != 0:
            raise typer.Exit(process.returncode)


@dataclass(slots=True)
class ArcCliService:
    """Helpers for inspecting and simulating the local ARC environment."""

    console: Console

    def build_template(self) -> TrainingTemplate:
        """Return the canonical PPO training template shown by the CLI."""

        return TrainingTemplate(
            name="openenv_arc_ppo",
            technique=RLTechnique.PPO,
            description="Compact PPO template for turn-based ARC-style OpenEnv games.",
            recommended_for=[
                "ARC grid editing",
                "Recorded rollouts",
                "Swarm workers via parallel sessions",
            ],
            capabilities=AlgorithmCapabilities(
                supports_discrete_actions=True,
                supports_continuous_actions=False,
                supports_multi_agent=True,
                on_policy=True,
            ),
            notes=[
                "Uses documented ACTION1-7 style commands plus submit semantics.",
                "Encodes train examples + current grid instead of leaking target outputs.",
                "Compatible with local env instances and OpenEnv FastAPI/WebSocket servers.",
            ],
        )

    def print_environment_table(self) -> None:
        """Render the supported environment targets exposed by the CLI."""

        table = Table(title="OpenEnv Training Targets")
        table.add_column("Environment", style="cyan")
        table.add_column("Mode", style="green")
        table.add_column("Notes")
        table.add_row(
            "arc_agi",
            "local/openenv",
            "ARC-style grid editing with train examples, recordings, and typed actions",
        )
        table.add_row("openenv_local", "server", "Connect to a local OpenEnv server over /ws")
        table.add_row("openenv_remote", "server", "Connect to a remote OpenEnv server over /ws")
        self.console.print(table)

    def print_descriptor_info(self, name: str) -> None:
        """Render the ARC descriptor summary for the supported local environment."""

        if name != "arc_agi":
            self.console.print(
                Panel(
                    "Descriptor support is implemented for arc_agi. "
                    "Use env schema to export the ARC descriptor JSON."
                )
            )
            return

        descriptor = build_arc_descriptor()
        table = Table(title="ARC OpenEnv Descriptor")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Play Modes", ", ".join(mode.value for mode in descriptor.play_modes))
        table.add_row(
            "Observation Encoder",
            descriptor.rl_projection.observation_encoder if descriptor.rl_projection else "n/a",
        )
        table.add_row(
            "Action Encoder",
            descriptor.rl_projection.action_encoder if descriptor.rl_projection else "n/a",
        )
        table.add_row(
            "Discrete Actions",
            str(descriptor.rl_projection.action_count if descriptor.rl_projection else "n/a"),
        )
        self.console.print(table)

    def emit_descriptor_schema(self, output: Optional[str]) -> None:
        """Print or save the ARC environment descriptor payload."""

        payload = build_arc_descriptor().model_dump()
        if output:
            Path(output).write_text(json.dumps(payload, indent=2))
            self.console.print(f"[green]Saved descriptor to {output}[/green]")
            return
        self.console.print_json(json.dumps(payload))

    def simulate(
        self,
        *,
        actions: str,
        steps: int,
        difficulty: str,
        seed: int,
        output: Optional[str],
    ) -> None:
        """Run a local ARC simulation and print or save the trace payload."""

        env = make_arc_env(difficulty=difficulty, seed=seed)
        codec = ArcActionCodec(env.max_grid_size)
        observation = env.reset(seed=seed)
        action_ids = [int(part) for part in actions.split(",") if part.strip()] if actions else []
        trace = [observation.model_dump()]

        for index in range(steps):
            action_index = action_ids[index] if index < len(action_ids) else codec.action_count - 1
            observation = env.step(codec.decode(action_index))
            trace.append(observation.model_dump())
            if observation.done:
                break

        recording = env.get_recording()
        payload = {
            "trace": trace,
            "recording": recording.model_dump() if recording is not None else None,
            "state": env.state.model_dump(),
        }
        if output:
            Path(output).write_text(json.dumps(payload, indent=2))
            self.console.print(f"[green]Saved simulation to {output}[/green]")
            return
        self.console.print_json(json.dumps(payload))


@dataclass(slots=True)
class TrainingCliService:
    """Build typed configs and execute PPO training or evaluation."""

    console: Console

    def build_train_config(
        self,
        *,
        backend: str,
        env_type: str,
        env_name: str,
        base_url: Optional[str],
        arc_data: Optional[str],
        difficulty: str,
        max_grid_size: int,
        max_steps: int,
        seed: int,
        output_dir: str,
        run_name: Optional[str],
        device: str,
        timesteps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> TrainConfig:
        """Build the canonical training config from CLI arguments."""

        config = TrainConfig()
        config.backend = TrainingBackend(backend)
        config.env.env_type = EnvironmentType(env_type)
        config.env.env_name = env_name
        config.env.base_url = base_url
        config.env.arc_data_path = arc_data
        config.env.difficulty = difficulty
        config.env.max_grid_size = max_grid_size
        config.env.max_steps = max_steps
        config.env.seed = seed
        config.metrics.output_dir = output_dir
        config.run_name = run_name
        config.device = device

        if timesteps is not None:
            config.rl.total_timesteps = timesteps
        if learning_rate is not None:
            config.rl.learning_rate = learning_rate
        if batch_size is not None:
            config.rl.batch_size = batch_size
        if n_steps is not None:
            config.rl.n_steps = n_steps
        return config

    def train(
        self,
        *,
        backend: str,
        env_type: str,
        env_name: str,
        base_url: Optional[str],
        arc_data: Optional[str],
        timesteps: int,
        learning_rate: float,
        batch_size: int,
        n_steps: int,
        difficulty: str,
        max_grid_size: int,
        max_steps: int,
        seed: int,
        output_dir: str,
        run_name: Optional[str],
        device: str,
        verbose: bool,
        template: TrainingTemplate,
    ) -> None:
        """Train a PPO policy against the selected OpenEnv runtime."""

        logger = ConsoleLogger(verbose=verbose)
        config = self.build_train_config(
            backend=backend,
            env_type=env_type,
            env_name=env_name,
            base_url=base_url,
            arc_data=arc_data,
            difficulty=difficulty,
            max_grid_size=max_grid_size,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            run_name=run_name,
            device=device,
            timesteps=timesteps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
        )
        output_path = config.ensure_output_dir()

        metrics = MetricsLogger(
            output_dir=str(output_path),
            run_name=config.run_name or f"{env_name}_{backend}",
        )
        metrics.log_config(
            {
                "backend": backend,
                "env_type": env_type,
                "env_name": env_name,
                "timesteps": timesteps,
                "learning_rate": learning_rate,
                "difficulty": difficulty,
                "max_grid_size": max_grid_size,
                "max_steps": max_steps,
                "seed": seed,
            }
        )

        logger.log_info(f"Training template: {template.name} ({template.technique.value})")
        env = create_rl_env(config.env)
        logger.log_success(
            f"OpenEnv runtime ready: obs_dim={env.observation_dim}, action_size={env.action_size}"
        )
        policy = create_policy_from_env(env, hidden_dims=config.rl.hidden_dims)
        trainer = PPOTrainer(
            policy=policy,
            env=env,
            output_dir=str(output_path),
            learning_rate=config.rl.learning_rate,
            gamma=config.rl.gamma,
            gae_lambda=config.rl.gae_lambda,
            clip_range=config.rl.clip_range,
            n_epochs=config.rl.n_epochs,
            batch_size=config.rl.batch_size,
            ent_coef=config.rl.ent_coef,
            vf_coef=config.rl.vf_coef,
            max_grad_norm=config.rl.max_grad_norm,
            n_steps=config.rl.n_steps,
            log_interval=config.rl.log_interval,
            save_interval=config.rl.save_interval,
            eval_interval=config.rl.eval_interval,
            eval_episodes=config.rl.eval_episodes,
            seed=config.env.seed,
            device=config.device,
            verbose=verbose,
            track_diagnostics=config.rl.track_diagnostics,
        )
        results = trainer.train(config.rl.total_timesteps)
        metrics.log_summary(results)
        env.close()
        self.console.print(
            Panel.fit(
                f"[bold green]Training complete[/bold green]\n"
                f"Reward: {results['final_mean_reward']:.3f} ± {results['final_std_reward']:.3f}\n"
                f"Best: {results['best_reward']:.3f}\n"
                f"Output: {results['output_dir']}",
                title="PPO Results",
            )
        )

    def evaluate(
        self,
        *,
        checkpoint: str,
        env_type: str,
        base_url: Optional[str],
        episodes: int,
        difficulty: str,
        max_grid_size: int,
        max_steps: int,
        seed: int,
    ) -> None:
        """Load a PPO checkpoint and evaluate it against the selected runtime."""

        config = self.build_train_config(
            backend=TrainingBackend.RL.value,
            env_type=env_type,
            env_name="arc_agi",
            base_url=base_url,
            arc_data=None,
            difficulty=difficulty,
            max_grid_size=max_grid_size,
            max_steps=max_steps,
            seed=seed,
            output_dir=str(Path(checkpoint).parent),
            run_name=None,
            device="cpu",
        )
        env = create_rl_env(config.env)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy = MLPPolicy(
            observation_dim=int(payload["observation_dim"]),
            action_size=int(payload["action_size"]),
        )
        policy.load_state_dict(payload["policy_state_dict"])
        trainer = PPOTrainer(
            policy=policy,
            env=env,
            output_dir=str(Path(checkpoint).parent / "eval_tmp"),
            verbose=False,
            track_diagnostics=False,
        )
        results = trainer.evaluate(episodes)
        env.close()
        self.console.print_json(json.dumps(results))


@dataclass(slots=True)
class SwarmCliService:
    """Helpers for parallel ARC swarm runs over local environment instances."""

    console: Console

    def _build_env_factory(self, difficulty: str, seed: int) -> Callable[..., ARCAGIEnvironment]:
        """Create the environment factory consumed by the swarm orchestrator."""

        def factory(
            task_id: Optional[str] = None,
            difficulty: str = difficulty,
            seed: int = seed,
        ) -> ARCAGIEnvironment:
            return make_arc_env(
                task_id=task_id,
                difficulty=difficulty,
                seed=seed,
                max_grid_size=6,
                max_steps=50,
            )

        return factory

    def _load_policy_factory(self, checkpoint: Optional[str]) -> Optional[Callable[[Any], int]]:
        """Load an optional deterministic PPO policy for swarm actions."""

        if not checkpoint:
            return None

        self.console.print(f"[info] Loading policy from {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy = MLPPolicy(
            observation_dim=int(payload.get("observation_dim", 112)),
            action_size=int(payload.get("action_size", 43)),
        )
        policy.load_state_dict(payload["policy_state_dict"])
        policy.eval()

        def make_action(obs_vec: Any) -> int:
            with torch.no_grad():
                logits, _ = policy.forward(torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0))
                return int(logits.argmax(-1).item())

        return make_action

    def run(
        self,
        *,
        games: Optional[str],
        agent: str,
        agent_type: str,
        tags: str,
        max_actions: int,
        output_dir: str,
        num_workers: int,
        difficulty: str,
        seed: int,
        checkpoint: Optional[str],
    ) -> None:
        """Run a swarm across one or more ARC task identifiers."""

        game_ids = [game.strip() for game in games.split(",") if game.strip()] if games else []
        config = SwarmConfig(
            agent_name=agent,
            agent_type=agent_type,
            max_actions=max_actions,
            tags=[tag for tag in tags.split(",") if tag] if tags else [],
            output_dir=output_dir,
            num_workers=num_workers,
            difficulty=difficulty,
            seed=seed,
        )
        self.console.print(
            Panel.fit(
                f"[bold cyan]ARC OpenEnv Swarm[/bold cyan]\n"
                f"Games: {len(game_ids)} ({', '.join(game_ids) if game_ids else 'auto-generated'})\n"
                f"Agent: {agent} ({agent_type})  Workers: {num_workers}\n"
                f"Output: {output_dir}/"
            )
        )

        if not game_ids:
            game_ids = [f"synthetic_{difficulty}_{index}" for index in range(num_workers)]

        orchestrator = SwarmOrchestrator(config)
        orchestrator.run(
            game_ids=game_ids,
            env_factory=self._build_env_factory(difficulty=difficulty, seed=seed),
            policy_factory=self._load_policy_factory(checkpoint),
        )
        summary = orchestrator.summary()
        self.console.print(
            Panel.fit(
                f"[bold green]Swarm complete[/bold green]\n"
                f"Games: {summary['total_games']}  Solved: {summary['solved']} ({summary['solve_rate']:.1%})\n"
                f"Mean score: {summary['mean_score']:.3f} ± {summary['std_score']:.3f}\n"
                f"Scorecard: {summary['scorecard_id']}\n"
                f"Recordings: {summary['recordings_dir']}",
                title="Swarm Results",
            )
        )


console = Console()
openenv_runner = OpenEnvCommandRunner(console=console)
arc_service = ArcCliService(console=console)
training_service = TrainingCliService(console=console)
swarm_service = SwarmCliService(console=console)


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated CLI option into trimmed values."""

    return [item.strip() for item in value.split(",") if item.strip()]


@env_app.command("list")
def env_list() -> None:
    """List the environment targets supported by this CLI."""

    arc_service.print_environment_table()


@env_app.command("info")
def env_info(name: str = "arc_agi") -> None:
    """Show descriptor-level information for the requested environment."""

    arc_service.print_descriptor_info(name)


@env_app.command("schema")
def env_schema(output: Optional[str] = None) -> None:
    """Print or save the ARC descriptor JSON schema payload."""

    arc_service.emit_descriptor_schema(output)


@openenv_app.command("init")
def openenv_init(env_name: str, output_dir: Optional[str] = None) -> None:
    """Delegate ``openenv init`` to the upstream executable."""

    args = ["init", env_name]
    if output_dir:
        args.extend(["--output-dir", output_dir])
    openenv_runner.run(args)


@openenv_app.command("build")
def openenv_build(
    env_path: str = ".",
    tag: Optional[str] = None,
    context: Optional[str] = None,
    dockerfile: Optional[str] = None,
    no_cache: bool = False,
    build_args: str = "",
) -> None:
    """Delegate ``openenv build`` with a small typed wrapper."""

    args = ["build", env_path]
    if tag:
        args.extend(["--tag", tag])
    if context:
        args.extend(["--context", context])
    if dockerfile:
        args.extend(["--dockerfile", dockerfile])
    if no_cache:
        args.append("--no-cache")
    for item in _parse_csv(build_args):
        args.extend(["--build-arg", item])
    openenv_runner.run(args)


@openenv_app.command("validate")
def openenv_validate(
    target: str = ".",
    url: Optional[str] = None,
    json_output: bool = False,
    timeout: float = 5.0,
    verbose: bool = False,
) -> None:
    """Delegate ``openenv validate`` to validate an env package or endpoint."""

    args = ["validate", target]
    if url:
        args.extend(["--url", url])
    if json_output:
        args.append("--json")
    args.extend(["--timeout", str(timeout)])
    if verbose:
        args.append("--verbose")
    openenv_runner.run(args)


@openenv_app.command("endpoint")
def openenv_endpoint(
    base_url: str = "http://127.0.0.1:8000",
    timeout: float = 5.0,
    json_output: bool = False,
) -> None:
    """Show the endpoint layout and validate it with the upstream CLI."""

    console.print(
        Panel.fit(
            f"[bold cyan]OpenEnv Endpoint[/bold cyan]\n"
            f"Base URL: {base_url}\n"
            f"Schema: {base_url.rstrip('/')}/schema\n"
            f"WebSocket: {base_url.rstrip('/')}/ws",
            title="Endpoint",
        )
    )
    args = ["validate", ".", "--url", base_url, "--timeout", str(timeout)]
    if json_output:
        args.append("--json")
    openenv_runner.run(args)


@openenv_app.command("push")
def openenv_push(
    directory: str = ".",
    repo_id: Optional[str] = None,
    base_image: Optional[str] = None,
    registry: Optional[str] = None,
    interface: bool = False,
    no_interface: bool = False,
    private: bool = False,
    create_pr: bool = False,
    exclude: Optional[str] = None,
) -> None:
    """Delegate ``openenv push`` to the upstream executable."""

    args = ["push", directory]
    if repo_id:
        args.extend(["--repo-id", repo_id])
    if base_image:
        args.extend(["--base-image", base_image])
    if registry:
        args.extend(["--registry", registry])
    if exclude:
        args.extend(["--exclude", exclude])
    if interface:
        args.append("--interface")
    if no_interface:
        args.append("--no-interface")
    if private:
        args.append("--private")
    if create_pr:
        args.append("--create-pr")
    openenv_runner.run(args)


@openenv_app.command("fork")
def openenv_fork(
    source_space: str,
    repo_id: Optional[str] = None,
    private: bool = False,
    set_env: str = "",
    set_secret: str = "",
    hardware: Optional[str] = None,
) -> None:
    """Delegate ``openenv fork`` to the upstream executable."""

    args = ["fork", source_space]
    if repo_id:
        args.extend(["--repo-id", repo_id])
    if hardware:
        args.extend(["--hardware", hardware])
    if private:
        args.append("--private")
    for item in _parse_csv(set_env):
        args.extend(["--set-env", item])
    for item in _parse_csv(set_secret):
        args.extend(["--set-secret", item])
    openenv_runner.run(args)


@openenv_app.command("serve")
def openenv_serve(env_path: str = ".", port: int = 8000, host: str = "0.0.0.0", reload: bool = False) -> None:
    """Delegate ``openenv serve`` to the upstream executable."""

    args = ["serve", env_path, "--port", str(port), "--host", host]
    if reload:
        args.append("--reload")
    openenv_runner.run(args)


@skills_app.command("preview")
def openenv_skills_preview() -> None:
    """Delegate ``openenv skills preview`` to the upstream executable."""

    openenv_runner.run(["skills", "preview"])


@skills_app.command("add")
def openenv_skills_add(
    claude: bool = False,
    codex: bool = False,
    cursor: bool = False,
    opencode: bool = False,
    global_install: bool = False,
    dest: Optional[str] = None,
    force: bool = False,
) -> None:
    """Delegate ``openenv skills add`` to the upstream executable."""

    args = ["skills", "add"]
    if claude:
        args.append("--claude")
    if codex:
        args.append("--codex")
    if cursor:
        args.append("--cursor")
    if opencode:
        args.append("--opencode")
    if global_install:
        args.append("--global")
    if force:
        args.append("--force")
    if dest:
        args.extend(["--dest", dest])
    openenv_runner.run(args)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the local ARC environment as an OpenEnv FastAPI/WebSocket server."""

    uvicorn.run("openenv_cli.server_v2:app", host=host, port=port, reload=reload)


@app.command()
def train(
    backend: str = "rl",
    env_type: str = "arc_agi",
    env_name: str = "arc_agi",
    base_url: Optional[str] = None,
    arc_data: Optional[str] = None,
    timesteps: int = 50_000,
    learning_rate: float = 3e-4,
    batch_size: int = 128,
    n_steps: int = 1024,
    difficulty: str = "easy",
    max_grid_size: int = 6,
    max_steps: int = 50,
    seed: int = 42,
    output_dir: str = "outputs",
    run_name: Optional[str] = None,
    device: str = "auto",
    verbose: bool = True,
) -> None:
    """Train a PPO policy against the local ARC env or a remote OpenEnv runtime."""

    training_service.train(
        backend=backend,
        env_type=env_type,
        env_name=env_name,
        base_url=base_url,
        arc_data=arc_data,
        timesteps=timesteps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        difficulty=difficulty,
        max_grid_size=max_grid_size,
        max_steps=max_steps,
        seed=seed,
        output_dir=output_dir,
        run_name=run_name,
        device=device,
        verbose=verbose,
        template=arc_service.build_template(),
    )


@app.command()
def evaluate(
    checkpoint: str,
    env_type: str = "arc_agi",
    base_url: Optional[str] = None,
    episodes: int = 20,
    difficulty: str = "easy",
    max_grid_size: int = 6,
    max_steps: int = 50,
    seed: int = 42,
) -> None:
    """Evaluate a saved PPO checkpoint against the selected runtime."""

    training_service.evaluate(
        checkpoint=checkpoint,
        env_type=env_type,
        base_url=base_url,
        episodes=episodes,
        difficulty=difficulty,
        max_grid_size=max_grid_size,
        max_steps=max_steps,
        seed=seed,
    )


@app.command()
def simulate(
    actions: str = "",
    steps: int = 15,
    difficulty: str = "easy",
    seed: int = 42,
    output: Optional[str] = None,
) -> None:
    """Simulate a local ARC episode from an optional comma-separated action list."""

    arc_service.simulate(actions=actions, steps=steps, difficulty=difficulty, seed=seed, output=output)


@app.command()
def quickstart(timesteps: int = 10_000) -> None:
    """Run a small local PPO training job with the default ARC configuration."""

    console.print(
        Panel.fit(
            "[bold cyan]OpenEnv Quickstart[/bold cyan]\n"
            "Local ARC env • PPO • typed ACTION1-7 schema • train-example aware observations"
        )
    )
    train(timesteps=timesteps, run_name="quickstart_v2")


@app.command()
def swarm(
    games: Optional[str] = None,
    agent: str = "ppo_agent",
    agent_type: str = "ppo",
    tags: str = "",
    max_actions: int = 100,
    output_dir: str = "outputs",
    num_workers: int = 4,
    difficulty: str = "easy",
    seed: int = 42,
    checkpoint: Optional[str] = None,
) -> None:
    """Run a parallel swarm of ARC sessions using local environments only."""

    swarm_service.run(
        games=games,
        agent=agent,
        agent_type=agent_type,
        tags=tags,
        max_actions=max_actions,
        output_dir=output_dir,
        num_workers=num_workers,
        difficulty=difficulty,
        seed=seed,
        checkpoint=checkpoint,
    )


@app.command()
def replay(recording_path: str, delay: float = 0.0, output: Optional[str] = None) -> None:
    """Replay a saved ARC recording in the terminal."""

    result = load_recording_from_file(recording_path)
    replayer = RecordingReplayer(result)
    replayer.replay(delay=delay)
    if output:
        Path(output).write_text(json.dumps(result.__dict__, default=str, indent=2))
        console.print(f"[green]Saved replay result to {output}[/green]")


def main() -> None:
    """Typer entrypoint used by ``python -m openenv_cli``."""

    app()
