from __future__ import annotations

import json
import os
from typing import Iterable

from openai import OpenAI

from egocentric_dataset_test.competition import (
    EgocentricFactoryAction,
    EgocentricFactoryCompetitionEnv,
    list_task_specs,
)


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
BENCHMARK = "egocentric_factory_competition_env"
TEMPERATURE = 0.0
MAX_TOKENS = 120

SYSTEM_PROMPT = (
    "You are controlling a real-world factory manipulation agent. "
    "Given a task and its current stage, provide a very short strategy sentence focused on safe, smooth manipulation."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: Iterable[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _build_user_prompt(observation) -> str:
    state_str = ", ".join(f"{value:.2f}" for value in observation.state_values)
    hint_str = ", ".join(f"{value:.2f}" for value in observation.action_hint)
    return (
        f"Task: {observation.task_prompt}\n"
        f"Difficulty: {observation.difficulty}\n"
        f"Current stage: {observation.current_stage}\n"
        f"Stage guidance: {observation.stage_guidance}\n"
        f"Progress: {observation.progress:.2f}\n"
        f"State values: [{state_str}]\n"
        f"Recommended action hint: [{hint_str}]\n"
        "Return a one-sentence manipulation strategy."
    )


def _request_task_plan(client: OpenAI | None, observation) -> str | None:
    if client is None:
        return None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation)},
            ],
            timeout=15,
        )
        content = (completion.choices[0].message.content or "").strip()
        return content or None
    except Exception:
        return None


def run_task(client: OpenAI | None, task_id: str) -> None:
    env = EgocentricFactoryCompetitionEnv(task_id=task_id, seed=7)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        observation = env.reset(task_id=task_id, seed=7)
        _request_task_plan(client, observation)
        while not observation.done and steps_taken < env.state.total_steps:
            action_values = [float(value) for value in observation.action_hint]
            observation = env.step(EgocentricFactoryAction(joint_targets=action_values))
            state = env.state
            steps_taken += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            action_str = json.dumps(action_values, separators=(",", ":"))
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=observation.done,
                error=observation.last_action_error,
            )
            score = float(state.grader_score)
            success = bool(state.success)
            if observation.done:
                break
    finally:
        state = env.state
        score = float(state.grader_score)
        success = bool(state.success)
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    for spec in list_task_specs():
        run_task(client, spec.task_id)


if __name__ == "__main__":
    main()
