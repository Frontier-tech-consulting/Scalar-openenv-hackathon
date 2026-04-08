from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from egocentric_dataset_test.competition.environment import (
    ACTION_DIMENSION,
    DEFAULT_BACKEND_MODE,
    EgocentricFactoryAction,
    EgocentricFactoryCompetitionEnv,
    EgocentricFactoryObservation,
    EgocentricFactoryState,
)
from egocentric_dataset_test.competition.tasks import DEFAULT_TASK_ID, list_task_specs

try:
    from openenv.core.env_server.types import ResetResponse, StepResponse
except Exception:  # pragma: no cover
    class ResetResponse(BaseModel):  # type: ignore[no-redef]
        observation: dict[str, Any]
        reward: float | None = None
        done: bool = False

    class StepResponse(BaseModel):  # type: ignore[no-redef]
        observation: dict[str, Any]
        reward: float | None = None
        done: bool = False


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default=DEFAULT_TASK_ID, description="Task identifier to start.")
    seed: int = Field(default=7, description="Deterministic seed for reproducible resets.")
    episode_id: str | None = Field(default=None, description="Optional custom episode identifier.")
    backend_mode: str = Field(default=DEFAULT_BACKEND_MODE, description="Runtime backend mode.")


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: EgocentricFactoryAction | None = Field(
        default=None,
        description="OpenEnv-style action envelope. Preferred for strict compatibility.",
    )
    joint_targets: list[float] = Field(default_factory=lambda: [0.0] * ACTION_DIMENSION)

    @model_validator(mode="after")
    def _normalize_action(self) -> "StepRequest":
        if self.action is not None:
            self.joint_targets = list(self.action.joint_targets)
        else:
            self.action = EgocentricFactoryAction(joint_targets=list(self.joint_targets))
        return self


class StateEnvelope(BaseModel):
    state: EgocentricFactoryState


ACTIVE_ENV: EgocentricFactoryCompetitionEnv | None = None


def _serialize_reset(observation: EgocentricFactoryObservation) -> ResetResponse:
    return ResetResponse(
        observation=observation.model_dump(mode="json"),
        reward=float(observation.reward or 0.0),
        done=bool(observation.done),
    )


def _serialize_step(observation: EgocentricFactoryObservation) -> StepResponse:
    return StepResponse(
        observation=observation.model_dump(mode="json"),
        reward=float(observation.reward or 0.0),
        done=bool(observation.done),
    )


def _get_active_env() -> EgocentricFactoryCompetitionEnv:
    if ACTIVE_ENV is None:
        raise HTTPException(status_code=400, detail="environment not initialized; call /reset first")
    return ACTIVE_ENV


def create_app() -> FastAPI:
    app = FastAPI(
        title="Egocentric Factory Competition Environment",
        version="0.2.0",
        description=(
            "Submission-ready OpenEnv environment for egocentric real-world manipulation tasks with "
            "three graded tasks and deterministic shaped rewards."
        ),
    )

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "name": "Egocentric Factory Competition Environment",
            "status": "ok",
            "tasks": [spec.task_id for spec in list_task_specs()],
            "endpoints": ["/reset", "/step", "/state", "/metadata", "/schema", "/tasks"],
            "backend_mode": DEFAULT_BACKEND_MODE,
            "simulation_origin": "Derived from the Egocentric-100K + MyoSuite/MuJoCo factory assembly stack and distilled into a shard-based surrogate_myo OpenEnv contract.",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tasks")
    def tasks() -> dict[str, object]:
        return {"tasks": [spec.to_dict() for spec in list_task_specs()]}

    @app.get("/metadata")
    def metadata() -> dict[str, object]:
        env = EgocentricFactoryCompetitionEnv()
        try:
            info = env.get_metadata()
            return {
                "environment": {
                    "name": info.name,
                    "description": info.description,
                    "version": info.version,
                    "author": info.author,
                    "backend_mode": DEFAULT_BACKEND_MODE,
                },
                "tasks": [spec.to_dict() for spec in list_task_specs()],
            }
        finally:
            env.close()

    @app.get("/schema")
    def schema() -> dict[str, object]:
        return {
            "action": EgocentricFactoryAction.model_json_schema(),
            "observation": EgocentricFactoryObservation.model_json_schema(),
            "state": EgocentricFactoryState.model_json_schema(),
        }

    @app.post("/reset", response_model=ResetResponse)
    def reset(req: ResetRequest | None = None) -> ResetResponse:
        global ACTIVE_ENV
        payload = req or ResetRequest()
        if ACTIVE_ENV is not None:
            ACTIVE_ENV.close()
        env = EgocentricFactoryCompetitionEnv(
            task_id=payload.task_id,
            seed=payload.seed,
            backend_mode=payload.backend_mode,
        )
        observation = env.reset(task_id=payload.task_id, seed=payload.seed, episode_id=payload.episode_id)
        ACTIVE_ENV = env
        return _serialize_reset(observation)

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest) -> StepResponse:
        env = _get_active_env()
        observation = env.step(req.action or EgocentricFactoryAction(joint_targets=req.joint_targets))
        return _serialize_step(observation)

    @app.get("/state", response_model=EgocentricFactoryState)
    def state() -> EgocentricFactoryState:
        return _get_active_env().state

    @app.get("/current_state", response_model=StateEnvelope)
    def current_state() -> StateEnvelope:
        return StateEnvelope(state=_get_active_env().state)

    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run("egocentric_dataset_test.competition.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
