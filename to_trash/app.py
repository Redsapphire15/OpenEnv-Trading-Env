from __future__ import annotations

from fastapi import FastAPI

from server.core.env.execution_desk_env import ExecutionDeskEnv
from models import HealthResponse, ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse


app = FastAPI(title="Execution Desk Assistant", version="0.1.0")
ENV = ExecutionDeskEnv(seed=7, max_steps=60)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", env_name="execution_desk_assistant")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", env_name="execution_desk_assistant")


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    observation, info = ENV.reset(seed=request.seed, options=request.options)
    return ResetResponse(observation=observation, info=info)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    observation, reward, terminated, truncated, info = ENV.step(request.action)
    return StepResponse(
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    current = ENV.state()
    return StateResponse(observation=current["observation"], info=current["info"])
