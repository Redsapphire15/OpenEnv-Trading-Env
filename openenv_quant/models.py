from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None)
    options: Optional[Dict[str, Any]] = Field(default=None)


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    env_name: str
