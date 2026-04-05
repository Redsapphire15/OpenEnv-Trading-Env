# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Execution desk environment adapter for OpenEnv server runtime."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ExecutionDeskAction, ExecutionDeskObservation
except ImportError:
    from models import ExecutionDeskAction, ExecutionDeskObservation

try:
    from trading_env.openenv_quant.env.execution_env import ExecutionDeskEnv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "openenv_quant package is required. Keep it in the workspace or vendor its modules into trading_env."
    ) from exc


class TradingEnvironment(Environment):
    """Adapter that serves the execution desk simulation through OpenEnv contracts."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 7, max_steps: int = 60):
        self._seed = seed
        self._max_steps = max_steps
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = ExecutionDeskEnv(seed=seed, max_steps=max_steps)

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any
    ) -> ExecutionDeskObservation:
        options = kwargs.get("options")
        if options is not None and not isinstance(options, dict):
            options = None
        options_dict: Dict[str, Any] = dict(options or {})
        if "max_steps" in kwargs and "max_steps" not in options_dict:
            options_dict["max_steps"] = kwargs["max_steps"]

        observation, info = self._env.reset(seed=seed or self._seed, options=options_dict or None)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return ExecutionDeskObservation(
            observation=observation,
            info=info,
            done=False,
            reward=0.0,
            metadata={"stage": observation.get("task_stage")},
        )

    def step(self, action: ExecutionDeskAction) -> ExecutionDeskObservation:  # type: ignore[override]
        action_payload = action.model_dump(exclude_none=True)
        observation, reward, terminated, truncated, info = self._env.step(action_payload)
        self._state.step_count += 1
        done = bool(terminated or truncated)
        return ExecutionDeskObservation(
            observation=observation,
            info=info,
            done=done,
            reward=float(reward),
            metadata={
                "terminated": terminated,
                "truncated": truncated,
                "stage": observation.get("task_stage"),
            },
        )

    @property
    def state(self) -> State:
        return self._state
