from __future__ import annotations

from typing import Any, Dict

from trading_env.server.core.env.base_state import ScenarioState


def evaluate_execution_complete(state: ScenarioState) -> Dict[str, Any]:
    execution = state.execution_truth
    error = abs(execution["target_position"] - execution["current_position"])
    working_orders = [order for order in state.outstanding_orders.values() if order.status == "working"]
    return {"ready": error <= execution["tolerance"] and not working_orders, "tracking_error": error}
