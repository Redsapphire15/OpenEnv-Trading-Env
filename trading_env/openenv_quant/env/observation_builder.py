from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from trading_env.openenv_quant.env.base_state import ScenarioState
from trading_env.openenv_quant.tasks.task1_data_verification import evaluate_data_readiness
from trading_env.openenv_quant.tasks.task2_system_monitoring import evaluate_system_readiness
from trading_env.openenv_quant.tasks.task3_execution_assistance import evaluate_execution_complete


def build_observation(state: ScenarioState) -> Dict[str, Any]:
    execution = state.execution_truth
    return {
        "task_stage": state.stage.value,
        "known_data": copy.deepcopy(state.tool_outputs),
        "system_status": {
            "oms_connected": state.system_truth["oms_connected"],
            "strategy_status": state.system_truth["strategy_status"],
            "escalated": state.escalated,
            "current_broker": state.current_broker,
        },
        "compliance_flags": {
            "compliance_ok": state.system_truth["compliance_ok"],
            "restricted": state.data_truth["restricted"],
            "escalation_reason": state.escalation_reason,
        },
        "position_state": {
            "current_position": execution["current_position"],
            "target_position": execution["target_position"],
            "tolerance": execution["tolerance"],
            "tracking_error": abs(execution["target_position"] - execution["current_position"]),
            "recent_slippage_bps": execution["recent_slippage_bps"],
        },
        "order_state": {
            "outstanding_orders": [order.snapshot() for order in state.outstanding_orders.values()],
            "open_order_count": sum(1 for order in state.outstanding_orders.values() if order.status == "working"),
            "recent_fills": execution["fills"][-5:],
        },
        "timestamps": {
            "sim_minute": state.now_minute,
            "step_count": state.step_count,
        },
    }


def build_info(state: ScenarioState, recency_limit_minutes: int, event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event = event or {}
    return {
        "completed_flags": copy.deepcopy(state.completed_flags),
        "issue_log": list(state.issue_log),
        "data_validation": evaluate_data_readiness(state, recency_limit_minutes),
        "system_readiness": evaluate_system_readiness(state),
        "execution_status": evaluate_execution_complete(state),
        "event": copy.deepcopy(event),
    }
