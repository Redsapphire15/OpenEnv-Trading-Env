from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trading_env.openenv_quant.utils.constants import BROKERS, Stage


@dataclass
class Order:
    order_id: int
    side: str
    requested_size: int
    remaining_size: int
    broker: str
    urgency: str
    status: str = "working"
    average_fill_price: float = 0.0
    filled_size: int = 0
    rejection_reason: Optional[str] = None
    created_step: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "side": self.side,
            "requested_size": self.requested_size,
            "remaining_size": self.remaining_size,
            "broker": self.broker,
            "urgency": self.urgency,
            "status": self.status,
            "average_fill_price": round(self.average_fill_price, 4),
            "filled_size": self.filled_size,
            "rejection_reason": self.rejection_reason,
            "created_step": self.created_step,
        }


@dataclass
class ScenarioState:
    stage: Stage = Stage.DATA_VALIDATION
    step_count: int = 0
    now_minute: int = 0
    max_steps: int = 60
    current_broker: str = BROKERS[0]
    issue_log: List[str] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    data_truth: Dict[str, Any] = field(default_factory=dict)
    data_anomalies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_truth: Dict[str, Any] = field(default_factory=dict)
    execution_truth: Dict[str, Any] = field(default_factory=dict)
    outstanding_orders: Dict[int, Order] = field(default_factory=dict)
    next_order_id: int = 1
    tool_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_failures: Dict[str, int] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    completed_flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "data_ready": False,
            "systems_ready": False,
            "execution_complete": False,
        }
    )
