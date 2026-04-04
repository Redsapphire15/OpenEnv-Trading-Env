from openenv_quant.tools.bloomberg_pull import execute as bloomberg_pull
from openenv_quant.tools.cancel_order import execute as cancel_order
from openenv_quant.tools.change_broker import execute as change_broker
from openenv_quant.tools.compliance_recheck import execute as compliance_recheck
from openenv_quant.tools.compliance_verify import execute as compliance_verify
from openenv_quant.tools.escalate_issue import execute as escalate_issue
from openenv_quant.tools.get_current_position import execute as get_current_position
from openenv_quant.tools.internal_report_fetch import execute as internal_report_fetch
from openenv_quant.tools.market_status_check import execute as market_status_check
from openenv_quant.tools.oms_position_check import execute as oms_position_check
from openenv_quant.tools.ping_oms_connection import execute as ping_oms_connection
from openenv_quant.tools.restart_strategy import execute as restart_strategy
from openenv_quant.tools.risk_system_check import execute as risk_system_check
from openenv_quant.tools.split_order import execute as split_order
from openenv_quant.tools.strategy_health_check import execute as strategy_health_check
from openenv_quant.tools.submit_order import execute as submit_order

TOOL_REGISTRY = {
    "bloomberg_pull": bloomberg_pull,
    "oms_position_check": oms_position_check,
    "risk_system_check": risk_system_check,
    "compliance_verify": compliance_verify,
    "internal_report_fetch": internal_report_fetch,
    "market_status_check": market_status_check,
    "ping_oms_connection": ping_oms_connection,
    "strategy_health_check": strategy_health_check,
    "compliance_recheck": compliance_recheck,
    "restart_strategy": restart_strategy,
    "escalate_issue": escalate_issue,
    "submit_order": submit_order,
    "split_order": split_order,
    "cancel_order": cancel_order,
    "change_broker": change_broker,
    "get_current_position": get_current_position,
}

__all__ = ["TOOL_REGISTRY"]
