from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_env.openenv_quant.env.execution_env import heuristic_policy, run_demo


if __name__ == "__main__":
    run_demo(policy=heuristic_policy)
