from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parent


def check_env_vars() -> None:
    missing = [name for name in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def check_openenv_spec() -> None:
    spec_path = ROOT / "openenv.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    required_keys = {"name", "entrypoint", "env_class", "endpoints", "models"}
    if not required_keys.issubset(spec):
        raise RuntimeError("openenv.yaml is missing required keys")


def check_api() -> None:
    from trading_env.openenv_quant import app as app_module
    from trading_env.openenv_quant.models import ResetRequest, StepRequest

    paths = {route.path for route in app_module.app.routes}
    for required_path in ["/health", "/reset", "/step", "/state"]:
        if required_path not in paths:
            raise RuntimeError(f"Missing API route: {required_path}")

    health_response = app_module.health()
    if health_response.status != "ok":
        raise RuntimeError("Health endpoint did not return ok")

    reset_response = app_module.reset(ResetRequest(seed=7, options=None))
    if not reset_response.observation:
        raise RuntimeError("Reset endpoint did not return an observation")

    state_response = app_module.state()
    if not state_response.observation:
        raise RuntimeError("State endpoint did not return an observation")

    step_response = app_module.step(
        StepRequest(action={"action_type": "CALL_TOOL", "tool_name": "bloomberg_pull"})
    )
    if not isinstance(step_response.reward, float):
        raise RuntimeError("Step endpoint did not return a float reward")


def check_inference() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "inference.py")],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        timeout=1200,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("[START]") or not lines[-1].startswith("[END]"):
        raise RuntimeError("Inference logs do not follow [START]/[STEP]/[END] framing")
    if len([line for line in lines if line.startswith("[STEP]")]) == 0:
        raise RuntimeError("Inference did not emit any [STEP] lines")


def check_graders() -> None:
    from trading_env.openenv_quant.graders.task_graders import run_all_graders

    scores = run_all_graders(seed=7)
    repeated_scores = run_all_graders(seed=7)
    if len(scores) < 3:
        raise RuntimeError("Expected at least 3 graders")
    if scores != repeated_scores:
        raise RuntimeError("Graders are not deterministic for the same seed")
    for task_name, score in scores.items():
        if not (0.0 <= score <= 1.0):
            raise RuntimeError(f"Out of range grader score for {task_name}: {score}")
    seed_sweep = [run_all_graders(seed=seed) for seed in [1, 2, 3, 7]]
    if len({json.dumps(item, sort_keys=True) for item in seed_sweep}) == 1:
        raise RuntimeError("Graders appear degenerate across seeds")


def check_space_ping() -> bool:
    space_url = os.getenv("SPACE_URL")
    if not space_url:
        return False
    with httpx.Client(timeout=10.0) as client:
        response = client.get(space_url.rstrip("/") + "/health")
        if response.status_code != 200:
            raise RuntimeError(f"Space health ping failed with status {response.status_code}")
    return True


def check_docker_build() -> bool:
    if os.getenv("RUN_DOCKER_CHECK") != "1":
        return False
    subprocess.run(
        ["docker", "build", "-t", "openenv-quant-check", str(ROOT)],
        cwd=ROOT.parent,
        check=True,
        timeout=1200,
    )
    return True


def main() -> None:
    check_env_vars()
    check_openenv_spec()
    check_api()
    check_inference()
    check_graders()
    checks = ["env", "openenv_spec", "api", "inference", "graders"]
    if check_space_ping():
        checks.append("space_ping")
    if check_docker_build():
        checks.append("docker_build")
    print(json.dumps({"status": "ok", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
