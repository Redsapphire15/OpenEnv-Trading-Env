import requests

BASE_URL = "http://0.0.0.0:8000"

resp = requests.post(f"{BASE_URL}/reset")
obs = resp.json()
print("Initial Observation:", obs)

action = {
    "action": {
        "action_type": "CALL_TOOL",
        "tool_name": "bloomberg_pull",
        "params": {"issue_id": 1}
    }
}

resp = requests.post(f"{BASE_URL}/step", json=action)
result = resp.json()
print("Step Result:", result)