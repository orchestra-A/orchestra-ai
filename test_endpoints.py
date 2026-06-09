"""Smoke tests for the Orchestra API production endpoints."""

import requests
BASE_URL = "https://orchestra-ai-36zm.onrender.com"
TIMEOUT = 120


def report(endpoint: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    line = f"{status} — {endpoint}"
    if detail and not passed:
        line += f" ({detail})"
    print(line)


def test_blueprint() -> dict | None:
    endpoint = "POST /blueprint"
    try:
        response = requests.post(
            f"{BASE_URL}/blueprint",
            json={"idea": "a todo app"},
            timeout=TIMEOUT,
        )
        data = response.json()
        passed = (
            response.ok
            and "tasks" in data
            and isinstance(data["tasks"], list)
        )
        report(endpoint, passed, f"status={response.status_code}")
        return data if passed else None
    except Exception as exc:
        report(endpoint, False, str(exc))
        return None


def test_assign(blueprint_data: dict | None) -> None:
    endpoint = "POST /assign"
    if not blueprint_data or "tasks" not in blueprint_data:
        report(endpoint, False, "skipped — blueprint failed")
        return

    try:
        response = requests.post(
            f"{BASE_URL}/assign",
            json={
                "tasks": blueprint_data["tasks"],
                "skills": {"Mitaali": ["Python", "AI"]},
            },
            timeout=TIMEOUT,
        )
        data = response.json()
        tasks = data.get("tasks", [])
        passed = (
            response.ok
            and isinstance(tasks, list)
            and len(tasks) > 0
            and all("assigned_to" in task for task in tasks)
        )
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_search() -> None:
    endpoint = "GET /search?question=database"
    try:
        response = requests.get(
            f"{BASE_URL}/search",
            params={"question": "database"},
            timeout=TIMEOUT,
        )
        data = response.json()
        passed = response.ok and "matches" in data
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_clover() -> None:
    endpoint = "POST /clover"
    try:
        response = requests.post(
            f"{BASE_URL}/clover",
            json={"question": "who is working on backend?"},
            timeout=TIMEOUT,
        )
        data = response.json()
        passed = response.ok and "answer" in data
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_standup() -> None:
    endpoint = "GET /standup"
    try:
        response = requests.get(f"{BASE_URL}/standup", timeout=TIMEOUT)
        data = response.json()
        passed = response.ok and "standup" in data
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_replan() -> None:
    endpoint = "GET /replan"
    try:
        response = requests.get(f"{BASE_URL}/replan", timeout=TIMEOUT)
        data = response.json()
        passed = response.ok and "suggestions" in data
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_tasks() -> None:
    endpoint = "GET /tasks"
    try:
        response = requests.get(f"{BASE_URL}/tasks", timeout=TIMEOUT)
        data = response.json()
        passed = response.ok and isinstance(data, list)
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_graph() -> None:
    endpoint = "GET /graph"
    try:
        response = requests.get(f"{BASE_URL}/graph", timeout=TIMEOUT)
        data = response.json()
        passed = response.ok and "nodes" in data and "edges" in data
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def test_onboarding() -> None:
    endpoint = "POST /onboarding"
    try:
        response = requests.post(
            f"{BASE_URL}/onboarding",
            json={"github_username": "mitaalisingh"},
            timeout=TIMEOUT,
        )
        data = response.json()
        passed = (
            response.ok
            and "profile" in data
            and "assignments" in data
        )
        report(endpoint, passed, f"status={response.status_code}")
    except Exception as exc:
        report(endpoint, False, str(exc))


def main() -> None:
    print(f"Testing Orchestra API at {BASE_URL}\n")
    blueprint_data = test_blueprint()
    test_assign(blueprint_data)
    test_search()
    test_clover()
    test_standup()
    test_replan()
    test_tasks()
    test_graph()
    test_onboarding()


if __name__ == "__main__":
    main()
