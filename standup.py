"""Generate a daily standup update from assigned tasks."""

import json
import os

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from query import get_all_tasks

MODEL_NAME = "gemini-2.5-flash-lite"

PROMPT_TEMPLATE = """You are formatting a daily standup update for an engineering team.

Project: {project_name}

Team task status (grouped by person):
{grouped}

{recent_activity}

Write a clean daily standup update that reads like something you'd paste into a team chat (Slack/Discord).

Rules:
- One section per person, in a natural standup voice.
- For each person, cover: completed yesterday, in progress today, and blockers (if any).
- If recent team activity is provided, use it to add what people actually did recently to each person's standup entry.
- Mention task IDs and titles where relevant.
- Skip empty categories for a person (e.g. no "blocked" line if they have none).
- Keep it concise, friendly, and scannable.
- Do not use markdown headers or bullet-heavy formatting — plain team-chat prose is fine.
- Output only the standup message. No preamble or explanation.
"""


def group_tasks_by_person(tasks: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Group tasks by assignee and split by status."""
    grouped: dict[str, dict[str, list[dict]]] = {}

    for task in tasks:
        person = task.get("assigned_to", "Unassigned")
        status = task.get("status", "")

        if person not in grouped:
            grouped[person] = {
                "completed": [],
                "in_progress": [],
                "blocked": [],
            }

        summary = {
            "id": task.get("id"),
            "title": task.get("title"),
            "track": task.get("track"),
        }

        if status == "completed":
            grouped[person]["completed"].append(summary)
        elif status == "in_progress":
            grouped[person]["in_progress"].append(summary)
        elif status == "blocked":
            grouped[person]["blocked"].append(summary)

    return grouped


def fetch_live_events() -> list:
    """Fetch live Discord and GitHub events from the Orchestra backend."""
    # The backend has moved Render accounts several times; read the current URL
    # from env (BACKEND_URL) and fall back to the live deployment so this never
    # silently points at a dead host again.
    backend_url = os.getenv(
        "BACKEND_URL", "https://orchestra-backend-30fy.onrender.com"
    ).rstrip("/")
    try:
        response = requests.get(
            f"{backend_url}/events",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("total", 0) == 0:
            return []
        return data.get("events", [])
    except Exception:
        return []


def generate_standup(grouped: dict, project_name: str, api_key: str) -> str:
    """Send grouped task data to Gemini and return formatted standup."""
    client = genai.Client(api_key=api_key)

    grouped_json = json.dumps(grouped, indent=2, ensure_ascii=False)
    events = fetch_live_events()
    recent_activity = ""
    if events:
        events_json = json.dumps(events, indent=2, ensure_ascii=False)
        recent_activity = (
            f"Recent team activity (from Discord/GitHub):\n{events_json}"
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            project_name=project_name,
            grouped=grouped_json,
            recent_activity=recent_activity,
        ),
        config=types.GenerateContentConfig(temperature=0.4),
    )

    return (response.text or "").strip()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    tasks = get_all_tasks()
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    grouped = group_tasks_by_person(tasks)
    standup = generate_standup(grouped, "Orchestra", api_key)

    print(standup)


if __name__ == "__main__":
    main()
