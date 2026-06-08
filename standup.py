"""Generate a daily standup update from assigned tasks."""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
ASSIGNED_FILE = "assigned.json"

PROMPT_TEMPLATE = """You are formatting a daily standup update for an engineering team.

Project: {project_name}

Team task status (grouped by person):
{grouped}

Write a clean daily standup update that reads like something you'd paste into a team chat (Slack/Discord).

Rules:
- One section per person, in a natural standup voice.
- For each person, cover: completed yesterday, in progress today, and blockers (if any).
- Mention task IDs and titles where relevant.
- Skip empty categories for a person (e.g. no "blocked" line if they have none).
- Keep it concise, friendly, and scannable.
- Do not use markdown headers or bullet-heavy formatting — plain team-chat prose is fine.
- Output only the standup message. No preamble or explanation.
"""


def load_assigned(path: str) -> dict:
    """Load assigned tasks from JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def generate_standup(grouped: dict, project_name: str, api_key: str) -> str:
    """Send grouped task data to Gemini and return formatted standup."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            project_name=project_name,
            grouped=json.dumps(grouped, indent=2, ensure_ascii=False),
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

    assigned = load_assigned(ASSIGNED_FILE)
    tasks = assigned.get("tasks", [])
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    grouped = group_tasks_by_person(tasks)
    standup = generate_standup(
        grouped, assigned.get("project_name", "Project"), api_key
    )

    print(standup)


if __name__ == "__main__":
    main()
