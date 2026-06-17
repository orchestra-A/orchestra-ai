"""Generate re-planning suggestions for blocked tasks."""

import json
import os

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from assign import fetch_skills_from_neo4j
from query import get_all_tasks

MODEL_NAME = "gemini-2.5-flash-lite"

PROMPT_TEMPLATE = """You are an engineering delivery lead helping re-plan work around blockers.

Project: {project_name}

Blocked task:
{blocked_task}

Tasks impacted because they depend on this blocked task:
{dependents}

Team tasks snapshot:
{all_tasks}

Team workload (tasks per person):
{workload}

Team skills:
{skills}

Suggest a practical re-plan for this blocker. Return ONLY a valid JSON object with this shape:
{{
  "blocked_task_id": "T1",
  "blocked_task_title": "string",
  "takeover_suggestion": "who could take over and why",
  "shift_suggestion": "what work can be shifted/re-sequenced now",
  "project_impact": "timeline/scope risk impact in concise terms"
}}

Rules:
- Keep suggestions realistic based on assignees, tracks, and dependencies.
- Use workload and skills to suggest the most suitable person to reassign to — prefer people with lower workload and matching skills.
- Mention specific people/task IDs where useful.
- Keep each field concise and actionable.
- Output JSON only, no markdown or extra text.
"""


def find_blocked_tasks(tasks: list[dict]) -> list[dict]:
    """Return tasks currently marked as blocked."""
    return [task for task in tasks if task.get("status") == "blocked"]


def find_dependents(blocked_task_id: str, tasks: list[dict]) -> list[dict]:
    """Find tasks that depend on a given blocked task."""
    dependents: list[dict] = []
    for task in tasks:
        deps = [str(dep) for dep in task.get("dependencies", [])]
        if blocked_task_id in deps:
            dependents.append(task)
    return dependents


def fetch_project_workload() -> dict:
    """Fetch task counts per person from the Orchestra project API."""
    try:
        response = requests.get(
            "https://orchestra-ai-36zm.onrender.com/project",
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("by_person", {})
    except Exception:
        return {}


def suggest_replan(
    blocked_task: dict, dependents: list[dict], all_tasks: list[dict], project_name: str, api_key: str
) -> dict:
    """Call Gemini for one blocked task re-plan suggestion."""
    client = genai.Client(api_key=api_key)
    workload = fetch_project_workload()
    skills = fetch_skills_from_neo4j()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            project_name=project_name,
            blocked_task=json.dumps(blocked_task, ensure_ascii=False, indent=2),
            dependents=json.dumps(dependents, ensure_ascii=False, indent=2),
            all_tasks=json.dumps(all_tasks, ensure_ascii=False),
            workload=json.dumps(workload, ensure_ascii=False, indent=2),
            skills=json.dumps(skills, ensure_ascii=False, indent=2),
        ),
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    return json.loads((response.text or "").strip())


def print_suggestions(suggestions: list[dict]) -> None:
    """Print re-planning suggestions cleanly."""
    print("\nRe-Planning Suggestions")
    print("=" * 50)

    if not suggestions:
        print("No blocked tasks found. No re-planning needed.")
        return

    for i, item in enumerate(suggestions, start=1):
        print(f"\n{i}. [{item.get('blocked_task_id', 'N/A')}] {item.get('blocked_task_title', '')}")
        print(f"   Takeover: {item.get('takeover_suggestion', '')}")
        print(f"   Shift:    {item.get('shift_suggestion', '')}")
        print(f"   Impact:   {item.get('project_impact', '')}")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    tasks = get_all_tasks()
    project_name = "Orchestra"
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    blocked_tasks = find_blocked_tasks(tasks)

    suggestions: list[dict] = []
    for blocked_task in blocked_tasks:
        blocked_id = str(blocked_task.get("id", ""))
        dependents = find_dependents(blocked_id, tasks)
        suggestions.append(
            suggest_replan(blocked_task, dependents, tasks, project_name, api_key)
        )

    print_suggestions(suggestions)


if __name__ == "__main__":
    main()
