"""Analyze skill gaps between assigned tasks and team capabilities."""

import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
ASSIGNED_FILE = "assigned.json"
SKILLS_FILE = "skills.json"
OUTPUT_FILE = "skill_gap_report.json"

PROMPT_TEMPLATE = """You are a technical lead reviewing whether the team can complete all assigned tasks.

Assigned tasks JSON:
{tasks}

Team skill profiles JSON:
{skills}

For each task, determine whether at least one team member has sufficient skills to complete it.
Consider task title, description, track, assigned_to, and each member's listed skills.

Return ONLY a single valid JSON object with this exact shape:
{{
  "project_name": "string",
  "tasks": [
    {{
      "id": "T1",
      "title": "string",
      "track": "string",
      "description": "string",
      "dependencies": ["T0"],
      "assigned_to": "member_name",
      "gap_detected": false,
      "missing_skill_or_role": null
    }}
  ]
}}

Rules:
- Keep all original task fields from the input (id, title, track, description, dependencies, assigned_to).
- Add "gap_detected" (boolean) for every task.
- Set "gap_detected" to true only when no team member has sufficient skills for that task.
- When "gap_detected" is true, set "missing_skill_or_role" to a concise suggestion (e.g. "mobile developer", "DevOps engineer", "ML engineer").
- When "gap_detected" is false, set "missing_skill_or_role" to null.
- Evaluate skills strictly based on the team profiles provided.
- Output JSON only. No markdown, no prose, no extra keys.
"""


def extract_json(text: str) -> str:
    """Extract JSON object from model output."""
    cleaned = text.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def load_json_file(path: str) -> dict:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def analyze_skill_gaps(assigned: dict, skills: dict, api_key: str) -> dict:
    """Call Gemini to detect skill gaps across all tasks."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            tasks=json.dumps(assigned, ensure_ascii=False),
            skills=json.dumps(skills, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    raw = response.text or ""
    payload = extract_json(raw)
    return json.loads(payload)


def print_flagged_summary(report: dict) -> None:
    """Print a clean summary of tasks with detected skill gaps."""
    tasks = report.get("tasks", [])
    flagged = [task for task in tasks if task.get("gap_detected") is True]

    print(f"\nSkill Gap Summary — {report.get('project_name', 'Project')}")
    print("=" * 50)

    if not flagged:
        print("No skill gaps detected. All tasks are covered by the team.")
        return

    print(f"{len(flagged)} task(s) flagged:\n")

    for i, task in enumerate(flagged, start=1):
        print(f"{i}. [{task.get('id', 'N/A')}] {task.get('title', '')}")
        print(f"   Track:       {task.get('track', '')}")
        print(f"   Assigned to: {task.get('assigned_to', '')}")
        print(f"   Missing:     {task.get('missing_skill_or_role', 'N/A')}")
        print(f"   Description: {task.get('description', '')}\n")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    assigned = load_json_file(ASSIGNED_FILE)
    skills = load_json_file(SKILLS_FILE)

    if not assigned.get("tasks"):
        raise RuntimeError("No tasks found in assigned.json.")
    if not skills:
        raise RuntimeError("No team profiles found in skills.json.")

    report = analyze_skill_gaps(assigned, skills, api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print_flagged_summary(report)
    print(f"Full report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
