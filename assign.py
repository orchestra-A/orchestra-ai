"""Assign roadmap tasks to team members using Gemini."""

import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
BLUEPRINT_FILE = "blueprint.json"
SKILLS_FILE = "skills.json"
OUTPUT_FILE = "assigned.json"

PROMPT_TEMPLATE = """You are an engineering manager assigning work to team members.

Given the project blueprint and team skills below, assign each task to the most suitable
person based on skill match.

Blueprint JSON:
{blueprint}

Team skills JSON:
{skills}

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
      "assigned_to": "member_name"
    }}
  ]
}}

Rules:
- Keep all original tasks and task fields from the input blueprint.
- Add exactly one "assigned_to" for every task.
- "assigned_to" must be one of the member names from the team skills JSON.
- Choose the best fit based on relevant skills and task type.
- Preserve dependency ids exactly as provided.
- Output JSON only. No markdown, no prose, no extra keys outside the described structure.
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


def assign_tasks(blueprint: dict, skills: dict, api_key: str) -> dict:
    """Call Gemini to assign tasks to team members."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            blueprint=json.dumps(blueprint, ensure_ascii=False),
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


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    blueprint = load_json_file(BLUEPRINT_FILE)
    skills = load_json_file(SKILLS_FILE)

    assigned = assign_tasks(blueprint, skills, api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(assigned, file, indent=2, ensure_ascii=False)

    print(json.dumps(assigned, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
