"""Generate a structured project blueprint from a raw app idea using Gemini."""

import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


MODEL_NAME = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are a senior software architect. Break the following app idea
into a structured project blueprint.

App idea:
\"\"\"{idea}\"\"\"

Return ONLY a single valid JSON object. Do not include markdown fences, prose,
comments, or any text outside the JSON. The JSON must match this exact schema:

{{
  "project_name": "string",
  "summary": "string",
  "tasks": [
    {{
      "id": "T1",
      "title": "string",
      "track": "string",
      "description": "string",
      "dependencies": ["T0", "..."],
      "status": "todo",
      "priority": "high" | "medium" | "low",
      "project_id": "P1",
      "created_at": "2026-06-02T14:35:00+00:00",
      "updated_at": "2026-06-02T14:35:00+00:00",
      "platform": "github"
    }}
  ]
}}

Rules:
- Use short stable ids like T1, T2, T3 ...
- "track" should be automatically chosen based on the type of work involved in each task.
- Use concise, descriptive track names such as "frontend", "backend", "AI", "mobile", "devops", "database", "design", "qa", "security", etc.
- Keep track naming consistent across tasks.
- "dependencies" is an array of task ids this task depends on (use [] if none).
- "status" must always be "todo".
- "priority" must be exactly one of: "high", "medium", "low", based on task importance.
- "project_id" must always be "P1".
- Cover the full stack needed to ship the idea (UI, APIs, data, AI/ML pieces).
- "summary" must be a short, plain-English explanation (3-5 sentences) of how you broke the work down and why — written so a teammate can quickly grasp the reasoning behind the task order and structure without reading every task.
- Output JSON only. The "summary" field is the only place for explanation — do not add any text outside the JSON object itself."""


def extract_json(text: str) -> str:
    """Strip code fences or stray prose and return the JSON substring."""
    cleaned = text.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def generate_blueprint(idea: str) -> dict:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(idea=idea),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    payload = extract_json(raw)
    blueprint = json.loads(payload)

    now_iso = datetime.now(timezone.utc).isoformat()
    for task in blueprint.get("tasks", []):
        task["created_at"] = now_iso
        task["updated_at"] = now_iso
        task["platform"] = "github"

    return blueprint


def main() -> None:
    if len(sys.argv) > 1:
        idea = " ".join(sys.argv[1:])
    else:
        idea = "A mobile app that uses AI to suggest recipes from a photo of your fridge."

    blueprint = generate_blueprint(idea)
    print(json.dumps(blueprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
