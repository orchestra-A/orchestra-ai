"""Generate a structured project blueprint from project details using Gemini."""

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


MODEL_NAME = "gemini-2.5-flash"

# Story points use the Fibonacci scale. Anything the model returns off-scale
# (4, 6, 0, a string, or nothing) is snapped to the nearest allowed value so the
# graph only ever stores a clean estimate — the capacity / velocity queries sum
# these, so a stray value would quietly skew the totals.
ALLOWED_POINTS = (1, 2, 3, 5, 8, 13)
DEFAULT_POINTS = 3


def normalize_points(raw) -> int:
    """Coerce any points value to the Fibonacci scale (1/2/3/5/8/13).

    Missing / non-numeric falls back to DEFAULT_POINTS; an in-range but
    off-scale number (e.g. 4, 6, 10) snaps to the nearest allowed value.
    """
    try:
        val = int(round(float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_POINTS
    if val in ALLOWED_POINTS:
        return val
    return min(ALLOWED_POINTS, key=lambda p: abs(p - val))

PROMPT_TEMPLATE = """You are a senior software architect. Break the following project
into a structured project blueprint.

Project name:
\"\"\"{name}\"\"\"

Description:
\"\"\"{description}\"\"\"

Tech stack:
\"\"\"{tech_stack}\"\"\"

Return ONLY a single valid JSON object. Do not include markdown fences, prose,
comments, or any text outside the JSON. The JSON must match this exact schema:

{{
  "project_name": "string",
  "summary": "string",
  "tasks": [
    {{
      "id": "{project_id}-T1",
      "title": "string",
      "track": "string",
      "description": "string",
      "dependencies": ["{project_id}-T0", "..."],
      "status": "upcoming",
      "priority": "high" | "medium" | "low",
      "points": 3,
      "deadline": "2026-06-15T00:00:00+00:00",
      "project_id": "{project_id}",
      "created_at": "2026-06-02T14:35:00+00:00",
      "updated_at": "2026-06-02T14:35:00+00:00",
      "platform": "github"
    }}
  ]
}}

Rules:
- Use short stable ids like {project_id}-T1, {project_id}-T2, {project_id}-T3 ... always prefix every task id with the project_id
- "track" should be automatically chosen based on the type of work involved in each task.
- Use concise, descriptive track names such as "frontend", "backend", "AI", "mobile", "devops", "database", "design", "qa", "security", etc.
- Keep track naming consistent across tasks.
- "dependencies" is an array of task ids this task depends on (use [] if none). Where tasks can be done independently and in parallel, model them with no dependency on each other — do not chain tasks sequentially unless one truly cannot start before another finishes.
- "status" must always be "upcoming".
- "priority" must be exactly one of: "high", "medium", "low", based on task importance.
- "points" is an agile story-point estimate of relative effort/complexity — NOT calendar time. Use the Fibonacci scale and pick exactly one of: 1 (trivial), 2, 3, 5, 8, 13 (very large or highly uncertain). Judge it on scope, unknowns, and how many moving parts the task touches. Most tasks should land at 2, 3, or 5; reserve 8 and 13 for genuinely large or ambiguous work, and 1 for near-trivial changes.
- "deadline" must be an ISO 8601 datetime string. Estimate a realistic deadline for each task starting from today, taking into account task complexity, priority, and dependencies — higher priority and tasks with no dependencies should have earlier deadlines. Spread deadlines across the project timeline so the full project completes within a reasonable timeframe.
- "project_id" must always be "{project_id}".
- "project_name" should match the given project name.
- Prefer the listed tech stack when shaping tasks; cover the full stack needed to ship (UI, APIs, data, AI/ML pieces as relevant).
- "summary" must be a short, plain-English explanation (3-5 sentences) of how you broke the work down and why — written so a teammate can quickly grasp the reasoning behind the task order and structure without reading every task.
- Output JSON only. The "summary" field is the only place for explanation — do not add any text outside the JSON object itself."""


def _remove_cycles(tasks: list[dict]) -> list[dict]:
    """Remove dependency edges that form cycles. Mutates tasks in place and returns them."""
    deps: dict[str, list[str]] = {t["id"]: list(t.get("dependencies", [])) for t in tasks}
    valid_ids = set(deps.keys())

    visited: set[str] = set()
    in_stack: set[str] = set()
    to_remove: list[tuple[str, str]] = []

    def dfs(node: str) -> None:
        if node in in_stack:
            return
        if node in visited:
            return
        visited.add(node)
        in_stack.add(node)
        for dep in list(deps.get(node, [])):
            if dep not in valid_ids:
                to_remove.append((node, dep))
            elif dep in in_stack:
                to_remove.append((node, dep))
            else:
                dfs(dep)
        in_stack.discard(node)

    for task_id in list(deps.keys()):
        if task_id not in visited:
            dfs(task_id)

    for task_id, dep_id in to_remove:
        if dep_id in deps.get(task_id, []):
            deps[task_id].remove(dep_id)

    task_map = {t["id"]: t for t in tasks}
    for task_id, clean_deps in deps.items():
        if task_id in task_map:
            task_map[task_id]["dependencies"] = clean_deps

    return tasks


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


def _new_project_id() -> str:
    return f"P{uuid.uuid4().hex[:8]}"


def generate_blueprint(
    name: str,
    description: str,
    tech_stack: list[str] | None = None,
    project_id: str | None = None,
) -> dict:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    # Reuse the caller's project_id when regenerating an existing project (the
    # frontend's "modify" flow), so the same project is overwritten in place
    # instead of a brand-new one being minted every time — which is what caused
    # duplicate projects. Only mint a fresh id when none is supplied (new project).
    project_id = project_id.strip() if project_id and project_id.strip() else _new_project_id()
    stack = tech_stack or []
    tech_stack_text = ", ".join(stack) if stack else "(not specified)"

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            name=name,
            description=description,
            tech_stack=tech_stack_text,
            project_id=project_id,
        ),
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
        task["project_id"] = project_id
        task["points"] = normalize_points(task.get("points"))

    blueprint["project_name"] = name
    blueprint["project_id"] = project_id
    blueprint["tasks"] = _remove_cycles(blueprint.get("tasks", []))
    return blueprint


def main() -> None:
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
        name = "CLI Project"
        tech_stack: list[str] = []
    else:
        name = "Fridge Recipe App"
        description = (
            "A mobile app that uses AI to suggest recipes from a photo of your fridge."
        )
        tech_stack = ["React Native", "Python", "Gemini"]

    blueprint = generate_blueprint(name, description, tech_stack)
    print(json.dumps(blueprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
