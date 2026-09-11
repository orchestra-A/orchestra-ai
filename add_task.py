"""Smart placement for a single task added to an existing project.

Where blueprint.py designs a whole project cold and assign.py distributes a full
task list, this module answers a narrower question: given ONE new task and the
project that already exists, where does it fit? One Gemini call returns the
assignee (best skill match), a story-point estimate, a track consistent with the
project's existing tracks, and any dependencies on tasks already in the project.

This is the AI-side value of "add task" — the roadmap is never regenerated, and
nothing already in the graph is touched; we only slot the new task in.
"""

import json

from google import genai
from google.genai import types

from blueprint import normalize_points

MODEL_NAME = "gemini-2.5-flash-lite"

PROMPT_TEMPLATE = """You are an engineering manager slotting ONE new task into an existing project.

The project's EXISTING tasks (for context — do NOT modify these, only refer to them):
{existing_tasks}

Team skills JSON (assign the new task to exactly one of these people):
{skills}

The NEW task to place:
{new_task}

Decide four things for the NEW task and return ONLY a single valid JSON object
with this exact shape:
{{
  "assigned_to": "member_name",
  "track": "string",
  "points": 3,
  "dependencies": ["<existing task id>", "..."]
}}

Rules:
- "assigned_to" MUST be exactly one of the member names in the team skills JSON — pick the best skill match for this task. Prefer someone who is not already the most loaded, all else equal.
- "track" should match the naming of the existing tasks' tracks when the work is the same kind (e.g. reuse "frontend", "backend", "AI"); only introduce a new track name if none fit.
- "points" is an agile story-point estimate of relative effort/complexity (NOT calendar time). Pick exactly one of the Fibonacci values: 1, 2, 3, 5, 8, 13. Most tasks are 2-5; reserve 8/13 for large or ambiguous work, 1 for near-trivial.
- "dependencies" is an array of EXISTING task ids (from the existing tasks above) that this new task genuinely depends on — i.e. work that must finish before it can start. Use [] if it can start independently. NEVER invent an id that is not in the existing tasks list.
- Output JSON only. No markdown, no prose, no extra keys.
"""


def _extract_json(text: str) -> str:
    """Return the JSON substring from model output (mirrors assign.extract_json)."""
    import re

    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def place_task(
    new_task: dict,
    existing_tasks: list[dict],
    skills: dict,
    api_key: str,
) -> dict:
    """Ask Gemini where a single new task fits in an existing project.

    Returns a dict with keys: assigned_to, track, points, dependencies. The
    caller is responsible for validating the assignee/dependencies against the
    real roster and task ids (the model can still name something that isn't
    there); this function just normalises points to the Fibonacci scale.
    """
    # Only feed the model the fields it needs to reason about placement — keep
    # the context small so the lite model stays fast and cheap.
    slim_existing = [
        {
            "id": t.get("id"),
            "title": t.get("title"),
            "track": t.get("track"),
            "description": t.get("description"),
        }
        for t in existing_tasks
    ]
    slim_new = {
        "title": new_task.get("title"),
        "description": new_task.get("description"),
        "track": new_task.get("track") or "",
    }

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            existing_tasks=json.dumps(slim_existing, ensure_ascii=False),
            skills=json.dumps(skills, ensure_ascii=False),
            new_task=json.dumps(slim_new, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    result = json.loads(_extract_json(raw))

    return {
        "assigned_to": result.get("assigned_to"),
        "track": (result.get("track") or "").strip() or new_task.get("track"),
        "points": normalize_points(result.get("points")),
        "dependencies": [d for d in result.get("dependencies", []) if isinstance(d, str)],
    }
