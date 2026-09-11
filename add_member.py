"""Smart onboarding for a member added to an existing project.

When a teammate joins mid-project, everything is usually already assigned — so
to give the newcomer work you have to *rebalance*: move some not-yet-started
tasks onto them. This module does that in two halves:

  rank_fit()         — one Gemini call: which reassignable tasks suit the new
                       member's skills, best first.
  select_rebalance() — pure, deterministic: given the fit ranking and everyone's
                       current story-point load, pick the minimum set of tasks to
                       move so the newcomer reaches a fair share of the effort,
                       pulled from the most-loaded teammates and never dragging
                       anyone below the fair share.

Only `upcoming` / unassigned tasks are ever candidates (the caller enforces
that via query.reassignable_tasks) — in-progress work is never moved.
"""

import json
import re

from google import genai
from google.genai import types

from blueprint import DEFAULT_POINTS

MODEL_NAME = "gemini-2.5-flash-lite"

PROMPT_TEMPLATE = """A new member is joining a project. Decide which of the reassignable tasks
they are a good fit to take on, based on their skills.

New member: {name}
Their skills JSON:
{skills}

Reassignable tasks JSON (id, title, track, description):
{tasks}

Return ONLY a single valid JSON object with this exact shape:
{{
  "fit_ids": ["<task id>", "<task id>", ...]
}}

Rules:
- Include a task id ONLY if this member's skills make them a reasonable person to do that task.
- Order the ids best-fit FIRST.
- Omit tasks that clearly don't match their skills.
- Use ONLY ids from the reassignable tasks above — never invent one.
- Output JSON only. No markdown, no prose, no extra keys.
"""


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def rank_fit(
    name: str,
    skills: list[str],
    candidate_tasks: list[dict],
    api_key: str,
) -> list[str]:
    """Return candidate task ids the member can do, best-fit first (Gemini)."""
    slim = [
        {
            "id": t.get("id"),
            "title": t.get("title"),
            "track": t.get("track"),
            "description": t.get("description"),
        }
        for t in candidate_tasks
    ]
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            name=name,
            skills=json.dumps(skills, ensure_ascii=False),
            tasks=json.dumps(slim, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    result = json.loads(_extract_json(raw))
    valid_ids = {t.get("id") for t in candidate_tasks}
    return [tid for tid in result.get("fit_ids", []) if tid in valid_ids]


def select_rebalance(
    candidates: list[dict],
    fit_ids: list[str],
    loads: dict[str, int],
    num_devs_incl_new: int,
) -> list[str]:
    """Pick which candidate task ids to move to the new member. Pure function.

    candidates : reassignable tasks, each {id, points, assigned_to (str|None)}.
    fit_ids    : task ids the member can do, best-fit first (from rank_fit).
    loads      : current remaining story points per EXISTING developer.
    num_devs_incl_new : team size counting the new member (for the fair share).

    Strategy: aim the newcomer at a fair share = total remaining effort / team
    size. Take free (unassigned) fitting tasks first, then pull fitting tasks off
    the most-loaded teammates — but only from someone still above the fair share,
    so we never overload one person to relieve another. Stop once the newcomer
    reaches the fair share (a final task may cross it) or nothing else fits.
    """
    fit_set = set(fit_ids)
    fit_rank = {tid: i for i, tid in enumerate(fit_ids)}

    pts: dict[str, int] = {}
    owner: dict[str, str | None] = {}
    for c in candidates:
        tid = c.get("id")
        if tid not in fit_set:
            continue
        p = c.get("points")
        pts[tid] = int(p) if isinstance(p, (int, float)) else DEFAULT_POINTS
        a = c.get("assigned_to")
        owner[tid] = a or None

    loads = {k: (v or 0) for k, v in loads.items()}
    unassigned_pts = sum(pts[t] for t in pts if owner[t] is None)
    total_work = sum(loads.values()) + unassigned_pts
    fair_share = total_work / max(num_devs_incl_new, 1)

    moved: list[str] = []
    member_pts = 0.0
    remaining = set(pts.keys())
    INF = float("inf")

    def rank(tid: str) -> int:
        return fit_rank.get(tid, 1_000_000)

    def src_load(tid: str) -> float:
        # Unassigned tasks are "free" — treat them as the richest source so they
        # get taken before we ever pull work off a teammate.
        return INF if owner[tid] is None else loads.get(owner[tid], 0)

    # Fill the newcomer toward the fair share. Each step we only take a task that
    # keeps them AT OR UNDER the fair share (the team mean), so they can never end
    # up the most-loaded person — the one outcome a rebalance must avoid. Sources
    # are unassigned tasks (free) or a teammate still above the fair share, and we
    # always pull from the most-loaded source first. The one exception: if the
    # newcomer would otherwise get nothing, we let them take a single task even if
    # it overshoots, so adding a member never leaves them idle.
    while member_pts < fair_share:
        need = fair_share - member_pts
        eligible = [
            t
            for t in remaining
            if owner[t] is None or loads.get(owner[t], 0) > fair_share
        ]
        fitting = [t for t in eligible if pts[t] <= need]
        if fitting:
            pool = fitting
        elif not moved and eligible:
            pool = eligible  # idle-guard: allow one overshooting task
        else:
            break
        pool.sort(key=lambda t: (-src_load(t), -pts[t], rank(t)))
        tid = pool[0]
        moved.append(tid)
        member_pts += pts[tid]
        if owner[tid] is not None:
            loads[owner[tid]] = loads.get(owner[tid], 0) - pts[tid]
        remaining.discard(tid)

    return moved
