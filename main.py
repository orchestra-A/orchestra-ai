"""FastAPI server for Orchestra + Clover."""

import json
import os
import re
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

from assign import assign_tasks, fetch_skills_from_neo4j
from blueprint import extract_json, generate_blueprint
from ingest import ingest_all
from skill_gap import analyze_skill_gaps
from clover import answer_question, stream_answer
from commit_intel import fetch_live_events, main as run_commit_intel
from graph_query import build_reactflow_graph, merge_developer_skills
from onboarding import build_profile
from query import get_all_tasks
from search import (
    ensure_indexed,
    get_embedding,
    invalidate_index,
)
from re_planner import (
    find_blocked_tasks,
    find_dependents,
    suggest_replan,
)
from standup import generate_standup, group_tasks_by_person


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    api_key = os.getenv("INTERNAL_API_KEY")
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


load_dotenv()

app = FastAPI(title="Orchestra + Clover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "endpoints": 12,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/team")
def get_team() -> dict[str, Any]:
    try:
        skills = fetch_skills_from_neo4j()
        if not skills:
            raise HTTPException(
                status_code=404, detail="No team skills found in Neo4j"
            )
        team = [
            {"name": name, "skills": skill_list}
            for name, skill_list in skills.items()
        ]
        return {"team": team}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class ManualSkillsRequest(BaseModel):
    name: str
    skills: list[str]


@app.post("/team/manual", dependencies=[Depends(verify_api_key)])
def add_manual_skills(body: ManualSkillsRequest) -> dict[str, Any]:
    """Manually add/correct a developer's skills in Neo4j.

    Human fallback for `onboarding.py` (which only infers skills from public
    GitHub). Merges — never overwrites: new skills are added to the developer's
    existing HAS_SKILL set without removing or duplicating any. Returns the
    developer's full current skill list after the merge.
    """
    name = body.name.strip()
    skills = [s.strip() for s in body.skills if s and s.strip()]

    if not name:
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if not skills:
        raise HTTPException(status_code=400, detail="skills cannot be empty.")

    try:
        return merge_developer_skills(name, skills)
    except RuntimeError as exc:  # missing NEO4J_* env vars
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # Neo4j unavailable / query failure
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


@app.get("/project")
def get_project() -> dict[str, Any]:
    try:
        tasks = get_all_tasks()
        by_status: dict[str, int] = {}
        by_person: dict[str, int] = {}
        by_track: dict[str, int] = {}

        for task in tasks:
            status = str(task.get("status", "unknown"))
            person = str(task.get("assigned_to", "Unassigned"))
            track = str(task.get("track", "unknown"))

            by_status[status] = by_status.get(status, 0) + 1
            by_person[person] = by_person.get(person, 0) + 1
            by_track[track] = by_track.get(track, 0) + 1

        return {
            "project_name": "Orchestra",
            "total_tasks": len(tasks),
            "by_status": by_status,
            "by_person": by_person,
            "by_track": by_track,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class BlueprintRequest(BaseModel):
    name: str
    description: str
    tech_stack: list[str] = []
    members: list[str] = []
    created_by: str = ""
    # Optional. When the frontend regenerates an existing project (its "modify"
    # flow), it passes that project's id here so we overwrite it in place instead
    # of minting a new one — without this, every modify created a duplicate
    # project. Omitted/blank for a brand-new project (a fresh id is generated).
    project_id: str | None = None


class AssignRequest(BaseModel):
    tasks: list[dict[str, Any]]
    skills: dict[str, list[str]]


class CloverRequest(BaseModel):
    question: str
    conversation_history: list[dict] = []
    project_id: str | None = None
    github_username: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    pending_action: dict | None = None


class OnboardingRequest(BaseModel):
    github_username: str


class TaskStatusRequest(BaseModel):
    status: str


class TaskEditRequest(BaseModel):
    # All optional — a PATCH may touch just one field. Status is intentionally
    # excluded; it has its own endpoint (PATCH /tasks/{id}/status).
    title: str | None = None
    description: str | None = None
    assigned_to: str | None = None
    track: str | None = None


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to a .env file in the project root.",
        )
    return api_key


def run_search(question: str, api_key: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Index assigned tasks and return top 3 matches for a question."""
    tasks = get_all_tasks()
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found in assigned.json.")

    embed_client = genai.Client(api_key=api_key)
    collection = ensure_indexed(embed_client, tasks)

    query_embedding = get_embedding(embed_client, question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches: list[dict[str, Any]] = []
    for i, metadata in enumerate(metadatas):
        distance = distances[i] if i < len(distances) else None
        matches.append(
            {
                "id": metadata.get("id"),
                "title": metadata.get("title"),
                "track": metadata.get("track"),
                "assigned_to": metadata.get("assigned_to"),
                "description": metadata.get("description"),
                "distance": distance,
            }
        )

    return matches


def push_tasks_to_backend(tasks: list[dict]) -> int:
    """POST each task to the Orchestra backend. Returns count of successes."""
    backend_url = os.getenv(
        "BACKEND_URL", "https://orchestra-backend-30fy.onrender.com"
    )
    succeeded = 0
    for task in tasks:
        try:
            response = requests.post(
                f"{backend_url}/tasks",
                json=task,
                timeout=30,
            )
            response.raise_for_status()
            succeeded += 1
        except Exception:
            continue
    return succeeded


def push_project_to_backend(
    name: str,
    description: str,
    tech_stack: list[str],
    members: list[str],
    project_id: str,
    summary: str = "",
    created_by: str = "",
) -> bool:
    """POST project details to the Orchestra backend. Returns True on success."""
    backend_url = os.getenv(
        "BACKEND_URL", "https://orchestra-backend-30fy.onrender.com"
    )
    try:
        response = requests.post(
            f"{backend_url}/projects",
            json={
                "name": name,
                "description": description,
                "tech_stack": tech_stack,
                "members": members,
                "id": project_id,
                # The backend stores the summary column as "blueprint_summary";
                # sending it under "summary" is silently dropped (verified live).
                "blueprint_summary": summary,
                # created_by is currently ignored by the backend (it doesn't read
                # it from the body yet) — sent so it populates the moment the
                # backend honours it; harmless until then.
                "created_by": created_by,
            },
            timeout=30,
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def validate_description(name: str, description: str, api_key: str) -> str | None:
    """Return an error reason if the description is not a meaningful software project."""
    prompt = f"""You are validating whether a project description represents a real, meaningful software project.

Project name:
\"\"\"{name}\"\"\"

Description:
\"\"\"{description}\"\"\"

Reject descriptions that are gibberish, random characters, or obvious test placeholders (e.g. "test", "asdf", "lorem ipsum", "meh", "haha"). Short but genuine descriptions like "a simple to do app" or "a chat app" are valid and must be accepted.

Return ONLY a single valid JSON object with this schema:
{{
  "valid": true | false,
  "reason": "string"
}}

Rules:
- Set "valid" to true if the description refers to any real app, tool, or software product — even if brief. Only reject if it is clearly nonsense or a placeholder.
- If "valid" is false, "reason" must briefly explain why (one short sentence).
- If "valid" is true, set "reason" to an empty string.
- Output JSON only."""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    payload = extract_json(raw)
    result = json.loads(payload)
    if result.get("valid"):
        return None
    return result.get("reason") or "Description is not a meaningful software project."


@app.post("/blueprint", dependencies=[Depends(verify_api_key)])
def create_blueprint(body: BlueprintRequest) -> dict[str, Any]:
    """Generate a task roadmap from a project name, description, and tech stack."""
    api_key = get_api_key()
    name = body.name.strip()
    description = body.description.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if not description:
        raise HTTPException(status_code=400, detail="description cannot be empty.")
    if len(description) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Description is too long. Maximum 2000 characters allowed.",
        )

    error = validate_description(name, description, api_key)
    if error:
        raise HTTPException(
            status_code=400, detail=f"Invalid project description: {error}"
        )

    tech_stack = [s.strip() for s in body.tech_stack if s and s.strip()]

    try:
        blueprint = generate_blueprint(
            name, description, tech_stack,
            project_id=body.project_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        known_skills = fetch_skills_from_neo4j()
        if body.members:
            # Scope assignment to this project's team. Without this, assign_tasks
            # sees every Developer in the graph (sample data, other projects) and
            # can hand a task to someone who isn't on this project. Use the
            # request's members as the roster; keep each one's known skills from
            # the graph, and default not-yet-onboarded members to no skills.
            skills = {member: known_skills.get(member, []) for member in body.members}
        else:
            # No members supplied — fall back to the full developer pool.
            skills = known_skills
        assigned = assign_tasks(blueprint, skills, api_key)
        # assign_tasks regenerates the JSON and drops top-level keys it wasn't
        # told about, so re-attach the blueprint's plain-English summary AND the
        # project_id here rather than hoping the second model preserves them.
        project_id = blueprint.get("project_id", "")
        summary = blueprint.get("summary", "")
        assigned["summary"] = summary
        assigned["project_id"] = project_id
        assigned["tech_stack"] = tech_stack
        # Stamp every task with the project id so the backend links each task row
        # to its project (task.project_id == project.id). assign_tasks drops the
        # project_id that blueprint.py set, so without this the tasks land
        # orphaned (project_id null) and never surface under the project.
        for task in assigned.get("tasks", []):
            task["project_id"] = project_id

        # Detect skill gaps and stamp them onto tasks before ingestion so
        # Neo4j stores gap_detected / missing_skill_or_role from the start.
        try:
            gap_report = analyze_skill_gaps({"tasks": assigned.get("tasks", []), "project_name": name}, skills, api_key)
            gap_by_id = {t["id"]: t for t in gap_report.get("tasks", [])}
            for task in assigned.get("tasks", []):
                gap_task = gap_by_id.get(task["id"], {})
                task["gap_detected"] = gap_task.get("gap_detected", False)
                task["missing_skill_or_role"] = gap_task.get("missing_skill_or_role")
            assigned["skill_gaps"] = [
                {"id": t["id"], "title": t["title"], "missing": t.get("missing_skill_or_role")}
                for t in assigned.get("tasks", [])
                if t.get("gap_detected")
            ]
        except Exception:
            pass

        ingest_all(assigned.get("tasks", []), skills)
        # Push the project FIRST so its row (with our pinned id) exists before the
        # tasks reference it — avoids the backend auto-creating a bare stub project.
        try:
            push_project_to_backend(
                name=name,
                description=description,
                tech_stack=tech_stack,
                members=body.members,
                project_id=project_id,
                summary=summary,
                created_by=body.created_by,
            )
        except Exception:
            pass
        try:
            push_tasks_to_backend(assigned.get("tasks", []))
        except Exception:
            pass
        invalidate_index()
        return assigned
    except Exception:
        return blueprint


@app.post("/assign", dependencies=[Depends(verify_api_key)])
def assign(body: AssignRequest) -> dict[str, Any]:
    """Assign tasks to team members based on skills."""
    if not body.tasks:
        raise HTTPException(status_code=400, detail="tasks cannot be empty.")
    if not body.skills:
        raise HTTPException(status_code=400, detail="skills cannot be empty.")

    api_key = get_api_key()
    blueprint = {"project_name": "Project", "tasks": body.tasks}

    try:
        return assign_tasks(blueprint, body.skills, api_key)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/search")
def search(
    question: str = Query(..., description="Natural language search question"),
    n_results: int = Query(3, description="Number of results to return"),
) -> dict[str, Any]:
    """Return the top 3 tasks matching a search question."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    api_key = get_api_key()

    try:
        matches = run_search(question.strip(), api_key, n_results)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"question": question.strip(), "matches": matches}


@app.post("/clover", dependencies=[Depends(verify_api_key)])
def clover(body: CloverRequest) -> StreamingResponse:
    """Stream a Clover answer chunk by chunk as Gemini generates it.

    Response is text/event-stream (SSE). Each event is a JSON line:
      data: {"chunk": "text"}                                     — answer fragment
      data: {"done": true, "conversation_history": [...]}         — final event
      data: {"error": "message", "status": 429}                   — on failure

    Arnav's proxy must forward chunks as they arrive (no buffering).
    Prince reads with response.body.getReader() and appends each chunk to the UI.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    api_key = get_api_key()

    def generate():
        try:
            for chunk_json in stream_answer(
                body.question.strip(),
                api_key,
                body.conversation_history,
                project_id=body.project_id,
                github_username=body.github_username,
                user_id=body.user_id,
                user_name=body.user_name,
                pending_action=body.pending_action,
            ):
                yield f"data: {chunk_json}\n\n"
        except Exception as exc:
            err = str(exc).lower()
            if "quota" in err or "429" in err or "resource exhausted" in err:
                msg = "Gemini API quota exceeded. Try again in a few minutes."
                code = 429
            elif "api key" in err or "401" in err or "403" in err or "invalid" in err:
                msg = "Gemini API key is invalid or expired. Contact the project admin."
                code = 503
            elif "timeout" in err or "deadline" in err or "504" in err:
                msg = "Clover took too long to respond. Try again."
                code = 504
            else:
                msg = f"Clover failed: {type(exc).__name__}: {exc}"
                code = 500
            # Emit the message as a chunk too, not just an error event. A client
            # that builds its answer only from `chunk` events (and ignores the
            # error event) would otherwise end up with empty text and render the
            # raw response envelope. Sending a chunk guarantees the user sees
            # readable text instead of blank JSON. Keep the error event for
            # clients that do handle it (status code, styling).
            yield f"data: {json.dumps({'chunk': msg})}\n\n"
            yield f"data: {json.dumps({'error': msg, 'status': code})}\n\n"
            yield f"data: {json.dumps({'done': True, 'conversation_history': (body.conversation_history or [])[-5:]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/standup")
def standup() -> dict[str, str]:
    """Generate a daily standup update from assigned.json."""
    api_key = get_api_key()

    try:
        tasks = get_all_tasks()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc

    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found in assigned.json.")

    grouped = group_tasks_by_person(tasks)
    try:
        standup_text = generate_standup(grouped, "Orchestra", api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Standup generation failed: {str(exc)}"
        ) from exc
    return {"standup": standup_text}


@app.get("/replan")
def replan() -> dict[str, Any]:
    """Generate re-planning suggestions for all blocked tasks."""
    api_key = get_api_key()

    try:
        tasks = get_all_tasks()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc

    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found in assigned.json.")

    blocked_tasks = find_blocked_tasks(tasks)
    if not blocked_tasks:
        return {"suggestions": [], "message": "No blocked tasks found"}

    project_name = "Orchestra"
    suggestions: list[dict[str, Any]] = []

    for blocked_task in blocked_tasks:
        blocked_id = str(blocked_task.get("id", ""))
        dependents = find_dependents(blocked_id, tasks)
        try:
            suggestions.append(
                suggest_replan(blocked_task, dependents, tasks, project_name, api_key)
            )
        except Exception as exc:
            suggestions.append({"error": str(exc), "blocked_task_id": blocked_id})

    return {"suggestions": suggestions}


@app.get("/commit-intel")
def commit_intel() -> dict[str, Any]:
    """Fetch live commit events from the Orchestra backend."""
    try:
        events = fetch_live_events()
        if not events:
            return {"total": 0, "events": [], "message": "No live events found."}
        return {"total": len(events), "events": events}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/onboarding", dependencies=[Depends(verify_api_key)])
def onboarding(body: OnboardingRequest) -> dict[str, Any]:
    """Generate a developer profile from GitHub and re-assign tasks."""
    username = body.github_username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="github_username cannot be empty.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise HTTPException(status_code=400, detail="Invalid GitHub username format")
    if len(username) > 39:
        raise HTTPException(status_code=400, detail="GitHub username too long")

    api_key = get_api_key()

    try:
        return build_profile(username, api_key)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"GitHub user '{username}' not found.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tasks")
def get_tasks() -> list[dict[str, Any]]:
    """Return every task from the Neo4j graph in CONTRACTS.md shape.

    Each task: id, title, track, description, status, assigned_to,
    dependencies (array of task ids), created_at, updated_at. This is the
    endpoint Member 3 (Arnav) consumes from his backend server.
    """
    try:
        return get_all_tasks()
    except RuntimeError as exc:  # missing NEO4J_* env vars
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # Neo4j unavailable / query failure
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


@app.patch("/tasks/{task_id}/status", dependencies=[Depends(verify_api_key)])
def update_task_status(task_id: str, body: TaskStatusRequest) -> dict[str, Any]:
    """Update a task's status in the Neo4j graph."""
    allowed_statuses = {"upcoming", "in_progress", "completed", "blocked"}
    if body.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(sorted(allowed_statuses))}",
        )

    try:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE") or None

        if not all([uri, username, password]):
            raise RuntimeError(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set. "
                "Add them to a .env file in the project root."
            )

        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                record = session.run(
                    """
                    MATCH (t:Task {id: $task_id})
                    SET t.status = $status
                    RETURN t.id AS id, t.status AS status
                    """,
                    task_id=task_id,
                    status=body.status,
                ).single()
        finally:
            driver.close()

        if record is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

        return {"id": record["id"], "status": record["status"]}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


def push_task_edit_to_backend(task_id: str, fields: dict[str, Any]) -> bool:
    """Best-effort: mirror a task field edit to the backend Postgres.

    The backend currently only exposes PATCH /tasks/{id}/status, so a general
    field PATCH will 404/405 and is swallowed here. This is forward-compatible:
    the moment the backend adds PATCH /tasks/{id}, edits sync automatically with
    no change on our side.
    """
    backend_url = os.getenv(
        "BACKEND_URL", "https://orchestra-backend-30fy.onrender.com"
    )
    try:
        response = requests.patch(
            f"{backend_url}/tasks/{task_id}", json=fields, timeout=30
        )
        return response.ok
    except Exception:
        return False


@app.patch("/tasks/{task_id}", dependencies=[Depends(verify_api_key)])
def edit_task(task_id: str, body: TaskEditRequest) -> dict[str, Any]:
    """Manually edit a single task's content in the Neo4j graph.

    Lets a PM fix what Gemini got wrong on one task — title, description, track,
    and/or assignee — without regenerating the whole roadmap. Reassigning also
    re-points the (Developer)-[:ASSIGNED_TO]->(Task) edge (not just the
    t.assigned_to property) so /graph, /tasks, and Clover all stay consistent.
    Status is not editable here — use PATCH /tasks/{id}/status.
    """
    # Collect only the fields the caller actually supplied.
    fields: dict[str, Any] = {}
    if body.title is not None:
        fields["title"] = body.title.strip()
    if body.description is not None:
        fields["description"] = body.description.strip()
    if body.track is not None:
        fields["track"] = body.track.strip()
    if body.assigned_to is not None:
        fields["assigned_to"] = body.assigned_to.strip()

    if not fields:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: title, description, track, assigned_to.",
        )
    if "assigned_to" in fields and not fields["assigned_to"]:
        raise HTTPException(status_code=400, detail="assigned_to cannot be blank.")

    fields["updated_at"] = datetime.utcnow().isoformat()

    try:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE") or None
        if not all([uri, username, password]):
            raise RuntimeError(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set. "
                "Add them to a .env file in the project root."
            )

        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                # 1. Update the scalar properties (and confirm the task exists).
                exists = session.run(
                    """
                    MATCH (t:Task {id: $task_id})
                    SET t += $fields
                    RETURN t.id AS id
                    """,
                    task_id=task_id,
                    fields=fields,
                ).single()
                if exists is None:
                    raise HTTPException(
                        status_code=404, detail=f"Task '{task_id}' not found."
                    )

                # 2. On reassignment, re-point the ASSIGNED_TO edge: drop any old
                #    Developer->Task edge and MERGE one to the new developer, so the
                #    graph relationship matches the t.assigned_to property we set.
                if "assigned_to" in fields:
                    session.run(
                        """
                        MATCH (t:Task {id: $task_id})
                        OPTIONAL MATCH (:Developer)-[r:ASSIGNED_TO]->(t)
                        DELETE r
                        WITH t
                        MERGE (d:Developer {name: $assignee})
                        MERGE (d)-[:ASSIGNED_TO]->(t)
                        """,
                        task_id=task_id,
                        assignee=fields["assigned_to"],
                    )

                # 3. Read the task back in the standard shape to return it.
                record = session.run(
                    """
                    MATCH (t:Task {id: $task_id})
                    OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
                    RETURN t.id AS id, t.title AS title, t.track AS track,
                           t.description AS description, t.status AS status,
                           t.assigned_to AS assigned_to,
                           [d IN collect(dep.id) WHERE d IS NOT NULL] AS dependencies,
                           t.created_at AS created_at, t.updated_at AS updated_at
                    """,
                    task_id=task_id,
                ).single()
        finally:
            driver.close()

        # Editing title/description changes what Clover + /search embed, so drop
        # the ChromaDB cache to force a re-index on the next semantic query.
        invalidate_index()

        # Best-effort mirror to the backend (excludes updated_at — the backend
        # stamps its own). No-op until the backend adds a task field PATCH.
        push_task_edit_to_backend(
            task_id, {k: v for k, v in fields.items() if k != "updated_at"}
        )

        return {
            "updated_fields": [k for k in fields if k != "updated_at"],
            "task": dict(record),
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


@app.get("/graph", dependencies=[Depends(verify_api_key)])
def graph(
    developers: bool = Query(
        True, description="Include Developer nodes and ASSIGNED_TO edges"
    ),
    skills: bool = Query(
        True, description="Include Skill nodes and HAS_SKILL edges"
    ),
) -> dict[str, Any]:
    """Return the Neo4j task graph as ReactFlow-ready nodes and edges.

    Shape: {"nodes": [...], "edges": [...]} — drop straight into
    <ReactFlow nodes edges />. Toggle `developers` / `skills` to narrow the
    view down to just the task dependency DAG.
    """
    try:
        return build_reactflow_graph(
            include_developers=developers, include_skills=skills
        )
    except RuntimeError as exc:  # missing NEO4J_* env vars
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # Neo4j unavailable / query failure
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


@app.delete("/projects/{project_id}", dependencies=[Depends(verify_api_key)])
def delete_project(project_id: str) -> dict[str, Any]:
    """Delete all Neo4j tasks (and their relationships) for a given project.

    Called by the backend after it deletes the project row from Postgres, so
    the graph doesn't accumulate orphaned tasks from deleted projects.
    """
    try:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE") or None

        if not all([uri, username, password]):
            raise RuntimeError(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set."
            )

        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                result = session.run(
                    """
                    MATCH (t:Task {project_id: $project_id})
                    DETACH DELETE t
                    RETURN count(t) AS deleted_count
                    """,
                    project_id=project_id,
                )
                record = result.single()
                deleted_count = record["deleted_count"] if record else 0
        finally:
            driver.close()

        global _chroma_indexed
        _chroma_indexed = False

        return {"project_id": project_id, "deleted_tasks": deleted_count}

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Graph database error: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn

    PORT = 8000
    print(f"Orchestra + Clover API running at http://localhost:{PORT}")
    print(f"  Task list (for Arnav):  http://localhost:{PORT}/tasks")
    print(f"  ReactFlow graph:        http://localhost:{PORT}/graph")
    print(f"  Interactive docs:       http://localhost:{PORT}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
