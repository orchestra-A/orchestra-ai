"""FastAPI server for Orchestra + Clover."""

import json
import os
import re
from datetime import datetime
from typing import Any

import chromadb
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

from assign import assign_tasks, fetch_skills_from_neo4j
from blueprint import generate_blueprint
from ingest import ingest_all
from clover import ask_clover, search_top_tasks
from commit_intel import fetch_live_events, main as run_commit_intel
from graph_query import build_reactflow_graph, merge_developer_skills
from onboarding import build_profile
from query import get_all_tasks
from search import (
    COLLECTION_NAME,
    get_embedding,
    index_tasks,
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

_chroma_client = None
_chroma_collection = None
_chroma_indexed = False

app = FastAPI(title="Orchestra + Clover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def init_chroma():
    global _chroma_client, _chroma_collection
    _chroma_client = chromadb.EphemeralClient()
    _chroma_collection = _chroma_client.get_or_create_collection(name="orchestra")
    print("[STARTUP] ChromaDB EphemeralClient initialised")

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
    idea: str
    project_id: str = "P1"


class AssignRequest(BaseModel):
    tasks: list[dict[str, Any]]
    skills: dict[str, list[str]]


class CloverRequest(BaseModel):
    question: str
    conversation_history: list[dict] = []


class OnboardingRequest(BaseModel):
    github_username: str


class TaskStatusRequest(BaseModel):
    status: str


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
    global _chroma_indexed
    tasks = get_all_tasks()
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found in assigned.json.")

    embed_client = genai.Client(api_key=api_key)
    if not _chroma_indexed:
        index_tasks(_chroma_collection, embed_client, tasks)
        _chroma_indexed = True

    query_embedding = get_embedding(embed_client, question)
    results = _chroma_collection.query(
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


@app.post("/blueprint", dependencies=[Depends(verify_api_key)])
def create_blueprint(body: BlueprintRequest) -> dict[str, Any]:
    """Generate a task roadmap from a raw app idea."""
    if not body.idea.strip():
        raise HTTPException(status_code=400, detail="idea cannot be empty.")
    if len(body.idea.strip()) > 2000:
        raise HTTPException(
            status_code=400,
            detail="App idea is too long. Maximum 2000 characters allowed.",
        )

    try:
        blueprint = generate_blueprint(body.idea.strip(), project_id=body.project_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    api_key = get_api_key()
    try:
        skills = fetch_skills_from_neo4j()
        assigned = assign_tasks(blueprint, skills, api_key)
        # assign_tasks regenerates the JSON and drops top-level keys it wasn't
        # told about, so re-attach the blueprint's plain-English summary here
        # rather than hoping the second model preserves it.
        assigned["summary"] = blueprint.get("summary")
        ingest_all(assigned.get("tasks", []), skills)
        try:
            push_tasks_to_backend(assigned.get("tasks", []))
        except Exception:
            pass
        global _chroma_indexed
        _chroma_indexed = False
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
def clover(body: CloverRequest) -> dict[str, Any]:
    """Answer a project question using RAG task context and Clover."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    api_key = get_api_key()

    try:
        relevant_tasks = search_top_tasks(body.question.strip(), api_key)
        answer = ask_clover(
            body.question.strip(), relevant_tasks, api_key, body.conversation_history
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    updated_history = body.conversation_history + [
        {"question": body.question.strip(), "answer": answer}
    ]
    return {"answer": answer, "conversation_history": updated_history[-5:]}


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
    allowed_statuses = {"completed", "in_progress", "blocked"}
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


if __name__ == "__main__":
    import uvicorn

    PORT = 8000
    print(f"Orchestra + Clover API running at http://localhost:{PORT}")
    print(f"  Task list (for Arnav):  http://localhost:{PORT}/tasks")
    print(f"  ReactFlow graph:        http://localhost:{PORT}/graph")
    print(f"  Interactive docs:       http://localhost:{PORT}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
