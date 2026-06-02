"""FastAPI server for Orchestra + Clover."""

import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

from assign import OUTPUT_FILE as ASSIGNED_FILE
from assign import assign_tasks, load_json_file
from blueprint import generate_blueprint
from graph_query import build_reactflow_graph
from search import (
    CHROMA_PATH,
    COLLECTION_NAME,
    get_embedding,
    index_tasks,
    load_assigned_tasks,
)

load_dotenv()

app = FastAPI(title="Orchestra + Clover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BlueprintRequest(BaseModel):
    idea: str


class AssignRequest(BaseModel):
    tasks: list[dict[str, Any]]
    skills: dict[str, list[str]]


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to a .env file in the project root.",
        )
    return api_key


def run_search(question: str, api_key: str) -> list[dict[str, Any]]:
    """Index assigned tasks and return top 3 matches for a question."""
    tasks = load_assigned_tasks(ASSIGNED_FILE)
    if not tasks:
        raise HTTPException(status_code=404, detail="No tasks found in assigned.json.")

    embed_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    index_tasks(collection, embed_client, tasks)

    query_embedding = get_embedding(embed_client, question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
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


@app.post("/blueprint")
def create_blueprint(body: BlueprintRequest) -> dict[str, Any]:
    """Generate a task roadmap from a raw app idea."""
    if not body.idea.strip():
        raise HTTPException(status_code=400, detail="idea cannot be empty.")

    try:
        return generate_blueprint(body.idea.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/assign")
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
) -> dict[str, Any]:
    """Return the top 3 tasks matching a search question."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    api_key = get_api_key()

    try:
        matches = run_search(question.strip(), api_key)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"question": question.strip(), "matches": matches}


@app.get("/tasks")
def get_tasks() -> dict[str, Any]:
    """Return all tasks from assigned.json."""
    try:
        assigned = load_json_file(ASSIGNED_FILE)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="assigned.json not found.") from exc

    return assigned


@app.get("/graph")
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

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
