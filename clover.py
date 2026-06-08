"""Clover — conversational project assistant powered by RAG + Gemini."""

import json
import os
import sys

import chromadb
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from commit_intel import fetch_live_events
from search import (
    ASSIGNED_FILE,
    CHROMA_PATH,
    COLLECTION_NAME,
    get_embedding,
    index_tasks,
    load_assigned_tasks,
)

MODEL_NAME = "gemini-2.5-flash-lite"
GRAPH_API_URL = "https://orchestra-ai-production.up.railway.app/graph"

SYSTEM_PROMPT = (
    "You are Clover, an AI project assistant. Answer questions about the project "
    "using only the task context, graph relationship data, and recent team activity provided. "
    "Graph data includes nodes (tasks, developers, skills) and edges "
    "(DEPENDS_ON, ASSIGNED_TO, HAS_SKILL) from the project knowledge graph. "
    "You also have access to recent team activity from Discord and GitHub events — "
    "use it to answer questions like \"what did X work on recently?\". "
    "Be specific and concise — mention actual names, task IDs, and titles in your answers."
)


def search_top_tasks(question: str, api_key: str) -> list[dict]:
    """Find the 3 most relevant tasks using search.py helpers and ChromaDB."""
    tasks = load_assigned_tasks(ASSIGNED_FILE)
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

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

    matches: list[dict] = []
    for i, metadata in enumerate(metadatas):
        distance = distances[i] if i < len(distances) else None
        matches.append({**metadata, "distance": distance})

    return matches


def fetch_graph() -> dict | None:
    """Fetch the project graph from the Orchestra API. Returns None on failure."""
    try:
        response = requests.get(GRAPH_API_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def is_relevant_node(node: dict, question: str) -> bool:
    """Return True if a graph node matches the question by title or assignee."""
    q = question.lower()
    data = node.get("data", {})
    label = str(data.get("label", "")).lower()
    assigned_to = str(data.get("assigned_to", "")).lower()

    if assigned_to and assigned_to in q:
        return True
    if label and label in q:
        return True
    for word in q.split():
        if len(word) > 2 and word in label:
            return True
    return False


def get_relevant_graph_context(question: str) -> dict | None:
    """Fetch and filter graph nodes/edges relevant to the question."""
    graph = fetch_graph()
    if not graph:
        return None

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    relevant_nodes = [node for node in nodes if is_relevant_node(node, question)]

    if not relevant_nodes:
        return {"nodes": [], "edges": []}

    relevant_ids = {node["id"] for node in relevant_nodes}
    relevant_edges = [
        edge
        for edge in edges
        if edge.get("source") in relevant_ids or edge.get("target") in relevant_ids
    ]

    return {"nodes": relevant_nodes, "edges": relevant_edges}


def ask_clover(question: str, task_context: list[dict], api_key: str) -> str:
    """Send retrieved tasks and graph context to Gemini and return an answer."""
    client = genai.Client(api_key=api_key)
    context_json = json.dumps(task_context, indent=2, ensure_ascii=False)

    prompt_parts = [f"Task context:\n{context_json}"]

    try:
        live_events = fetch_live_events()
        if live_events:
            commit_json = json.dumps(live_events, indent=2, ensure_ascii=False)
            prompt_parts.append(f"Recent activity context:\n{commit_json}")
    except Exception:
        pass

    graph_context = get_relevant_graph_context(question)
    if graph_context is not None:
        graph_json = json.dumps(graph_context, indent=2, ensure_ascii=False)
        prompt_parts.append(f"Graph context:\n{graph_json}")

    prompt_parts.append(f"User question: {question}")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="\n\n".join(prompt_parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )

    return (response.text or "").strip()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Ask Clover a project question: ").strip()

    if not question:
        raise RuntimeError("Question cannot be empty.")

    relevant_tasks = search_top_tasks(question, api_key)
    answer = ask_clover(question, relevant_tasks, api_key)

    print(f"\nQuestion: {question}\n")
    print("Clover:")
    print(answer)


if __name__ == "__main__":
    main()
