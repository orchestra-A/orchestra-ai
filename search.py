"""Semantic task search over assigned roadmap tasks using ChromaDB."""

import json
import os

import chromadb
from dotenv import load_dotenv
from google import genai

ASSIGNED_FILE = "assigned.json"
COLLECTION_NAME = "orchestra"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_PATH = "chroma_db"


def load_assigned_tasks(path: str) -> list[dict]:
    """Read assigned tasks from JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("tasks", [])


def task_to_text(task: dict) -> str:
    """Convert a task record into searchable text."""
    return (
        f"Title: {task.get('title', '')}\n"
        f"Description: {task.get('description', '')}\n"
        f"Track: {task.get('track', '')}\n"
        f"Assigned to: {task.get('assigned_to', '')}"
    )


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """Create a vector embedding for one text input."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    # New SDK returns rich objects; support both object and dict-like shapes.
    if hasattr(response, "embeddings") and response.embeddings:
        first = response.embeddings[0]
        if hasattr(first, "values"):
            return list(first.values)
        if isinstance(first, dict) and "values" in first:
            return first["values"]

    if hasattr(response, "embedding") and response.embedding:
        if hasattr(response.embedding, "values"):
            return list(response.embedding.values)
        if isinstance(response.embedding, dict) and "values" in response.embedding:
            return response.embedding["values"]

    raise ValueError("Unexpected embedding response format from Gemini.")


def index_tasks(collection, embed_client: genai.Client, tasks: list[dict]) -> None:
    """Index tasks into Chroma."""
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for idx, task in enumerate(tasks):
        task_id = str(task.get("id") or f"TASK_{idx + 1}")
        text = task_to_text(task)
        ids.append(task_id)
        documents.append(text)
        embeddings.append(get_embedding(embed_client, text))
        metadatas.append(
            {
                "id": task_id,
                "title": str(task.get("title", "")),
                "track": str(task.get("track", "")),
                "assigned_to": str(task.get("assigned_to", "")),
                "description": str(task.get("description", "")),
                "priority": str(task.get("priority", "")),
                "status": str(task.get("status", "")),
                "dependencies": json.dumps(task.get("dependencies", [])),
                "project_id": str(task.get("project_id", "")),
            }
        )

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def print_results(query: str, results: dict) -> None:
    """Print top matching tasks clearly."""
    print(f"\nQuestion: {query}")
    print("\nTop 3 matching tasks:\n")

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not metadatas:
        print("No matches found.")
        return

    for i, metadata in enumerate(metadatas, start=1):
        distance = distances[i - 1] if i - 1 < len(distances) else None
        score_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "N/A"
        print(f"{i}. [{metadata.get('id', 'N/A')}] {metadata.get('title', '')}")
        print(f"   Track: {metadata.get('track', '')}")
        print(f"   Assigned to: {metadata.get('assigned_to', '')}")
        print(f"   Description: {metadata.get('description', '')}")
        print(f"   Distance: {score_text}\n")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    tasks = load_assigned_tasks(ASSIGNED_FILE)
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    embed_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Ignore if the collection does not exist yet.
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    index_tasks(collection, embed_client, tasks)

    question = input("Ask a task search question: ").strip()
    if not question:
        raise RuntimeError("Question cannot be empty.")

    query_embedding = get_embedding(embed_client, question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "distances"],
    )
    print_results(question, results)


if __name__ == "__main__":
    main()
