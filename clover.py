"""Clover — conversational project assistant powered by RAG + Gemini."""

import json
import os
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from search import (
    ASSIGNED_FILE,
    CHROMA_PATH,
    COLLECTION_NAME,
    get_embedding,
    index_tasks,
    load_assigned_tasks,
)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are Clover, an AI project assistant. Answer questions about the project "
    "using only the task context provided. Be specific and concise — mention actual "
    "names, task IDs, and titles in your answers."
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


def ask_clover(question: str, task_context: list[dict], api_key: str) -> str:
    """Send retrieved tasks to Gemini and return a conversational answer."""
    # TODO: add Neo4j context here — where graph data will plug in later

    client = genai.Client(api_key=api_key)
    context_json = json.dumps(task_context, indent=2, ensure_ascii=False)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            f"Task context:\n{context_json}\n\n"
            f"User question: {question}"
        ),
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
