"""Link developer events to roadmap tasks and store enriched commits in ChromaDB."""

import hashlib
import json
import os
import re
import sys

import chromadb
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.0-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"
ASSIGNED_FILE = "assigned.json"
COLLECTION_NAME = "commits"
CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """You analyze developer activity events and link them to project tasks.

Event JSON:
{event}

Available tasks:
{tasks}

Return ONLY a single valid JSON object with this exact shape:
{{
  "linked_task_id": "T1",
  "linked_task_title": "string",
  "reason": "string explaining why this event maps to that task"
}}

Rules:
- "linked_task_id" must be one of the task ids from the available tasks list.
- "linked_task_title" must match the title of the chosen task.
- Use the event details (especially commit message text), member, action, and track context.
- Prefer tasks assigned to the same member when the match is otherwise close.
- Output JSON only. No markdown, no prose, no extra keys.
"""


def extract_json(text: str) -> str:
    """Extract JSON object from model output."""
    cleaned = text.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def load_json_file(path: str) -> dict:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def fetch_live_events() -> list:
    """Fetch live normalized events from the Orchestra backend."""
    try:
        response = requests.get(
            "https://orchestra-backend-2v5a.onrender.com/events",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("total", 0) == 0:
            return []
        return data.get("events", [])
    except Exception:
        return []


def validate_event(event: dict) -> None:
    """Ensure required normalized event fields are present."""
    required = ("platform", "actor", "event_type", "timestamp", "action_summary")
    missing = [field for field in required if field not in event]
    if missing:
        raise ValueError(f"Event is missing required fields: {', '.join(missing)}")


def link_event_to_task(event: dict, tasks: list[dict], client: genai.Client) -> dict:
    """Use Gemini to identify the most relevant task for an event."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            event=json.dumps(event, ensure_ascii=False),
            tasks=json.dumps(tasks, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    raw = response.text or ""
    payload = extract_json(raw)
    return json.loads(payload)


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """Create a vector embedding for one text input."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

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


def enriched_to_text(enriched: dict) -> str:
    """Convert enriched commit into searchable text for embeddings."""
    return (
        f"Platform: {enriched.get('platform', '')}\n"
        f"Actor: {enriched.get('actor', '')}\n"
        f"Event type: {enriched.get('event_type', '')}\n"
        f"Timestamp: {enriched.get('timestamp', '')}\n"
        f"Action summary: {enriched.get('action_summary', '')}\n"
        f"Linked task: [{enriched.get('linked_task_id', '')}] "
        f"{enriched.get('linked_task_title', '')}\n"
        f"Reason: {enriched.get('reason', '')}"
    )


def make_commit_id(event: dict) -> str:
    """Create a stable unique id for a commit event."""
    raw = (
        f"{event.get('platform', '')}|"
        f"{event.get('member', '')}|"
        f"{event.get('timestamp', '')}|"
        f"{event.get('details', '')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def store_in_chroma(collection, client: genai.Client, enriched: dict) -> None:
    """Persist enriched commit in ChromaDB commits collection."""
    commit_id = make_commit_id(enriched)
    text = enriched_to_text(enriched)
    embedding = get_embedding(client, text)

    collection.add(
        ids=[commit_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "platform": str(enriched.get("platform", "")),
                "actor": str(enriched.get("actor", "")),
                "event_type": str(enriched.get("event_type", "")),
                "timestamp": str(enriched.get("timestamp", "")),
                "action_summary": str(enriched.get("action_summary", "")),
                "linked_task_id": str(enriched.get("linked_task_id", "")),
                "linked_task_title": str(enriched.get("linked_task_title", "")),
                "reason": str(enriched.get("reason", "")),
            }
        ],
    )


def print_enriched(enriched: dict) -> None:
    """Print enriched commit clearly."""
    print("\nEnriched Commit Event")
    print("=" * 40)
    print(f"Platform:          {enriched.get('platform', '')}")
    print(f"Actor:             {enriched.get('actor', '')}")
    print(f"Event type:        {enriched.get('event_type', '')}")
    print(f"Timestamp:         {enriched.get('timestamp', '')}")
    print(f"Action summary:    {enriched.get('action_summary', '')}")
    print(f"Linked Task ID:    {enriched.get('linked_task_id', '')}")
    print(f"Linked Task Title: {enriched.get('linked_task_title', '')}")
    print(f"Reason:            {enriched.get('reason', '')}")
    print("=" * 40)
    print("\nFull JSON:")
    print(json.dumps(enriched, indent=2, ensure_ascii=False))


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    events = fetch_live_events()
    if not events:
        print("No live events found.")
        return

    assigned = load_json_file(ASSIGNED_FILE)
    tasks = assigned.get("tasks", [])
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    client = genai.Client(api_key=api_key)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    for event in events:
        validate_event(event)
        link = link_event_to_task(event, tasks, client)
        enriched = {**event, **link}
        store_in_chroma(collection, client, enriched)
        print_enriched(enriched)


if __name__ == "__main__":
    main()
