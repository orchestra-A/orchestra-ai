"""Answer natural-language dependency questions over assigned tasks."""

import json
import os
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
ASSIGNED_FILE = "assigned.json"

PROMPT_TEMPLATE = """You interpret dependency questions about a project task list.

User question:
"{question}"

Available tasks:
{tasks}

Return ONLY a single valid JSON object with this exact shape:
{{
  "target_task_ids": ["T1", "T2"],
  "interpretation": "short explanation of what the user is asking about"
}}

Rules:
- "target_task_ids" must contain one or more task ids from the available tasks list.
- Match by explicit task id (e.g. T16), track (e.g. frontend), title keywords, or scope (e.g. "blocking the frontend").
- If the question is broad (e.g. "what is blocking the frontend?"), include all relevant frontend tasks.
- If the question names a specific task id, include only that task.
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


def build_task_index(tasks: list[dict]) -> dict[str, dict]:
    """Map task id to task record."""
    return {str(task["id"]): task for task in tasks if task.get("id")}


def identify_target_tasks(
    question: str, tasks: list[dict], client: genai.Client
) -> dict:
    """Use Gemini to map a natural-language question to task ids."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            question=question,
            tasks=json.dumps(tasks, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    raw = response.text or ""
    payload = extract_json(raw)
    return json.loads(payload)


def collect_dependency_chain(
    task_id: str,
    task_index: dict[str, dict],
    visited: set[str] | None = None,
) -> list[str]:
    """Recursively collect all dependency ids for a task (deepest first)."""
    if visited is None:
        visited = set()

    if task_id in visited or task_id not in task_index:
        return []

    visited.add(task_id)
    task = task_index[task_id]
    chain: list[str] = []

    for dep_id in task.get("dependencies", []):
        dep_id = str(dep_id)
        chain.extend(collect_dependency_chain(dep_id, task_index, visited))
        if dep_id not in chain:
            chain.append(dep_id)

    return chain


def format_task_line(task: dict) -> str:
    """Format one task for display."""
    return (
        f"[{task.get('id', 'N/A')}] {task.get('title', '')} "
        f"({task.get('track', '')}) — assigned to {task.get('assigned_to', 'N/A')}"
    )


def print_dependency_tree(
    task_id: str,
    task_index: dict[str, dict],
    depth: int = 0,
    visited: set[str] | None = None,
) -> None:
    """Print dependency tree showing what blocks what."""
    if visited is None:
        visited = set()

    if task_id in visited or task_id not in task_index:
        return

    visited.add(task_id)
    task = task_index[task_id]
    indent = "  " * depth
    prefix = "└── blocked by: " if depth > 0 else ""
    print(f"{indent}{prefix}{format_task_line(task)}")

    dependencies = [str(dep) for dep in task.get("dependencies", [])]
    if not dependencies:
        if depth == 0:
            print(f"{indent}  (no blockers — can start immediately)")
        return

    for dep_id in dependencies:
        print_dependency_tree(dep_id, task_index, depth + 1, visited)


def print_results(
    question: str,
    interpretation: str,
    target_ids: list[str],
    task_index: dict[str, dict],
) -> None:
    """Print readable dependency chains for target tasks."""
    print(f"\nQuestion: {question}")
    print(f"Interpretation: {interpretation}\n")
    print("=" * 60)

    valid_targets = [tid for tid in target_ids if tid in task_index]
    if not valid_targets:
        print("No matching tasks found for this question.")
        return

    for i, task_id in enumerate(valid_targets, start=1):
        task = task_index[task_id]
        print(f"\n{i}. Dependency chain for {task_id}")
        print("-" * 60)

        chain = collect_dependency_chain(task_id, task_index)
        if chain:
            print("Full blocker chain (must complete in order):")
            for j, dep_id in enumerate(chain, start=1):
                dep = task_index[dep_id]
                print(f"  {j}. {format_task_line(dep)}")
            print(f"  {len(chain) + 1}. {format_task_line(task)}  ← target task")
        else:
            print(f"  {format_task_line(task)}  ← target task")
            print("  (no blockers — can start immediately)")

        print("\n  Tree view:")
        print_dependency_tree(task_id, task_index)
        print()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    assigned = load_json_file(ASSIGNED_FILE)
    tasks = assigned.get("tasks", [])
    if not tasks:
        raise RuntimeError("No tasks found in assigned.json.")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Ask a dependency question: ").strip()

    if not question:
        raise RuntimeError("Question cannot be empty.")

    task_index = build_task_index(tasks)
    client = genai.Client(api_key=api_key)

    result = identify_target_tasks(question, tasks, client)
    target_ids = [str(tid) for tid in result.get("target_task_ids", [])]
    interpretation = result.get("interpretation", "")

    print_results(question, interpretation, target_ids, task_index)


if __name__ == "__main__":
    main()
