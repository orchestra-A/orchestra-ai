"""Relationship/traversal queries over the Neo4j task graph.

These are the callable functions Clover (clover.py) plugs in to answer specific
relational questions directly from Neo4j — instead of string-matching over the
ReactFlow /graph blob. Each function *returns* plain JSON-serialisable dicts (it
does not print), so the caller can json.dumps them straight into a Gemini prompt
or a FastAPI response.

Three questions, three functions:
    "What tasks is <person> working on?"      -> tasks_for_person(name)
    "What tasks are blocked?"                  -> blocked_tasks()
    "What are the dependencies of task T1?"    -> dependencies_of("T1")

Task dicts match CONTRACTS.md §3 field names (id, title, track, description,
status, assigned_to, dependencies) so they line up with query.get_all_tasks().
Graph schema (see ingest.py):
    (Developer)-[:ASSIGNED_TO]->(Task)
    (Task)-[:DEPENDS_ON]->(Task)
    status in: todo | in_progress | completed | blocked
"""

import json
import sys

from dotenv import load_dotenv

from graph_query import _database, get_driver
from query import _task_sort_key

# Shared RETURN projection so every function hands back the same task shape.
# `t` must be bound to a Task node; collects its direct dependency ids too.
_TASK_PROJECTION = """
    OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
    WITH t, [d IN collect(dep.id) WHERE d IS NOT NULL] AS dependencies
    RETURN t.id AS id,
           t.title AS title,
           t.track AS track,
           t.description AS description,
           t.status AS status,
           t.assigned_to AS assigned_to,
           dependencies
"""


def _run(cypher: str, **params) -> list[dict]:
    """Run a read query and return rows as sorted, clean task dicts."""
    driver = get_driver()
    with driver.session(database=_database()) as session:
        rows = [dict(row) for row in session.run(cypher, **params)]
    for task in rows:
        if "dependencies" in task and task["dependencies"]:
            task["dependencies"] = sorted(task["dependencies"], key=_task_sort_key)
    rows.sort(key=lambda t: _task_sort_key(t.get("id", "")))
    return rows


def tasks_for_person(name: str) -> list[dict]:
    """Tasks assigned to a developer. Answers "What is <person> working on?".

    Case-insensitive on the developer name so "naman"/"Naman" both match.
    Returns [] if the person has no tasks (or does not exist).
    """
    cypher = f"""
        MATCH (d:Developer)-[:ASSIGNED_TO]->(t:Task)
        WHERE toLower(d.name) = toLower($name)
        {_TASK_PROJECTION}
    """
    return _run(cypher, name=name)


def blocked_tasks() -> list[dict]:
    """Every task whose status is 'blocked'. Answers "What tasks are blocked?"."""
    cypher = f"""
        MATCH (t:Task)
        WHERE t.status = 'blocked'
        {_TASK_PROJECTION}
    """
    return _run(cypher)


def dependencies_of(task_id: str, recursive: bool = False) -> list[dict]:
    """Tasks that <task_id> depends on. Answers "Dependencies of task T1?".

    recursive=False -> direct dependencies only (one DEPENDS_ON hop).
    recursive=True  -> the full transitive chain (every upstream task), useful
                       for "what must finish before T1 can start?".
    Returns [] if the task has no dependencies or does not exist.
    """
    hop = "*1.." if recursive else ""
    cypher = f"""
        MATCH (src:Task {{id: $task_id}})-[:DEPENDS_ON{hop}]->(t:Task)
        {_TASK_PROJECTION}
    """
    return _run(cypher, task_id=task_id)


def main() -> None:
    """Quick manual check: python relationship_queries.py [person] [task_id]."""
    load_dotenv()
    person = sys.argv[1] if len(sys.argv) > 1 else "Naman"
    task_id = sys.argv[2] if len(sys.argv) > 2 else "T1"

    print(f"\n# Tasks {person} is working on:")
    print(json.dumps(tasks_for_person(person), indent=2, ensure_ascii=False))

    print("\n# Blocked tasks:")
    print(json.dumps(blocked_tasks(), indent=2, ensure_ascii=False))

    print(f"\n# Direct dependencies of {task_id}:")
    print(json.dumps(dependencies_of(task_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
