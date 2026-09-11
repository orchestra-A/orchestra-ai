"""Ingest the AI-generated roadmap into Neo4j as a connected task graph.

Reads the task list (assigned.json) and developer skills (skills.json) and
injects them into Neo4j as Task / Developer / Skill nodes with dependency,
assignment, and skill relationships. The script is idempotent — running it
repeatedly MERGEs the same nodes and relationships rather than duplicating.
"""

import json
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

from query import project_id_from_task_id

DEFAULT_STATUS = "upcoming"
# Story-point fallback when a task arrives without one (e.g. a legacy task, or a
# manual push). blueprint.normalize_points already snaps generated tasks to the
# Fibonacci scale; this only guards tasks that never went through it.
DEFAULT_POINTS = 3

# The canonical status vocabulary the whole AI service uses (Clover, the PATCH
# endpoint, standup, query all agree on these four). Anything else — legacy
# values like "todo"/"pending", or a stray push — is coerced to one of these so
# the graph never drifts out of sync with our own code again.
CANONICAL_STATUSES = {"upcoming", "in_progress", "completed", "blocked"}
_STATUS_ALIASES = {
    "todo": "upcoming", "to do": "upcoming", "to_do": "upcoming",
    "pending": "upcoming", "backlog": "upcoming",
    "not_started": "upcoming", "not started": "upcoming",
    "stopped": "blocked", "halted": "blocked",
    "done": "completed", "complete": "completed",
    "in progress": "in_progress", "inprogress": "in_progress", "active": "in_progress",
}


def normalize_status(raw) -> str:
    """Coerce any status value to the canonical AI vocabulary.

    upcoming / in_progress / completed / blocked. None, empty, or an
    unrecognised value all fall back to 'upcoming' (not started).
    """
    if raw is None:
        return DEFAULT_STATUS
    key = str(raw).strip().lower()
    if not key:
        return DEFAULT_STATUS
    if key in CANONICAL_STATUSES:
        return key
    return _STATUS_ALIASES.get(key, DEFAULT_STATUS)


def create_constraints(session) -> None:
    """Ensure uniqueness constraints so MERGE stays idempotent and fast."""
    session.run(
        "CREATE CONSTRAINT task_id IF NOT EXISTS "
        "FOR (t:Task) REQUIRE t.id IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT developer_name IF NOT EXISTS "
        "FOR (d:Developer) REQUIRE d.name IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT skill_name IF NOT EXISTS "
        "FOR (s:Skill) REQUIRE s.name IS UNIQUE"
    )


def ingest_tasks(session, tasks: list[dict]) -> None:
    """MERGE Task nodes with the properties defined in CONTRACTS.md §3.

    Node shape: node_type, id, title, track, assigned_to, status (CONTRACTS.md
    §3). We also carry description plus the created_at / updated_at timestamps so
    query.py can return the full task record, and the optional skill-gap fields
    (gap_detected, missing_skill_or_role) when the source file provides them.
    Setting a property to null removes it in Neo4j, so tasks without a gap (or
    without timestamps) simply won't have those properties.

    Every task's status is coerced to the canonical vocabulary first, so a
    legacy value from the source ("todo", "pending", ...) can never re-enter the
    graph and desync it from our code.
    """
    for task in tasks:
        task["status"] = normalize_status(task.get("status"))
        # Persist project_id so the graph itself knows each task's project
        # (Clover scoping, capacity planning, /tasks all rely on it). Prefer the
        # value the caller set, fall back to the id prefix.
        if not task.get("project_id"):
            task["project_id"] = project_id_from_task_id(task.get("id"))
    session.run(
        """
        UNWIND $tasks AS task
        MERGE (t:Task {id: task.id})
        SET t.node_type = 'Task',
            t.title = task.title,
            t.track = task.track,
            t.description = task.description,
            t.assigned_to = task.assigned_to,
            t.project_id = task.project_id,
            t.status = coalesce(task.status, $default_status),
            t.points = coalesce(task.points, $default_points),
            t.created_at = task.created_at,
            t.updated_at = task.updated_at,
            t.gap_detected = task.gap_detected,
            t.missing_skill_or_role = task.missing_skill_or_role
        """,
        tasks=tasks,
        default_status=DEFAULT_STATUS,
        default_points=DEFAULT_POINTS,
    )


def ingest_dependencies(session, tasks: list[dict]) -> None:
    """MERGE (Task)-[:DEPENDS_ON]->(Task) edges from each task's dependencies."""
    session.run(
        """
        UNWIND $tasks AS task
        UNWIND task.dependencies AS dep_id
        MATCH (t:Task {id: task.id})
        MATCH (dep:Task {id: dep_id})
        MERGE (t)-[:DEPENDS_ON]->(dep)
        """,
        tasks=tasks,
    )


def ingest_assignments(session, tasks: list[dict]) -> None:
    """MERGE Developer nodes and (Developer)-[:ASSIGNED_TO]->(Task) edges."""
    assigned = [t for t in tasks if t.get("assigned_to")]
    session.run(
        """
        UNWIND $tasks AS task
        MATCH (t:Task {id: task.id})
        MERGE (d:Developer {name: task.assigned_to})
        MERGE (d)-[:ASSIGNED_TO]->(t)
        """,
        tasks=assigned,
    )


def ingest_skills(session, skills: dict) -> None:
    """MERGE Developer / Skill nodes and (Developer)-[:HAS_SKILL]->(Skill) edges."""
    rows = [
        {"developer": developer, "skill": skill}
        for developer, skill_list in skills.items()
        for skill in skill_list
    ]
    session.run(
        """
        UNWIND $rows AS row
        MERGE (d:Developer {name: row.developer})
        MERGE (s:Skill {name: row.skill})
        MERGE (d)-[:HAS_SKILL]->(s)
        """,
        rows=rows,
    )


def print_summary(session) -> None:
    """Print node and relationship counts so the graph can be verified."""
    counts = session.run(
        """
        OPTIONAL MATCH (t:Task) WITH count(DISTINCT t) AS tasks
        OPTIONAL MATCH (d:Developer) WITH tasks, count(DISTINCT d) AS developers
        OPTIONAL MATCH (s:Skill) WITH tasks, developers, count(DISTINCT s) AS skills
        OPTIONAL MATCH (:Task)-[dep:DEPENDS_ON]->(:Task)
        WITH tasks, developers, skills, count(dep) AS depends_on
        OPTIONAL MATCH (:Developer)-[a:ASSIGNED_TO]->(:Task)
        WITH tasks, developers, skills, depends_on, count(a) AS assigned_to
        OPTIONAL MATCH (:Developer)-[h:HAS_SKILL]->(:Skill)
        WITH tasks, developers, skills, depends_on, assigned_to,
             count(h) AS has_skill
        OPTIONAL MATCH (g:Task) WHERE g.gap_detected = true
        RETURN tasks, developers, skills, depends_on, assigned_to,
               has_skill, count(g) AS skill_gaps
        """
    ).single()

    print("\nIngestion complete. Graph summary:")
    print(f"  Task nodes:           {counts['tasks']}")
    print(f"  Developer nodes:      {counts['developers']}")
    print(f"  Skill nodes:          {counts['skills']}")
    print(f"  DEPENDS_ON edges:     {counts['depends_on']}")
    print(f"  ASSIGNED_TO edges:    {counts['assigned_to']}")
    print(f"  HAS_SKILL edges:      {counts['has_skill']}")
    print(f"  Tasks w/ skill gap:   {counts['skill_gaps']}")


def ingest_all(tasks: list[dict], skills: dict) -> None:
    """Run the full Neo4j ingestion pipeline for tasks and skills."""
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    # Aura issues the instance ID as the default database name (not "neo4j"),
    # so honour NEO4J_DATABASE when set and fall back to the driver default.
    database = os.getenv("NEO4J_DATABASE") or None

    if not all([uri, username, password]):
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set. "
            "Add them to a .env file in the project root."
        )

    if not tasks:
        raise RuntimeError("No tasks provided.")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            create_constraints(session)
            ingest_tasks(session, tasks)
            ingest_dependencies(session, tasks)
            ingest_assignments(session, tasks)
            ingest_skills(session, skills)
            print_summary(session)
    finally:
        driver.close()


def main() -> None:
    payload = json.loads(sys.stdin.read())
    tasks = payload.get("tasks", [])
    skills = payload.get("skills", {})
    ingest_all(tasks, skills)


if __name__ == "__main__":
    main()
