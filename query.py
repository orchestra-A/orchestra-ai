"""Read-layer queries over the Orchestra task graph in Neo4j.

Where ingest.py writes the task graph, this script reads it back out and
answers the relationship questions a graph is good at: what is startable now,
who owns what, which tasks block the most work, and what a task is waiting on.

Run after ingest.py has populated the graph:
    .venv/bin/python query.py
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Status enum per CONTRACTS.md §3: todo | in_progress | completed | blocked.
DONE_STATUS = "completed"


def summary(session) -> None:
    """Print node and relationship counts for a quick sanity check."""
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
        RETURN tasks, developers, skills, depends_on, assigned_to,
               count(h) AS has_skill
        """
    ).single()

    print("=== Graph summary ===")
    print(f"  Tasks: {counts['tasks']}   Developers: {counts['developers']}   "
          f"Skills: {counts['skills']}")
    print(f"  DEPENDS_ON: {counts['depends_on']}   "
          f"ASSIGNED_TO: {counts['assigned_to']}   "
          f"HAS_SKILL: {counts['has_skill']}")


def ready_tasks(session) -> None:
    """Tasks that can start now — every dependency is done (or they have none)."""
    rows = session.run(
        """
        MATCH (t:Task)
        WHERE t.status <> $done
          AND NOT EXISTS {
              MATCH (t)-[:DEPENDS_ON]->(dep:Task)
              WHERE dep.status <> $done
          }
        OPTIONAL MATCH (d:Developer)-[:ASSIGNED_TO]->(t)
        RETURN t.id AS id, t.title AS title,
               coalesce(d.name, 'unassigned') AS owner
        ORDER BY t.id
        """,
        done=DONE_STATUS,
    )
    print("\n=== Ready to start now (no incomplete dependencies) ===")
    found = False
    for row in rows:
        found = True
        print(f"  [{row['id']}] {row['title']}  ->  {row['owner']}")
    if not found:
        print("  (none)")


def assignments_per_developer(session) -> None:
    """Each developer and the tasks assigned to them."""
    rows = session.run(
        """
        MATCH (d:Developer)
        OPTIONAL MATCH (d)-[:ASSIGNED_TO]->(t:Task)
        RETURN d.name AS developer,
               collect(t.id) AS task_ids,
               count(t) AS total
        ORDER BY developer
        """
    )
    print("\n=== Tasks per developer ===")
    for row in rows:
        ids = ", ".join(sorted(tid for tid in row["task_ids"] if tid))
        print(f"  {row['developer']} ({row['total']}): {ids or '—'}")


def developer_skills(session) -> None:
    """Each developer and the skills they hold."""
    rows = session.run(
        """
        MATCH (d:Developer)
        OPTIONAL MATCH (d)-[:HAS_SKILL]->(s:Skill)
        RETURN d.name AS developer, collect(s.name) AS skills
        ORDER BY developer
        """
    )
    print("\n=== Developer skills ===")
    for row in rows:
        skills = ", ".join(sorted(s for s in row["skills"] if s))
        print(f"  {row['developer']}: {skills or '—'}")


def most_blocking_tasks(session, limit: int = 5) -> None:
    """Tasks that the most other tasks depend on — the bottlenecks to watch."""
    rows = session.run(
        """
        MATCH (blocker:Task)<-[:DEPENDS_ON]-(dependent:Task)
        RETURN blocker.id AS id, blocker.title AS title,
               count(dependent) AS blocks
        ORDER BY blocks DESC, id
        LIMIT $limit
        """,
        limit=limit,
    )
    print(f"\n=== Top {limit} blocking tasks (most depended-on) ===")
    found = False
    for row in rows:
        found = True
        print(f"  [{row['id']}] {row['title']}  — blocks {row['blocks']} task(s)")
    if not found:
        print("  (no dependencies in graph)")


def skill_gaps(session) -> None:
    """Tasks flagged with a skill gap — assignee lacks a needed skill/role."""
    rows = session.run(
        """
        MATCH (t:Task)
        WHERE t.gap_detected = true
        RETURN t.id AS id, t.title AS title,
               coalesce(t.assigned_to, 'unassigned') AS owner,
               t.missing_skill_or_role AS missing
        ORDER BY t.id
        """
    )
    print("\n=== Skill gaps (assignee missing a needed skill/role) ===")
    found = False
    for row in rows:
        found = True
        print(f"  [{row['id']}] {row['title']}  ->  {row['owner']} "
              f"(needs: {row['missing']})")
    if not found:
        print("  (no skill gaps flagged)")


def _task_sort_key(task_id: str) -> tuple:
    """Sort 'T2' before 'T10' by ordering on the numeric suffix when present."""
    digits = "".join(ch for ch in task_id if ch.isdigit())
    return (0, int(digits)) if digits else (1, task_id)


def get_all_tasks() -> list[dict]:
    """Return every task as a clean JSON-serialisable record.

    Field names match CONTRACTS.md exactly: id, title, track, description,
    status, assigned_to, dependencies (array of task ids), created_at,
    updated_at. Opens and closes its own driver so callers (e.g. the FastAPI
    /tasks endpoint) can use it as a one-shot function.
    """
    load_dotenv()
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
            rows = session.run(
                """
                MATCH (t:Task)
                OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Task)
                WITH t, [d IN collect(dep.id) WHERE d IS NOT NULL] AS dependencies
                RETURN t.id AS id,
                       t.title AS title,
                       t.track AS track,
                       t.description AS description,
                       t.status AS status,
                       t.assigned_to AS assigned_to,
                       dependencies,
                       t.created_at AS created_at,
                       t.updated_at AS updated_at
                """
            )
            tasks = [dict(row) for row in rows]
    finally:
        driver.close()

    for task in tasks:
        task["dependencies"] = sorted(task["dependencies"], key=_task_sort_key)
    tasks.sort(key=lambda t: _task_sort_key(t["id"]))
    return tasks


def main() -> None:
    load_dotenv()
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
            summary(session)
            ready_tasks(session)
            assignments_per_developer(session)
            developer_skills(session)
            most_blocking_tasks(session)
            skill_gaps(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
