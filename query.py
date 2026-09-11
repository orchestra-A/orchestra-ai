"""Read-layer queries over the Orchestra task graph in Neo4j.

Where ingest.py writes the task graph, this script reads it back out and
answers the relationship questions a graph is good at: what is startable now,
who owns what, which tasks block the most work, and what a task is waiting on.

Run after ingest.py has populated the graph:
    .venv/bin/python query.py
"""

import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Status enum: upcoming | in_progress | completed | blocked.
DONE_STATUS = "completed"

# A task id is "{project_id}-T{n}" (e.g. "P4a584a19-T3"). Old sample ids like
# "T1" have no project prefix. This recovers the project_id from the id so the
# graph, the vector index, and Clover scoping always know a task's project even
# when the node itself is missing the property.
_PROJECT_ID_FROM_ID = re.compile(r"^(.*)-T\d+$")


def project_id_from_task_id(task_id) -> str | None:
    """Return the project_id embedded in a task id, or None if it has no prefix."""
    m = _PROJECT_ID_FROM_ID.match(str(task_id or ""))
    return m.group(1) if m else None


def next_task_id(project_id: str, existing_ids) -> str:
    """Mint the next '{project_id}-T{n}' id for a project.

    n is one past the highest existing -T suffix among that project's task ids,
    so ids stay stable and collision-free when a task is added to an existing
    project (the same scheme blueprint.py generates). Falls back to T1 when the
    project has no numbered tasks yet.
    """
    max_n = 0
    suffix = re.compile(rf"^{re.escape(project_id)}-T(\d+)$")
    for tid in existing_ids:
        m = suffix.match(str(tid or ""))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{project_id}-T{max_n + 1}"


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


def patch_task_status(task_id: str, new_status: str) -> dict | None:
    """Update a single task's status in Neo4j. Returns the updated task or None if not found."""
    from datetime import datetime, timezone

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE") or None

    if not all([uri, username, password]):
        return None

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            record = session.run(
                """
                MATCH (t:Task {id: $task_id})
                SET t.status = $status, t.updated_at = $updated_at
                RETURN t.id AS id, t.title AS title, t.status AS status
                """,
                task_id=task_id,
                status=new_status,
                updated_at=datetime.now(timezone.utc).isoformat(),
            ).single()
            return dict(record) if record else None
    except Exception:
        return None
    finally:
        driver.close()


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
                       t.project_id AS project_id,
                       t.points AS points,
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
        # Always expose project_id: prefer the stored property, fall back to the
        # id prefix. Everything downstream (the vector index, Clover project /
        # user scoping, capacity planning) filters on this, so it must never be
        # missing for a task that belongs to a project.
        if not task.get("project_id"):
            task["project_id"] = project_id_from_task_id(task["id"])
    tasks.sort(key=lambda t: _task_sort_key(t["id"]))
    return tasks


def reassignable_tasks(project_id: str | None = None) -> list[dict]:
    """Tasks that are SAFE to move to a newly-added member.

    A task is reassignable if it is unassigned or still `upcoming` (not started)
    — moving those costs nobody any work. Tasks that are in_progress, completed,
    or blocked are deliberately excluded: someone has already invested in them,
    so the rebalance never rips them away. This is the pool the smart "add
    member" flow rebalances over; pass project_id to scope to one project.

    Returns id, title, track, description, status, assigned_to, project_id and
    points for each task (enough to skill-rank and load-balance).
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
                WHERE ($project_id IS NULL OR t.project_id = $project_id)
                  AND (t.assigned_to IS NULL OR t.assigned_to = ''
                       OR t.status = 'upcoming')
                RETURN t.id AS id, t.title AS title, t.track AS track,
                       t.description AS description, t.status AS status,
                       t.assigned_to AS assigned_to, t.project_id AS project_id,
                       t.points AS points
                """,
                project_id=project_id,
            )
            tasks = [dict(row) for row in rows]
    finally:
        driver.close()

    return tasks


def capacity_by_developer(project_id: str | None = None) -> dict:
    """Story-point load and velocity per developer, summed over ASSIGNED_TO.

    This is what story points buy a graph: instead of counting tasks (which
    treats a 1-point tweak and a 13-point migration as equal), we weight each
    developer's load by effort. For each developer we return:
      - task_count        how many tasks they're assigned
      - total_points      sum of points across those tasks (their load)
      - completed_points  points already done  (their velocity)
      - remaining_points  points still open    (total - completed)
    Ordered by remaining_points DESC so the most-loaded person is first.

    Pass project_id to scope to a single project; omit for the whole graph.
    Also returns a project-wide totals block for a quick burn-up read.
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
            per_dev = [
                dict(r)
                for r in session.run(
                    """
                    MATCH (d:Developer)-[:ASSIGNED_TO]->(t:Task)
                    WHERE $project_id IS NULL OR t.project_id = $project_id
                    WITH d.name AS developer,
                         count(t) AS task_count,
                         sum(coalesce(t.points, 0)) AS total_points,
                         sum(CASE WHEN t.status = $done
                                  THEN coalesce(t.points, 0) ELSE 0 END)
                           AS completed_points
                    RETURN developer, task_count, total_points, completed_points,
                           total_points - completed_points AS remaining_points
                    ORDER BY remaining_points DESC, developer
                    """,
                    project_id=project_id,
                    done=DONE_STATUS,
                )
            ]
            totals = session.run(
                """
                MATCH (t:Task)
                WHERE $project_id IS NULL OR t.project_id = $project_id
                RETURN count(t) AS task_count,
                       sum(coalesce(t.points, 0)) AS total_points,
                       sum(CASE WHEN t.status = $done
                                THEN coalesce(t.points, 0) ELSE 0 END)
                         AS completed_points
                """,
                project_id=project_id,
                done=DONE_STATUS,
            ).single()
    finally:
        driver.close()

    total_pts = totals["total_points"] or 0
    done_pts = totals["completed_points"] or 0
    return {
        "project_id": project_id,
        "developers": per_dev,
        "totals": {
            "task_count": totals["task_count"] or 0,
            "total_points": total_pts,
            "completed_points": done_pts,
            "remaining_points": total_pts - done_pts,
            "percent_complete": round(100 * done_pts / total_pts, 1) if total_pts else 0.0,
        },
    }


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
