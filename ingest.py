"""Ingest the AI-generated roadmap into Neo4j as a connected task graph.

Reads the task list (assigned.json) and developer skills (skills.json) and
injects them into Neo4j as Task / Developer / Skill nodes with dependency,
assignment, and skill relationships. The script is idempotent — running it
repeatedly MERGEs the same nodes and relationships rather than duplicating.
"""

import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

ASSIGNED_FILE = "assigned.json"
SKILL_GAP_FILE = "skill_gap_report.json"
SKILLS_FILE = "skills.json"
DEFAULT_STATUS = "todo"


def resolve_task_source() -> str:
    """Pick the richest available task file.

    skill_gap_report.json is a superset of assigned.json — same task fields
    plus gap_detected / missing_skill_or_role — so prefer it when present and
    fall back to assigned.json otherwise.
    """
    return SKILL_GAP_FILE if os.path.exists(SKILL_GAP_FILE) else ASSIGNED_FILE


def load_tasks(path: str) -> list[dict]:
    """Read the roadmap tasks from JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("tasks", [])


def load_skills(path: str) -> dict:
    """Read the developer -> skills mapping from JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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

    Node shape: node_type, id, title, track, assigned_to, status. We also carry
    the optional skill-gap fields (gap_detected, missing_skill_or_role) when the
    source file provides them. Setting a property to null removes it in Neo4j,
    so tasks without a gap simply won't have those properties.
    """
    session.run(
        """
        UNWIND $tasks AS task
        MERGE (t:Task {id: task.id})
        SET t.node_type = 'Task',
            t.title = task.title,
            t.track = task.track,
            t.description = task.description,
            t.assigned_to = task.assigned_to,
            t.status = coalesce(task.status, $default_status),
            t.gap_detected = task.gap_detected,
            t.missing_skill_or_role = task.missing_skill_or_role
        """,
        tasks=tasks,
        default_status=DEFAULT_STATUS,
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


def main() -> None:
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

    source = resolve_task_source()
    tasks = load_tasks(source)
    if not tasks:
        raise RuntimeError(f"No tasks found in {source}.")
    print(f"Loading tasks from {source}")
    skills = load_skills(SKILLS_FILE)

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


if __name__ == "__main__":
    main()
