# Neo4j stores the knowledge graph — tasks, developers, skills, dependencies, and all relationships between them
"""Graph read layer that returns the task graph as ReactFlow-ready data.

Where query.py prints to the terminal, this module *returns* plain dicts so the
FastAPI layer (main.py) can serve the Neo4j task graph straight to Prince's
ReactFlow canvas. The output is exactly what `<ReactFlow nodes edges />` expects:

    {
      "nodes": [{"id", "type", "position": {"x", "y"}, "data": {...}}, ...],
      "edges": [{"id", "source", "target", "type", "markerEnd", "data"}, ...]
    }

Node id scheme keeps the three node types collision-free in one flat list:
    Task       -> the task id              ("T1")
    Developer  -> "dev:<name>"             ("dev:Naman")
    Skill      -> "skill:<name>"           ("skill:Neo4j")

Depends only on neo4j + python-dotenv so it can be imported and tested without
pulling in the rest of the server stack.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """Lazily create and cache one Neo4j driver for the whole process."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([uri, username, password]):
            raise RuntimeError(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set. "
                "Add them to a .env file in the project root."
            )
        _driver = GraphDatabase.driver(uri, auth=(username, password))
        _driver.verify_connectivity()
    return _driver


def _database() -> Optional[str]:
    # Aura issues the instance ID as the default database name (not "neo4j").
    return os.getenv("NEO4J_DATABASE") or None


def merge_developer_skills(name: str, new_skills: list[str]) -> dict[str, Any]:
    """Add skills to a developer as Skill nodes + HAS_SKILL edges.

    Creates the Developer node if it doesn't exist. MERGE makes this additive
    and duplicate-free, so existing skills are never removed — this is the
    manual fallback for `onboarding.py`'s GitHub-inferred skills. Returns the
    developer's full current skill set after the merge.
    """
    driver = get_driver()
    with driver.session(database=_database()) as session:
        record = session.run(
            """
            MERGE (d:Developer {name: $name})
            WITH d
            UNWIND $new_skills AS skill_name
              MERGE (s:Skill {name: skill_name})
              MERGE (d)-[:HAS_SKILL]->(s)
            WITH DISTINCT d
            MATCH (d)-[:HAS_SKILL]->(s:Skill)
            RETURN d.name AS name, collect(s.name) AS skills
            """,
            name=name,
            new_skills=new_skills,
        ).single()

    return {"name": record["name"], "skills": sorted(record["skills"])}


def _task_node(task: dict) -> dict[str, Any]:
    return {
        "id": task["id"],
        "type": "task",
        "data": {
            "label": task["title"],
            "track": task.get("track"),
            "status": task.get("status"),
            "assigned_to": task.get("assigned_to"),
            "gap_detected": bool(task.get("gap_detected")),
            "missing_skill_or_role": task.get("missing_skill_or_role"),
        },
    }


def _developer_node(name: str) -> dict[str, Any]:
    return {
        "id": f"dev:{name}",
        "type": "developer",
        "data": {"label": name},
    }


def _skill_node(name: str) -> dict[str, Any]:
    return {
        "id": f"skill:{name}",
        "type": "skill",
        "data": {"label": name},
    }


def _edge(relationship: str, source: str, target: str) -> dict[str, Any]:
    return {
        "id": f"{relationship}-{source}-{target}",
        "source": source,
        "target": target,
        "type": "smoothstep",
        "markerEnd": {"type": "arrowclosed"},
        "data": {"relationship": relationship},
    }


def build_reactflow_graph(
    include_developers: bool = True,
    include_skills: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Read the graph from Neo4j and return ReactFlow nodes + edges.

    Tasks and their DEPENDS_ON edges are always included. Developer nodes /
    ASSIGNED_TO edges and Skill nodes / HAS_SKILL edges are toggled by the
    flags so the caller can request just the task DAG or the full graph.
    """
    driver = get_driver()
    with driver.session(database=_database()) as session:
        tasks = [
            dict(r)
            for r in session.run(
                """
                MATCH (t:Task)
                RETURN t.id AS id, t.title AS title, t.track AS track,
                       t.status AS status, t.assigned_to AS assigned_to,
                       t.gap_detected AS gap_detected,
                       t.missing_skill_or_role AS missing_skill_or_role
                ORDER BY t.id
                """
            )
        ]
        dep_edges = [
            dict(r)
            for r in session.run(
                """
                MATCH (t:Task)-[:DEPENDS_ON]->(dep:Task)
                RETURN t.id AS source, dep.id AS target
                """
            )
        ]
        developers = (
            [r["name"] for r in session.run(
                "MATCH (d:Developer) RETURN d.name AS name ORDER BY d.name")]
            if include_developers else []
        )
        skills = (
            [r["name"] for r in session.run(
                "MATCH (s:Skill) RETURN s.name AS name ORDER BY s.name")]
            if include_skills else []
        )
        assigned = (
            [dict(r) for r in session.run(
                """
                MATCH (d:Developer)-[:ASSIGNED_TO]->(t:Task)
                RETURN d.name AS developer, t.id AS task
                """)]
            if include_developers else []
        )
        has_skill = (
            [dict(r) for r in session.run(
                """
                MATCH (d:Developer)-[:HAS_SKILL]->(s:Skill)
                RETURN d.name AS developer, s.name AS skill
                """)]
            if include_developers and include_skills else []
        )

    # ---- nodes ----------------------------------------------------------
    nodes: list[dict[str, Any]] = []
    for task in tasks:
        nodes.append(_task_node(task))
    for name in developers:
        nodes.append(_developer_node(name))
    for name in skills:
        nodes.append(_skill_node(name))

    # ---- edges ----------------------------------------------------------
    edges: list[dict[str, Any]] = [
        _edge("DEPENDS_ON", e["source"], e["target"]) for e in dep_edges
    ]
    for row in assigned:
        edges.append(_edge("ASSIGNED_TO", f"dev:{row['developer']}", row["task"]))
    for row in has_skill:
        edges.append(
            _edge("HAS_SKILL", f"dev:{row['developer']}", f"skill:{row['skill']}")
        )

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    import json

    graph = build_reactflow_graph()
    print(json.dumps(graph, indent=2))
