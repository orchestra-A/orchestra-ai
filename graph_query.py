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

# Horizontal/vertical spacing for the starter layout (pixels). ReactFlow needs
# a position on every node; we lay tasks out left-to-right by dependency depth
# and put developers / skills in their own columns. Prince can re-run a client
# side layout (dagre/elk) on top of this — the positions are just a sane start.
X_GAP = 280
Y_GAP = 130

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


def _task_levels(task_ids: list[str], dep_edges: list[dict]) -> dict[str, int]:
    """Assign each task a depth = longest dependency chain beneath it.

    A task with no dependencies is level 0; otherwise its level is
    1 + the deepest dependency. Cycles are guarded against so a malformed
    graph degrades to level 0 instead of recursing forever.
    """
    deps: dict[str, list[str]] = {tid: [] for tid in task_ids}
    for edge in dep_edges:
        if edge["source"] in deps:
            deps[edge["source"]].append(edge["target"])

    level: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(tid: str) -> int:
        if tid in level:
            return level[tid]
        if tid in visiting:  # cycle guard
            return 0
        visiting.add(tid)
        deepest = 0
        for dep in deps.get(tid, []):
            deepest = max(deepest, depth(dep) + 1)
        visiting.discard(tid)
        level[tid] = deepest
        return deepest

    for tid in task_ids:
        depth(tid)
    return level


def _task_node(task: dict, x: float, y: float) -> dict[str, Any]:
    return {
        "id": task["id"],
        "type": "task",
        "position": {"x": float(x), "y": float(y)},
        "data": {
            "label": task["title"],
            "track": task.get("track"),
            "status": task.get("status"),
            "assigned_to": task.get("assigned_to"),
            "gap_detected": bool(task.get("gap_detected")),
            "missing_skill_or_role": task.get("missing_skill_or_role"),
        },
    }


def _developer_node(name: str, x: float, y: float) -> dict[str, Any]:
    return {
        "id": f"dev:{name}",
        "type": "developer",
        "position": {"x": float(x), "y": float(y)},
        "data": {"label": name},
    }


def _skill_node(name: str, x: float, y: float) -> dict[str, Any]:
    return {
        "id": f"skill:{name}",
        "type": "skill",
        "position": {"x": float(x), "y": float(y)},
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
    levels = _task_levels([t["id"] for t in tasks], dep_edges)
    by_level: dict[int, list[dict]] = {}
    for task in tasks:
        by_level.setdefault(levels[task["id"]], []).append(task)

    nodes: list[dict[str, Any]] = []
    for level, group in sorted(by_level.items()):
        for row, task in enumerate(group):
            nodes.append(_task_node(task, x=level * X_GAP, y=row * Y_GAP))

    max_level = max(levels.values(), default=0)
    for row, name in enumerate(developers):  # developers in a left-hand column
        nodes.append(_developer_node(name, x=-1.6 * X_GAP, y=row * Y_GAP))
    skill_x = (max_level + 2) * X_GAP
    for row, name in enumerate(skills):  # skills in a right-hand column
        nodes.append(_skill_node(name, x=skill_x, y=row * Y_GAP))

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
