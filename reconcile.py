"""Reconcile the Neo4j task graph against the backend (source of truth).

Over time the graph accumulates tasks whose project was deleted in the backend
(orphans), tasks with no project_id (legacy junk), and tasks belonging to
archived projects — none of which the backend prunes from the graph. They
pollute cross-project reads (/capacity, Clover team-load, standup, /project).

This script uses the backend's /projects list as the source of truth for which
projects are live, and deletes graph Task nodes that no longer belong to one.

SAFE BY DEFAULT — dry run, reports only. Pass --apply to actually delete.

  python reconcile.py                      # dry run: show what WOULD be deleted
  python reconcile.py --apply              # delete orphans + null-project tasks
  python reconcile.py --apply --drop-archived   # also delete archived projects' tasks
  python reconcile.py --apply --keep-null       # preserve tasks with no project_id

Guardrails (any tripped -> abort WITHOUT deleting):
  - backend fetch fails, or returns zero projects (would nuke everything)
  - backend response looks paginated (total != returned count)
Archived projects still EXIST in the backend (is_archived=true) and could be
un-archived, so their tasks are KEPT unless --drop-archived is given.

NOTE: the deployed API caches a ChromaDB index in memory; after a delete, restart
/ redeploy the AI service (or hit an endpoint that calls invalidate_index) so the
vector index drops the removed tasks. This script only touches Neo4j.
"""

import argparse
import os
import sys

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

BACKEND_URL = os.getenv(
    "BACKEND_URL", "https://orchestra-backend-30fy.onrender.com"
)


def fetch_backend_projects() -> tuple[set[str], set[str]]:
    """Return (live_ids, archived_ids). Aborts the run on anything suspicious.

    Because the whole point is to DELETE by absence from this set, a bad fetch
    (empty, paginated, wrong shape) must never be treated as "these projects no
    longer exist" — so we exit hard instead of returning a partial set.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/projects", timeout=90)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        sys.exit(f"ABORT: could not fetch backend projects ({exc}).")

    projects = data.get("projects") if isinstance(data, dict) else data
    if not isinstance(projects, list) or not projects:
        sys.exit("ABORT: backend returned zero projects — refusing to delete everything.")

    # Pagination guard: if the backend only handed us a page, absence from it is
    # NOT proof a project was deleted.
    total = data.get("total") if isinstance(data, dict) else None
    if total is not None and total != len(projects):
        sys.exit(
            f"ABORT: backend returned {len(projects)} of {total} projects "
            "(paginated?) — cannot safely determine orphans."
        )

    live = {p.get("id") for p in projects if p.get("id")}
    archived = {p.get("id") for p in projects if p.get("id") and p.get("is_archived")}
    return live, archived


def graph_project_counts(session) -> dict:
    """Map each project_id present on Task nodes to its task count (None = no id)."""
    rows = session.run("MATCH (t:Task) RETURN t.project_id AS pid, count(*) AS n")
    return {r["pid"]: r["n"] for r in rows}


def delete_tasks_for_project(session, project_id) -> int:
    """DETACH DELETE every Task for one project_id (None = tasks with no id)."""
    if project_id is None:
        rec = session.run(
            "MATCH (t:Task) WHERE t.project_id IS NULL "
            "DETACH DELETE t RETURN count(t) AS n"
        ).single()
    else:
        rec = session.run(
            "MATCH (t:Task {project_id: $pid}) DETACH DELETE t RETURN count(t) AS n",
            pid=project_id,
        ).single()
    return rec["n"] if rec else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile Neo4j tasks against backend projects.")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--drop-archived", action="store_true", help="also delete archived projects' tasks")
    ap.add_argument("--keep-null", action="store_true", help="keep tasks that have no project_id")
    args = ap.parse_args()

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pw = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE") or None
    if not all([uri, user, pw]):
        sys.exit("NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set (.env).")

    live_ids, archived_ids = fetch_backend_projects()

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        driver.verify_connectivity()
        with driver.session(database=db) as session:
            counts = graph_project_counts(session)

            # Archived projects are still IN the backend list (live_ids), so
            # "orphan" = a project_id the backend doesn't know at all.
            orphan_pids = [p for p in counts if p is not None and p not in live_ids]
            archived_pids = [p for p in counts if p in archived_ids]
            null_count = counts.get(None, 0)

            def tot(pids):
                return sum(counts[p] for p in pids)

            print(f"Backend : {len(live_ids)} live projects ({len(archived_ids)} archived)")
            print(f"Graph   : {sum(counts.values())} tasks across {len(counts)} project buckets\n")
            print(f"Orphan (project deleted in backend): {len(orphan_pids)} projects, {tot(orphan_pids)} tasks  -> DELETE")
            print(f"Null project_id (legacy)           : {null_count} tasks  -> {'KEEP' if args.keep_null else 'DELETE'}")
            print(f"Archived projects (still exist)    : {len(archived_pids)} projects, {tot(archived_pids)} tasks  -> {'DELETE' if args.drop_archived else 'KEEP'}")

            to_delete: list = list(orphan_pids)
            if null_count and not args.keep_null:
                to_delete.append(None)
            if args.drop_archived:
                to_delete.extend(archived_pids)

            planned = sum(counts.get(p, 0) for p in to_delete)
            print(f"\nPlanned: delete {planned} tasks across {len(to_delete)} buckets.")

            if not args.apply:
                print("\nDRY RUN — nothing deleted. Re-run with --apply to execute.")
                return

            deleted = 0
            for pid in to_delete:
                n = delete_tasks_for_project(session, pid)
                deleted += n
                print(f"  deleted {n:>3}  [{'null project_id' if pid is None else pid}]")
            print(f"\nDone. Deleted {deleted} tasks. "
                  "Restart/redeploy the AI service so ChromaDB drops them too.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
