"""One-time backfill: sync task statuses from the backend into the Neo4j graph.

The backend (Postgres) is the source of truth for task status; the AI repo's
Neo4j graph defaults every task to 'todo' on ingest, so the two drift apart.
After the backend's state-machine sync is live, run this ONCE to push each
task's real status into the graph via the AI repo's PATCH endpoint — so
/tasks, /graph, and Clover reflect reality.

Flow:
    GET  {BACKEND_URL}/tasks                  -> tasks + their Postgres statuses
    PATCH {AI_URL}/tasks/{id}/status          -> write each status into Neo4j

Config (env / .env):
    INTERNAL_API_KEY   required — the AI repo's x-api-key (never hardcode it)
    BACKEND_URL        optional — defaults to the current backend deployment
    AI_URL             optional — defaults to the current AI deployment

Usage:
    python backfill.py --dry-run     # preview only, writes nothing (do this first)
    python backfill.py               # apply
    python backfill.py --verify      # after applying, re-read the graph to confirm

Notes:
  * The PATCH endpoint only accepts: in_progress | completed | blocked. 'todo' is
    the graph default and is rejected by the endpoint, so todo tasks are skipped.
  * The script first checks that backend task IDs actually exist in the graph and
    refuses to mass-PATCH non-existent IDs unless you pass --force — see the
    ID-mismatch warning. (Backend IDs look like 'task_008'; the graph may use
    'T1'-style IDs. If they don't line up, fix the ID scheme upstream first.)
"""

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

DEFAULT_BACKEND_URL = "https://orchestra-backend-30fy.onrender.com"
DEFAULT_AI_URL = "https://orchestra-ai-36zm.onrender.com"
TIMEOUT = 90  # Render free tier cold start can take 30-50s
RETRIES = 3   # ride out cold-start 5xx / timeouts on free tier

# Statuses the AI repo's PATCH /tasks/{id}/status accepts (see main.update_task_status).
# 'todo' is the graph default and is rejected by the endpoint, so we never send it.
PATCHABLE = {"in_progress", "completed", "blocked"}


def fetch_tasks(base_url: str) -> list[dict]:
    """GET {base_url}/tasks and return the task list (handles list or {tasks:[]}).

    Retries on cold-start failures (5xx / timeouts) before giving up.
    """
    last_exc: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(f"{base_url}/tasks", timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("tasks", data.get("data", []))
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < RETRIES:
                print(f"  ...{base_url}/tasks failed (attempt {attempt}/{RETRIES}), retrying: {exc}")
                time.sleep(5)
    raise last_exc  # type: ignore[misc]


def patch_status(ai_url: str, task_id: str, status: str, headers: dict) -> tuple[bool, str]:
    """PATCH one task's status on the AI repo. Returns (ok, detail)."""
    try:
        response = requests.patch(
            f"{ai_url}/tasks/{task_id}/status",
            json={"status": status},
            headers=headers,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"request error: {exc}"
    if response.ok:
        return True, ""
    return False, f"HTTP {response.status_code}: {response.text[:140]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill task statuses into the Neo4j graph.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes, write nothing.")
    parser.add_argument("--verify", action="store_true", help="After applying, re-read the graph to confirm.")
    parser.add_argument("--force", action="store_true", help="PATCH even IDs not found in the graph (will 404).")
    args = parser.parse_args()

    load_dotenv()
    backend_url = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")
    ai_url = os.getenv("AI_URL", DEFAULT_AI_URL).rstrip("/")
    api_key = os.getenv("INTERNAL_API_KEY")

    if not api_key and not args.dry_run:
        sys.exit("INTERNAL_API_KEY is not set. Add it to .env (the AI repo's x-api-key).")

    headers = {"x-api-key": api_key or "", "Content-Type": "application/json"}

    print(f"Backend (source): {backend_url}/tasks")
    print(f"AI repo (target): {ai_url}/tasks/{{id}}/status")
    print(f"Mode: {'DRY-RUN (no writes)' if args.dry_run else 'APPLY'}\n")

    # 1. Read both sides.
    try:
        backend_tasks = fetch_tasks(backend_url)
    except requests.RequestException as exc:
        sys.exit(f"Failed to read backend /tasks: {exc}")
    try:
        graph_ids = {str(t.get("id")) for t in fetch_tasks(ai_url)}
    except requests.RequestException as exc:
        sys.exit(f"Failed to read AI /tasks (needed for the ID check): {exc}")

    print(f"Backend tasks: {len(backend_tasks)}  |  Graph tasks: {len(graph_ids)}")

    # 2. Pre-flight ID-overlap check — catches the task_008 vs T1 mismatch early.
    backend_ids = {str(t.get("id")) for t in backend_tasks}
    overlap = backend_ids & graph_ids
    print(f"IDs present in BOTH systems: {len(overlap)} / {len(backend_ids)}")
    if not overlap and not args.force:
        sys.exit(
            "\nNo backend task IDs exist in the graph — the two systems use "
            "different ID schemes (e.g. 'task_008' vs 'T1') or hold different "
            "task sets. Backfilling would 404 on every task. Align the IDs "
            "upstream (ingest the graph from the same source/IDs as the backend) "
            "before running this. Use --force to attempt anyway."
        )

    # 3. Backfill.
    updated = skipped_todo = skipped_missing = failed = 0
    touched: list[tuple[str, str]] = []

    for task in backend_tasks:
        task_id = str(task.get("id") or "").strip()
        status = str(task.get("status") or "").strip().lower()
        if not task_id:
            continue
        if status not in PATCHABLE:
            skipped_todo += 1  # todo / empty / unknown — nothing to push
            continue
        if task_id not in graph_ids and not args.force:
            print(f"  SKIP   {task_id} -> {status}  (id not in graph)")
            skipped_missing += 1
            continue

        if args.dry_run:
            print(f"  WOULD  {task_id} -> {status}")
            touched.append((task_id, status))
            continue

        ok, detail = patch_status(ai_url, task_id, status, headers)
        if ok:
            print(f"  OK     {task_id} -> {status}")
            updated += 1
            touched.append((task_id, status))
        else:
            print(f"  FAIL   {task_id} -> {status}  ({detail})")
            failed += 1

    print(
        f"\nSummary: {updated} updated, {skipped_todo} skipped (todo/empty), "
        f"{skipped_missing} skipped (not in graph), {failed} failed."
    )

    # 4. Optional verification — read the graph back and show the statuses we set.
    if args.verify and not args.dry_run and touched:
        print("\nVerifying against the graph...")
        live = {str(t.get("id")): t.get("status") for t in fetch_tasks(ai_url)}
        for task_id, expected in touched:
            actual = live.get(task_id)
            mark = "OK" if actual == expected else "MISMATCH"
            print(f"  {mark}  {task_id}: graph={actual!r} expected={expected!r}")
        print("\nFinal manual check: ask Clover about one of these tasks.")


if __name__ == "__main__":
    main()
