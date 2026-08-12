"""Clover — conversational project assistant powered by RAG + Gemini."""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from commit_intel import fetch_live_events, link_event_to_task
from graph_query import build_reactflow_graph
from query import get_all_tasks, patch_task_status
from search import ensure_indexed, get_embedding

MODEL_NAME = "gemini-2.5-flash"

# Sentinel so ask_clover can tell "caller pre-fetched this (possibly None)" apart
# from "caller didn't pass it, fetch it yourself". fetch_graph()/fetch_live_events()
# can both legitimately return None, so a plain None default wouldn't distinguish.
_UNSET = object()

SYSTEM_PROMPT = """You are Clover, an AI project assistant for a software development team. You have access to three sources of context:
1. Task context — structured task data with IDs, titles, assignees, tracks, and statuses
2. Graph context — knowledge graph showing relationships between people, tasks, and skills
3. Recent activity context — live Discord and GitHub events showing what the team has been doing

Use these rules to answer questions:
- "What did X work on recently?" or "What has X been doing?" → prioritise recent activity context
- "Who is working on X?" or "Who owns X?" → prioritise task context and graph context
- "What tasks are blocked?" or "What is blocked?" → prioritise task context, look for blocked status or unmet dependencies
- "What skills does X have?" or "Who can do X?" → prioritise graph context and task context
- For all other questions → use whichever context is most relevant

Always be specific — mention actual names, task titles, and timestamps in your answers. When referencing a task, cite its title and project name together, e.g. "Implement How to Use UI (calculator app)". Never include raw task IDs in your response — they appear separately as clickable cards. If the context does not contain enough information to answer, say so clearly.

When someone asks about their own tasks ("what am I working on?", "what are my tasks?"), highlight the 3-4 most important or in-progress ones and summarise the rest in one sentence (e.g. "and 5 more upcoming tasks"). Never dump the full list.

Formatting rules — strictly follow these:
- Write in plain conversational text only. No markdown, no asterisks, no bold, no bullet symbols, no headers.
- Use plain dashes (-) for lists if needed, nothing else.
- Keep responses concise — 3 to 6 sentences unless the question genuinely needs more detail."""


# Fetches the full project list from the backend (best-effort, retried).
def fetch_projects() -> list[dict]:
    """Return the raw projects list from the backend, or [] on repeated failure.

    Retries once: the backend runs on a free tier that cold-starts, so the first
    request after idle can time out or come back empty. A second attempt usually
    hits a now-warm backend. This matters for navigation / project-switch, which
    can't resolve a project by name without this list — a cold miss here is what
    made "take me to <project>" return an empty, action-less response.
    """
    backend_url = os.getenv("BACKEND_URL", "https://orchestra-backend-30fy.onrender.com")
    for attempt in range(2):
        try:
            resp = requests.get(f"{backend_url}/projects", timeout=30)
            resp.raise_for_status()
            projects = resp.json().get("projects", [])
            if projects:
                return projects
        except Exception:
            pass
        if attempt == 0:
            time.sleep(1)  # let a cold backend finish warming before the retry
    return []


# Fetches a {project_id: project_name} map from the backend (best-effort).
def fetch_project_names() -> dict[str, str]:
    """Return {project_id: name} from the backend, or {} on any failure.

    The graph only stores the project id (embedded in each task id, e.g.
    "P4a584a19-T3"); the human-readable name lives in the backend's projects
    table. We pull it so Clover can cite "PantryPal" instead of a bare id.
    """
    return {p["id"]: p.get("name", "") for p in fetch_projects() if p.get("id")}


# Resolves a task id to its project name using the id prefix (e.g. "P4a..-T3").
def project_name_for_task(task_id: str, name_map: dict[str, str]) -> str:
    """Return the project name whose id prefixes this task id, else ""."""
    if not task_id or not name_map:
        return ""
    for pid, pname in name_map.items():
        if pid and (task_id == pid or task_id.startswith(f"{pid}-")):
            return pname
    return ""


# Resolves a user_id to their GitHub username via the backend.
def fetch_github_username(user_id: str) -> str | None:
    """Call GET /users/{user_id} and return github_username, or None on failure."""
    backend_url = os.getenv("BACKEND_URL", "https://orchestra-backend-30fy.onrender.com")
    try:
        resp = requests.get(f"{backend_url}/users/{user_id}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("github_username")
    except Exception:
        return None


# In-process cache for the user directory. The backend's /users is slow (~6s) and
# the list barely changes, so re-fetching it on every Clover call both wastes time
# and risks timing out under the request's concurrent load — which silently breaks
# handle resolution ("my tasks" can't map a GitHub name to the username tasks are
# keyed by). Cache it and serve stale on failure.
_USERS_CACHE: dict = {"data": [], "ts": 0.0}
_USERS_TTL = 300  # seconds


# Fetches the full user directory from the backend (cached, best-effort).
def fetch_users() -> list[dict]:
    """Return the backend's user list (cached ~5 min), or the last good list.

    Each row carries the person's several handles — Orchestra `username`,
    `github_username`, `discord_username` — which we use to recognise the current
    user in task data no matter which handle the frontend sent us.
    """
    now = time.time()
    if _USERS_CACHE["data"] and (now - _USERS_CACHE["ts"] < _USERS_TTL):
        return _USERS_CACHE["data"]
    backend_url = os.getenv("BACKEND_URL", "https://orchestra-backend-30fy.onrender.com")
    try:
        resp = requests.get(f"{backend_url}/users", timeout=20)
        resp.raise_for_status()
        users = resp.json().get("users", [])
        if users:
            _USERS_CACHE["data"] = users
            _USERS_CACHE["ts"] = now
        return users
    except Exception:
        return _USERS_CACHE["data"]  # serve the last good directory on a slow/failed call


# Fetches the list of project IDs the user belongs to from the backend.
def fetch_user_project_ids(user_id: str) -> list[str]:
    """Return project IDs the user created or is a member of. Empty list on failure."""
    backend_url = os.getenv("BACKEND_URL", "https://orchestra-backend-30fy.onrender.com")
    try:
        resp = requests.get(f"{backend_url}/projects", params={"user_id": user_id}, timeout=10)
        resp.raise_for_status()
        projects = resp.json().get("projects", [])
        return [p["id"] for p in projects if p.get("id")]
    except Exception:
        return []



# Finds the 3 tasks that best match the user's question using semantic search.
def search_top_tasks(question: str, api_key: str, project_id: str | None = None, allowed_project_ids: list[str] | None = None, tasks: list[dict] | None = None) -> list[dict]:
    """Find the 3 most relevant tasks using the shared, cached ChromaDB index.

    Pass `tasks` to reuse a task list already fetched this request; otherwise it
    reads the graph itself. Sharing one fetch avoids the redundant full-graph
    Neo4j round-trips that used to happen 2-4 times per Clover call.
    """
    if tasks is None:
        tasks = get_all_tasks()

    if not tasks:
        return []

    embed_client = genai.Client(api_key=api_key)
    # Reuse the shared index (embeds every task once, cached across requests)
    # rather than re-embedding on every call. Scope to the project at query time
    # via a metadata filter instead of indexing a per-project subset.
    collection = ensure_indexed(embed_client, tasks)

    query_embedding = get_embedding(embed_client, question)
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": 15,
        "include": ["metadatas", "distances"],
    }
    if project_id:
        query_kwargs["where"] = {"project_id": project_id}
    elif allowed_project_ids:
        query_kwargs["where"] = {"project_id": {"$in": allowed_project_ids}}
    results = collection.query(**query_kwargs)

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # If the project has no tasks in the index, fall back to global search so
    # Clover can still answer — the user shouldn't need to know a blueprint
    # hasn't been run yet.
    if not metadatas and (project_id or allowed_project_ids):
        fallback_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": 15,
            "include": ["metadatas", "distances"],
        }
        results = collection.query(**fallback_kwargs)
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

    matches: list[dict] = []
    for i, metadata in enumerate(metadatas):
        distance = distances[i] if i < len(distances) else None
        matches.append({**metadata, "distance": distance})

    return matches


# Builds the project knowledge graph directly from Neo4j (in-process).
def fetch_graph() -> dict | None:
    """Build the project graph in-process from Neo4j. Returns None on failure.

    Calls build_reactflow_graph() directly instead of doing an HTTP round-trip
    to our own /graph endpoint: that self-call carries no x-api-key, so the auth
    guard returns 401 and the graph context silently goes missing. In-process is
    also faster and needs no network. Same {nodes, edges} shape either way.
    """
    try:
        return build_reactflow_graph()
    except Exception:
        return None


# Checks if a graph node is related to the question by name or assignee.
def is_relevant_node(node: dict, question: str) -> bool:
    """Return True if a graph node matches the question by title or assignee."""
    q = question.lower()
    data = node.get("data", {})
    label = str(data.get("label", "")).lower()
    assigned_to = str(data.get("assigned_to", "")).lower()

    if assigned_to and assigned_to in q:
        return True
    if label and label in q:
        return True
    for word in q.split():
        if len(word) > 2 and word in label:
            return True
    return False


# Fetches the graph and keeps only the nodes and edges that match the question.
def get_relevant_graph_context(question: str, graph: dict | None) -> dict | None:
    """Fetch and filter graph nodes/edges relevant to the question."""
    if not graph:
        return None

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    relevant_nodes = [node for node in nodes if is_relevant_node(node, question)]

    if not relevant_nodes:
        return {"nodes": [], "edges": []}

    relevant_ids = {node["id"] for node in relevant_nodes}
    relevant_edges = [
        edge
        for edge in edges
        if edge.get("source") in relevant_ids or edge.get("target") in relevant_ids
    ]

    return {"nodes": relevant_nodes, "edges": relevant_edges}


# Builds relationship details for matched tasks using the full project graph.
def enrich_with_graph(task_ids: list[str], graph: dict) -> list[dict]:
    """Enrich task IDs with dependencies, assignees, and dependents from the graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    nodes_by_id = {node["id"]: node for node in nodes}

    enriched: list[dict] = []
    for task_id in task_ids:
        task_node = nodes_by_id.get(task_id)
        if not task_node:
            continue

        dependencies: list[dict] = []
        dependents: list[dict] = []
        assigned_to = None

        for edge in edges:
            if edge.get("source") != task_id and edge.get("target") != task_id:
                continue

            relationship = edge.get("data", {}).get("relationship", "")
            source = edge.get("source")
            target = edge.get("target")

            if source == task_id and relationship == "DEPENDS_ON":
                dep_node = nodes_by_id.get(target)
                if dep_node:
                    dependencies.append(dep_node)
            elif target == task_id and relationship == "DEPENDS_ON":
                dependent_node = nodes_by_id.get(source)
                if dependent_node:
                    dependents.append(dependent_node)
            elif target == task_id and relationship == "ASSIGNED_TO":
                developer_node = nodes_by_id.get(source)
                if developer_node:
                    assigned_to = developer_node

        enriched.append(
            {
                "task": task_node,
                "dependencies": dependencies,
                "assigned_to": assigned_to,
                "dependents": dependents,
            }
        )

    return enriched


def _compact_events(events: list[dict], limit: int = 15) -> list[dict]:
    """Trim live events before they go into the prompt.

    fetch_live_events() returns the whole backlog (dozens of events, each carrying
    a bulky raw_metadata webhook blob). Dumping all of it raw wastes thousands of
    tokens and buries the signal. Keep only the most recent `limit` events and only
    the fields that help Clover reason about recent activity.
    """
    if not events:
        return []
    keep = ("platform", "event_type", "actor", "timestamp", "repo", "channel", "action_summary")
    ordered = sorted(events, key=lambda e: str(e.get("timestamp", "")), reverse=True)
    return [{k: e[k] for k in keep if e.get(k) is not None} for e in ordered[:limit]]


# Builds the ordered prompt sections from all pre-fetched context sources.
def _build_prompt_parts(
    question: str,
    task_context: list[dict],
    conversation_history: list[dict] | None,
    live_events,
    full_graph,
    project_id: str | None = None,
    project_names: dict[str, str] | None = None,
    all_tasks: list[dict] | None = None,
) -> list[str]:
    """Shared prompt builder used by both ask_clover and stream_answer.

    `all_tasks` may be a task list already fetched this request; when omitted the
    project-scoping step below reads the graph itself.
    """
    if project_id and full_graph:
        scope_source = all_tasks if all_tasks is not None else get_all_tasks()
        project_task_ids = {t.get("id") for t in scope_source if t.get("project_id") == project_id}
        full_graph["nodes"] = [n for n in full_graph.get("nodes", []) if n.get("id") in project_task_ids or n.get("type") == "developer"]
        full_graph["edges"] = [e for e in full_graph.get("edges", []) if e.get("source") in project_task_ids or e.get("target") in project_task_ids]

    # Tag each task with its project name so Gemini can cite it (the graph only
    # carries the project id, embedded in the task id). Best-effort — an empty
    # map just means tasks are cited by id alone, as before.
    name_map = fetch_project_names() if project_names is None else project_names
    tagged_context = []
    for task in task_context:
        pname = project_name_for_task(str(task.get("id", "")), name_map)
        tagged_context.append({**task, "project": pname} if pname else task)

    context_json = json.dumps(tagged_context, indent=2, ensure_ascii=False)
    prompt_parts = [f"Task context:\n{context_json}"]

    if conversation_history:
        history_text = "Conversation history (most recent last):\n"
        for item in conversation_history[-5:]:
            if not isinstance(item, dict):
                continue
            # Be tolerant of the client's history shape. Our own is
            # {question, answer}, but chat UIs often send {role, content} or
            # {sender, text} bubbles (incl. an opening greeting with no
            # question). Never index a key directly — a missing key used to
            # raise KeyError and turn the whole request into a 500.
            q = str(item.get("question") or "")
            a = str(item.get("answer") or "")
            if not q and not a:
                content = str(item.get("content") or item.get("text") or item.get("message") or "")
                if not content:
                    continue
                role = str(item.get("role") or item.get("sender") or "").lower()
                if role in ("user", "human"):
                    q = content
                else:
                    a = content
            history_text += f"User: {q}\nClover: {a}\n"
        history_text += (
            "\nIf the current question refers back to this conversation "
            '(e.g. "those changes", "that task", "they"), resolve the reference '
            "using the exchanges above."
        )
        prompt_parts.append(history_text)

    compact_events = _compact_events(live_events)
    if compact_events:
        commit_json = json.dumps(compact_events, indent=2, ensure_ascii=False)
        prompt_parts.append(f"Recent activity context:\n{commit_json}")

    graph_context = get_relevant_graph_context(question, full_graph)
    if graph_context is not None:
        graph_json = json.dumps(graph_context, indent=2, ensure_ascii=False)
        prompt_parts.append(f"Graph context:\n{graph_json}")

    task_ids = [t.get("id") for t in task_context if t.get("id")]
    if full_graph:
        enriched = enrich_with_graph(task_ids, full_graph)
        if enriched:
            enriched_json = json.dumps(enriched, indent=2, ensure_ascii=False)
            prompt_parts.append(
                "Enriched graph context (relationships for matched tasks):\n"
                f"{enriched_json}"
            )

    prompt_parts.append(f"User question: {question}")
    return prompt_parts


# Sends all context to Gemini and returns Clover's answer as text.
def ask_clover(
    question: str,
    task_context: list[dict],
    api_key: str,
    conversation_history: list[dict] = None,
    project_id: str | None = None,
    graph=_UNSET,
    live_events=_UNSET,
    project_names: dict[str, str] | None = None,
) -> str:
    """Send retrieved tasks and graph context to Gemini and return an answer.

    `graph` and `live_events` may be passed in pre-fetched (see answer_question,
    which retrieves them in parallel with the semantic search). When left unset,
    they're fetched here so the CLI / direct callers keep working unchanged.
    `project_names` is the {id: name} map for citing project names (fetched in
    _build_prompt_parts when None).
    """
    full_graph = fetch_graph() if graph is _UNSET else graph
    if live_events is _UNSET:
        try:
            live_events = fetch_live_events()
        except Exception:
            live_events = None

    client = genai.Client(api_key=api_key)
    prompt_parts = _build_prompt_parts(
        question, task_context, conversation_history, live_events, full_graph,
        project_id, project_names,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="\n\n".join(prompt_parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )

    return (response.text or "").strip()


# Yields Gemini response chunks as they're generated, then a final history event.
def stream_answer(
    question: str,
    api_key: str,
    conversation_history: list[dict] = None,
    project_id: str | None = None,
    github_username: str | None = None,
    user_id: str | None = None,
    pending_action: dict | None = None,
    user_name: str | None = None,
):
    """Retrieve context in parallel, then stream Gemini's answer chunk by chunk.

    Each yielded value is a JSON string in one of two shapes:
      {"chunk": "text"}              — a piece of the answer as Gemini generates it
      {"done": true, "conversation_history": [...]}  — sent once at the end

    The caller wraps each in "data: ...\\n\\n" for SSE. Errors are yielded as
    {"error": "message", "status": <code>} so the frontend can display them
    even though HTTP headers are already sent by the time we fail.
    """
    # If the user is confirming a pending action from the previous turn, execute it
    # immediately and stream a confirmation — skip all retrieval and Gemini.
    if pending_action and _CONFIRMATION_RE.match(question):
        task_id = pending_action.get("task_id")
        new_status = pending_action.get("new_status")
        title = pending_action.get("title", task_id)
        if task_id and new_status:
            result = patch_task_status(task_id, new_status)
            if result:
                msg = f"Done — I've marked \"{title}\" as {new_status}."
            else:
                msg = f"Sorry, I couldn't update \"{title}\" — the task may no longer exist."
            yield json.dumps({"chunk": msg})
            updated_history = (conversation_history or []) + [{"question": question, "answer": msg}]
            yield json.dumps({"done": True, "conversation_history": updated_history[-5:]})
            return

    # Task update detection runs after the parallel fetch so semantic matching
    # can use the already-retrieved relevant_tasks (no extra API calls needed).
    task_update = None
    task_update_prompt = None
    pending_action_out: dict | None = None

    want_graph = _needs_graph(question)
    want_events = _needs_events(question)
    if not want_graph and not want_events:
        want_graph = want_events = True

    empty_direct_lookup = False
    self_prefiltered = False
    have_identity_hint = bool(user_id or github_username or user_name)
    with ThreadPoolExecutor(max_workers=6) as pool:
        project_ids_future = pool.submit(fetch_user_project_ids, user_id) if user_id and not project_id else None
        gh_username_future = pool.submit(fetch_github_username, user_id) if user_id and not github_username else None
        # Pull the user directory so we can recognise the caller in task data by any
        # of their handles (Orchestra username / GitHub / Discord), not only the one
        # the frontend happened to send. Only when we have something to resolve.
        users_future = pool.submit(fetch_users) if have_identity_hint else None
        # Fetch the full task list ONCE per request, in parallel with the graph/
        # events/projects fetches. Everything below reuses it instead of each
        # re-querying Neo4j (search, graph scoping, capacity planning, prompt build).
        tasks_data_future = pool.submit(get_all_tasks)
        graph_future = pool.submit(fetch_graph) if want_graph else None
        events_future = pool.submit(_safe_fetch_live_events) if want_events else None
        projects_future = pool.submit(fetch_projects)

        allowed_project_ids = project_ids_future.result() if project_ids_future else None
        if gh_username_future:
            github_username = gh_username_future.result() or github_username
        all_tasks = tasks_data_future.result()
        users = users_future.result() if users_future else []
        identity = resolve_identity(user_id, github_username, user_name, users)

        # Status/assignee questions ("what's blocked?", "what is Isha working on?")
        # get a direct filter over the real task data — the full matching set, not
        # the text-nearest tasks a semantic search would return. Update COMMANDS,
        # though, must keep semantic search: the fuzzy-match update path below needs
        # the per-task `distance` scores, which the direct filter doesn't produce.
        # So reads use the direct filter; update intents fall through to the index.
        direct_tasks = (
            None if _has_update_intent(question)
            else retrieve_by_status_or_assignee(
                question, all_tasks, project_id, allowed_project_ids
            )
        )
        if direct_tasks is None:
            relevant_tasks = search_top_tasks(
                question, api_key, project_id, allowed_project_ids, tasks=all_tasks
            )
        else:
            relevant_tasks = direct_tasks
            empty_direct_lookup = len(direct_tasks) == 0

        # First-person self-lookup: when the user asks about their own tasks and we
        # can recognise them in the data by ANY of their handles, filter to exactly
        # their tasks here rather than leaving the match to the model. This is what
        # makes "what are my tasks" work when the GitHub handle (e.g. "ArnavXT")
        # differs from the display name in assigned_to (e.g. "Arnav").
        if (
            not _has_update_intent(question)
            and _is_first_person_task_q(question)
            and identity and identity.get("cores")
            and (project_id or allowed_project_ids)
        ):
            scoped_self = _scope_tasks(all_tasks, project_id, allowed_project_ids)
            mine = [
                t for t in scoped_self
                if _assignee_matches_aliases(identity["cores"], t.get("assigned_to", ""))
            ]
            if mine:
                relevant_tasks = sorted(mine, key=lambda t: str(t.get("id", "")))[:25]
                empty_direct_lookup = False
                self_prefiltered = True

        graph = graph_future.result() if graph_future else None
        live_events = events_future.result() if events_future else None
        projects = projects_future.result()
        project_names = {p["id"]: p.get("name", "") for p in projects if p.get("id")}

    # Task update detection: regex ID match first (fast path), then semantic matching.
    action = _detect_task_action(question)
    if action:
        task_id, new_status = action
        result = patch_task_status(task_id, new_status)
        task_update = {
            "task_id": task_id,
            "new_status": new_status,
            "title": result.get("title") if result else None,
            "success": result is not None,
        }
    elif _has_update_intent(question) and relevant_tasks:
        new_status = _detect_status_from_question(question)
        if new_status:
            top = relevant_tasks[0]
            distance = float(top.get("distance") or 1.0)
            if distance < _TASK_HIGH_CONFIDENCE:
                task_id = top.get("id")
                title = top.get("title", task_id)
                pending_action_out = {"task_id": task_id, "new_status": new_status, "title": title}
                task_update_prompt = (
                    f"System: The user wants to mark \"{title}\" as {new_status}. "
                    f"Ask for confirmation in one friendly sentence, e.g. "
                    f"\"I'll mark '{title}' as {new_status} — shall I go ahead?\" "
                    "Do not update anything yet."
                )
            elif distance < _TASK_LOW_CONFIDENCE:
                candidates = "\n".join(
                    f"- {t.get('title')} (ID: {t.get('id')})"
                    for t in relevant_tasks[:3]
                    if t.get("title")
                )
                task_update_prompt = (
                    f"System: The user wants to mark a task as {new_status} but it's unclear which one. "
                    f"The closest matches are:\n{candidates}\n"
                    "Ask the user which task they meant in a friendly way, showing the task titles as options. "
                    "Do not update anything yet."
                )
            else:
                task_update_prompt = (
                    "System: I couldn't find a task matching the user's description. "
                    "Apologise briefly and ask them to describe the task differently or paste the task ID."
                )

    # Scope graph to the user's projects so other users' tasks don't leak into context.
    if graph and allowed_project_ids and not project_id:
        allowed_task_ids = {t["id"] for t in all_tasks if t.get("project_id") in allowed_project_ids}
        graph["nodes"] = [n for n in graph.get("nodes", []) if n.get("id") in allowed_task_ids or n.get("type") == "developer"]
        graph["edges"] = [e for e in graph.get("edges", []) if e.get("source") in allowed_task_ids or e.get("target") in allowed_task_ids]

    # Detect navigation/project-switch early so we can inject a confirmation hint
    # into the prompt — otherwise a bare command like "open the amazon workflow"
    # gives Gemini nothing to say and produces an empty response.
    nav_action = (
        _detect_repo_redirect(question, project_id, projects)
        or _detect_project_switch(question, project_names)
        or _detect_navigation(question, project_id, project_names)
    )

    # Project-specific nav (workflow, kanban, team, etc.) with no project in context
    # → ask which project rather than firing a broken action with project_id=null.
    _PROJECT_SPECIFIC_DESTS = {"workflow", "tasks", "team", "activity", "blueprint"}
    if (nav_action
            and nav_action.get("type") == "navigate"
            and nav_action.get("destination") in _PROJECT_SPECIFIC_DESTS
            and not nav_action.get("project_id")):
        dest = nav_action.get("destination")
        nav_action = None
        fuzzy = _fuzzy_project_match(question, project_names)
        if fuzzy:
            _fid, fname = fuzzy
            matched_word = _find_matched_word(question, fname)
            task_update_prompt = (
                f"System: The user said '{matched_word}' which partially matches the project '{fname}'. "
                f"Ask them in one friendly sentence: 'By \"{matched_word}\" did you mean {fname}?'"
            )
        else:
            known = [name for name in project_names.values() if name]
            projects_hint = f" The available projects are: {', '.join(known)}." if known else ""
            task_update_prompt = (
                f"System: The user wants to go to the {dest} page but hasn't said which project.{projects_hint} "
                "Ask them which project they mean in one friendly sentence."
            )

    client = genai.Client(api_key=api_key)
    prompt_parts = _build_prompt_parts(
        question, relevant_tasks, conversation_history, live_events, graph,
        project_id, project_names, all_tasks=all_tasks,
    )

    # Tell Gemini who "my"/"me"/"I" refers to so first-person questions ("what are
    # my tasks") resolve instead of the model refusing for lack of a name.
    for part in reversed(_identity_prompt_parts(
        identity, question, all_tasks, project_id, allowed_project_ids,
        prefiltered=self_prefiltered,
    )):
        prompt_parts.insert(0, part)

    # A direct status/assignee lookup that matched nothing is a real "there are
    # none" answer — tell Gemini so it doesn't invent tasks to fill the silence.
    if empty_direct_lookup:
        prompt_parts.insert(
            0,
            "System note: a direct lookup of the task graph found NO tasks matching "
            "that status or person. Tell the user plainly that there are none — do "
            "not invent or guess tasks.",
        )

    if nav_action:
        action_type = nav_action.get("type")
        dest = nav_action.get("destination", "")
        pid = nav_action.get("project_id", "")
        pname = project_names.get(pid, "") if pid else ""
        if action_type == "switch_project":
            where = f"{dest} of {pname}" if dest and pname else (pname or dest or "the project")
            prompt_parts.insert(0,
                f"System action complete: You are navigating the user to the {where}. "
                "Do NOT say you cannot do this or that you lack information. "
                "Simply confirm in one short casual sentence that you're taking them there right now.")
        elif action_type == "navigate":
            where = f"{dest} page" + (f" for {pname}" if pname else "")
            prompt_parts.insert(0,
                f"System action complete: You are navigating the user to the {where}. "
                "Do NOT say you cannot do this or that you lack information. "
                "Simply confirm in one short casual sentence that you're taking them there right now.")
        elif action_type == "open_url":
            prompt_parts.insert(0,
                "System action complete: You are opening the GitHub repo for the user. "
                "Do NOT say you cannot do this. "
                "Simply confirm in one short casual sentence that you're opening it.")

    if _needs_capacity_planning(question):
        if project_id:
            scoped_tasks = [t for t in all_tasks if t.get("project_id") == project_id]
        elif allowed_project_ids:
            scoped_tasks = [t for t in all_tasks if t.get("project_id") in allowed_project_ids]
        else:
            scoped_tasks = all_tasks
        user_hint = f" The user's GitHub username is {github_username} — use this to identify which tasks are assigned to them (the assigned_to field may use their display name or a variation of it)." if github_username else ""
        prompt_parts.insert(
            0,
            f"Capacity planning context:{user_hint} The user is asking how much they can realistically get done.\n"
            f"Full task list:\n{json.dumps(scoped_tasks, indent=2, ensure_ascii=False)}\n\n"
            "Rules for a realistic estimate:\n"
            "- A focused developer can typically complete 2-3 tasks per day.\n"
            "- Prioritise tasks already in_progress (finishing beats starting).\n"
            "- Next pick high-priority todos with no unresolved blockers.\n"
            "- Skip blocked tasks entirely for today.\n"
            "- Be honest and specific — name exactly which tasks to focus on and why, don't just list everything.",
        )

    if task_update_prompt:
        prompt_parts.insert(0, task_update_prompt)
    elif task_update:
        if task_update["success"]:
            prompt_parts.insert(
                0,
                f"System action: Task {task_update['task_id']} "
                f"(\"{task_update['title']}\") has been updated to "
                f"\"{task_update['new_status']}\". "
                "Confirm this to the user naturally in one sentence.",
            )
        else:
            prompt_parts.insert(
                0,
                f"System action failed: Could not find or update task "
                f"{task_update['task_id']}. Let the user know naturally.",
            )

    full_answer = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents="\n\n".join(prompt_parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    ):
        if chunk.text:
            full_answer += chunk.text
            yield json.dumps({"chunk": chunk.text})

    # Never let an empty answer reach the client — it renders the raw done-event
    # JSON. Gemini can come back empty on a bare command (e.g. a nav request whose
    # project the backend was too cold to resolve). Emit a sensible fallback so
    # the user always sees text, and a nav miss reads as "try again", not silence.
    if not full_answer.strip():
        if nav_action:
            full_answer = "Taking you there now."
        elif any(_phrase_present(kw, question.lower()) for kw in _NAV_KEYWORDS):
            full_answer = (
                "I couldn't reach the project list just now — mind trying that again in a moment?"
            )
        else:
            full_answer = "I don't have enough information to answer that right now."
        yield json.dumps({"chunk": full_answer})

    task_updates: list[dict] = []
    if github_username and _needs_github_update(question):
        try:
            task_updates = update_tasks_from_github(github_username, api_key, project_id)
        except Exception:
            pass

    # Don't suggest tasks for navigation requests — they'd be irrelevant noise.
    suggested_tasks = [] if nav_action else [
        {"id": t.get("id"), "title": t.get("title"), "project_id": t.get("project_id")}
        for t in relevant_tasks[:5]
        if t.get("id") and t.get("title")
    ]

    updated_history = (conversation_history or []) + [
        {"question": question, "answer": full_answer}
    ]
    done_payload: dict = {"done": True, "conversation_history": updated_history[-5:]}
    if task_updates:
        done_payload["task_updates"] = task_updates
    if suggested_tasks:
        done_payload["suggested_tasks"] = suggested_tasks
    if nav_action:
        done_payload["action"] = nav_action
    if pending_action_out:
        done_payload["pending_action"] = pending_action_out
    yield json.dumps(done_payload)


# Fetches live events but never raises — matches how ask_clover treated failures.
def _safe_fetch_live_events():
    """fetch_live_events(), returning None instead of raising on any failure."""
    try:
        return fetch_live_events()
    except Exception:
        return None


_TASK_ID_RE = re.compile(r'\b(?:[A-Z0-9]+-)?T\d+\b', re.IGNORECASE)
_TASK_HIGH_CONFIDENCE = 0.35  # distance below this → update directly
_TASK_LOW_CONFIDENCE = 0.65   # distance above this → no match, apologise


def _scope_tasks(tasks: list[dict], project_id: str | None = None,
                 allowed_project_ids: list[str] | None = None) -> list[dict]:
    """Narrow a task list to a single project or a set of the user's projects."""
    if project_id:
        return [t for t in tasks if t.get("project_id") == project_id]
    if allowed_project_ids:
        allowed = set(allowed_project_ids)
        return [t for t in tasks if t.get("project_id") in allowed]
    return list(tasks)


# Status-question patterns → the canonical status they map to. Semantic search
# can't answer these reliably (it returns the 3 text-nearest tasks, not the full
# set in a given state), so we filter the graph data directly instead.
_STATUS_INTENT: list[tuple[str, list[str]]] = [
    ("blocked", [r"\bblocked\b", r"\bstuck\b", r"\bblocking\b", r"can'?t\s+(?:start|proceed|move)"]),
    ("in_progress", [r"\bin[\s-]?progress\b", r"being\s+worked\s+on",
                     r"currently\s+working", r"\bunderway\b", r"\bongoing\b"]),
    ("completed", [r"\bcompleted\b", r"\bfinished\b", r"what'?s\s+done",
                   r"\bdone\s+tasks?\b", r"tasks?\s+(?:are\s+)?done",
                   r"already\s+(?:done|completed|finished)"]),
    ("upcoming", [r"\bupcoming\b", r"not\s+started", r"\btodo\b", r"to[\s-]?do",
                  r"haven'?t\s+started", r"yet\s+to\s+start", r"\bbacklog\b",
                  r"what'?s\s+left", r"remaining\s+tasks?"]),
]

_ASSIGNEE_TRIGGER = re.compile(
    r"working\s+on|assigned\s+to|responsible\s+for|handling|"
    r"'s\s+tasks?|tasks?\s+(?:for|of)\b|\bowns?\b|who\s+has|what\s+is\s+\w+\s+doing",
    re.IGNORECASE,
)


def _status_intent(question: str) -> str | None:
    """Return the canonical status a question is asking about, or None."""
    q = question.lower()
    for status, patterns in _STATUS_INTENT:
        if any(re.search(p, q) for p in patterns):
            return status
    return None


def _assignee_intent(question: str, tasks: list[dict]) -> str | None:
    """Return the assignee a question is about, matched against real assignees.

    Only fires when the phrasing is relational ("working on", "assigned to", ...)
    AND a name that actually appears in the task data is named in the question, so
    it can't misfire on arbitrary words.
    """
    if not _ASSIGNEE_TRIGGER.search(question):
        return None
    q = question.lower()
    names = {str(t.get("assigned_to")) for t in tasks if t.get("assigned_to")}
    for name in sorted(names, key=len, reverse=True):
        nl = name.strip().lower()
        if not nl or nl == "none":
            continue
        if re.search(rf"\b{re.escape(nl)}\b", q):
            return name
        first = nl.split()[0] if nl.split() else nl
        if len(first) >= 3 and re.search(rf"\b{re.escape(first)}\b", q):
            return name
    return None


def retrieve_by_status_or_assignee(
    question: str,
    tasks: list[dict],
    project_id: str | None = None,
    allowed_project_ids: list[str] | None = None,
) -> list[dict] | None:
    """Direct graph-data retrieval for status/assignee questions.

    Returns the FULL set of tasks matching the asked-about status and/or assignee
    (scoped to the user's projects), or None when the question isn't a
    status/assignee query — in which case the caller falls back to semantic search.
    An empty list means "it is such a query, but nothing matches" (e.g. nothing is
    blocked), which is a real answer, not a fall-through.
    """
    scoped = _scope_tasks(tasks, project_id, allowed_project_ids)
    status = _status_intent(question)
    assignee = _assignee_intent(question, scoped)
    if status is None and assignee is None:
        return None
    matches = scoped
    if status is not None:
        matches = [t for t in matches if str(t.get("status", "")).strip().lower() == status]
    if assignee is not None:
        al = str(assignee).strip().lower()
        matches = [t for t in matches if str(t.get("assigned_to", "")).strip().lower() == al]
    return sorted(matches, key=lambda t: str(t.get("id", "")))[:25]


# Matches first-person questions about the user's own work ("what are my tasks",
# "what am I working on", "assigned to me", "my workload"). Needs BOTH a first-
# person pronoun AND a task/work word so it doesn't fire on every "I ...".
_FIRST_PERSON_RE = re.compile(r"\b(?:my|mine|me|i)\b", re.IGNORECASE)
_SELF_TASK_RE = re.compile(
    r"\btasks?\b|\bworking\b|\bwork\b|\bassigned\b|\bdoing\b|\bto-?do\b|"
    r"\bresponsible\b|\bplate\b|\bworkload\b|\bwork\s?load\b",
    re.IGNORECASE,
)


def _is_first_person_task_q(question: str) -> bool:
    """True if the user is asking about their OWN tasks/work."""
    return bool(_FIRST_PERSON_RE.search(question) and _SELF_TASK_RE.search(question))


# "who am i", "what's my name/username/handle" — identity questions. Kept separate
# from task questions so we can answer them from resolved identity, and crucially
# refuse to GUESS one from recent activity when we can't (which produced wrong,
# ever-changing answers like a random event actor's handle).
_IDENTITY_Q_RE = re.compile(
    r"who\s+am\s+i|what'?s\s+my\s+(?:name|username|handle|user\s*name)|"
    r"what\s+is\s+my\s+(?:name|username|handle|user\s*name)|my\s+username",
    re.IGNORECASE,
)


def _is_identity_question(question: str) -> bool:
    """True if the user is asking who they are / what their username is."""
    return bool(_IDENTITY_Q_RE.search(question))


def _handle_core(s: str) -> str:
    """Reduce a handle to its leading name token, lowercased.

    Collapses the many shapes a person's handle takes to a common root so they all
    line up with the short display name used in `assigned_to`:
        "ArnavXT" -> "arnav"   (camelCase boundary)
        "Arnav21" -> "arnav"   (stops at the digits)
        "arnav.xo" -> "arnav"  (stops at the separator)
        "Arnav"   -> "arnav"
    """
    s = str(s or "").strip()
    if not s:
        return ""
    # First name-ish token with a lowercase tail, anywhere in the string — so a
    # leading separator (".inaman") or digits don't swallow the whole handle.
    m = re.search(r"[A-Za-z][a-z]+", s)  # camelCase-aware (stops at the next uppercase/digit)
    if m:
        return m.group(0).lower()
    m = re.search(r"[A-Za-z]+", s)        # all-caps / single-letter fallback
    return m.group(0).lower() if m else ""


def _assignee_matches_aliases(alias_cores: set[str], assigned_to: str) -> bool:
    """True if `assigned_to` names the same person as any of the user's handles.

    Compares on the normalised root and allows a first-name-style prefix match
    (so "prince" matches "princen") while staying strict enough (>=4 chars) not to
    collide unrelated short names.
    """
    a = _handle_core(assigned_to)
    if not a:
        return False
    for n in alias_cores:
        if not n:
            continue
        if n == a:
            return True
        short, long = (a, n) if len(a) <= len(n) else (n, a)
        if len(short) >= 4 and long.startswith(short):
            return True
    return False


def resolve_identity(
    user_id: str | None,
    github_username: str | None,
    user_name: str | None,
    users: list[dict],
) -> dict | None:
    """Resolve the caller to a display name + every handle they're known by.

    We may be handed any one of user_id / github_username / user_name by the
    frontend. We look the person up in the backend user directory to gather their
    other handles too, so "my tasks" resolves whether their GitHub name matches
    their Orchestra display name or not. Returns None when we have nothing to go on.
    """
    if not (user_id or github_username or user_name):
        return None

    seeds = [x for x in (user_name, github_username) if x]
    # Whatever the frontend sent (an id or any handle), find the directory row by
    # matching it against ANY of the person's handle fields — id, username, GitHub,
    # or Discord — then borrow all their other handles from that row.
    provided = {str(v).strip().lower() for v in (user_id, github_username, user_name) if v}
    row = None
    if users and provided:
        for u in users:
            handles = {
                str(u.get(k, "")).strip().lower()
                for k in ("user_id", "username", "github_username", "discord_username")
            }
            handles.discard("")
            if provided & handles:
                row = u
                break

    raw: list[str] = list(seeds)
    if row:
        raw += [row.get("username"), row.get("github_username"), row.get("discord_username")]
    raw = [str(x) for x in raw if x]
    # de-dupe, preserve order (for the human-readable note)
    raw = list(dict.fromkeys(raw))

    cores = {_handle_core(x) for x in raw}
    cores.discard("")

    display = user_name or (row.get("username") if row else None) or github_username
    if not display and not cores:
        return None
    return {"display": display, "aliases": raw, "cores": cores}


def _identity_prompt_parts(
    identity: dict | None,
    question: str,
    all_tasks: list[dict],
    project_id: str | None,
    allowed_project_ids: list[str] | None,
    prefiltered: bool = False,
) -> list[str]:
    """Prompt notes that tell Gemini who "my"/"me"/"I" refers to.

    `identity` is the dict from resolve_identity() (display name + every handle the
    user is known by), or None when the request carried no identity at all. When we
    know the user we inject their name and handles for every question. For a first-
    person task question we prefer a server-side match (`prefiltered=True`, the task
    context already holds exactly their tasks); if that found nothing we instead
    hand Gemini the project-scoped list so it can match by any handle itself.
    """
    first_person = _is_first_person_task_q(question)
    identity_q = _is_identity_question(question)
    if identity and identity.get("display"):
        display = identity["display"]
        aliases = identity.get("aliases") or []
        alias_str = ", ".join(aliases) if aliases else display
        parts = [
            f"User identity: you are talking to {display}. In task data this person "
            f"may appear as any of these handles: {alias_str} — the assigned_to field "
            "usually uses a short display name (often the first name). When the user "
            'says "my", "me", "mine", or "I", they mean tasks assigned to this person. '
            f"If they ask who they are or for their username, tell them: {display}."
        ]
        if first_person and prefiltered:
            parts.append(
                "The Task context above has already been filtered to exactly this "
                "user's tasks — list those (say plainly if there are none); don't "
                "invent tasks that aren't there."
            )
        elif first_person and (project_id or allowed_project_ids):
            scoped = _scope_tasks(all_tasks, project_id, allowed_project_ids)
            slim = [
                {
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "status": t.get("status"),
                    "assigned_to": t.get("assigned_to"),
                    "project_id": t.get("project_id"),
                }
                for t in scoped
            ][:60]
            parts.append(
                "The user is asking about their own tasks. Below is the full task "
                "list for their project(s); list the ones assigned to this person by "
                "matching any of the handles above against assigned_to (allow for "
                "short/display-name variations), and don't invent tasks that aren't "
                f"here:\n{json.dumps(slim, indent=2, ensure_ascii=False)}"
            )
        return parts
    if first_person or identity_q:
        # We couldn't resolve who's asking (no identity sent, or the id didn't match
        # any user record). Critically: do NOT let the model invent a name from
        # recent activity — that produced wrong, ever-changing answers. Say we don't
        # know and ask, warmly.
        return [
            "You could NOT determine who the user is — the request carried no "
            "resolvable identity. Do NOT guess their name or username from recent "
            "activity, events, or task data; that would be wrong. In one friendly "
            "sentence, say you're not sure who they are and ask them to tell you "
            "their username."
        ]
    return []


_COMPLETED_RE = re.compile(r'\b(?:done|finish(?:ed)?|complet(?:ed)?|wrap(?:ped)?\s*up)\b', re.IGNORECASE)
_IN_PROGRESS_RE = re.compile(r'\b(?:start(?:ing|ed)?|working\s+on|began|beginning|picking\s+up)\b', re.IGNORECASE)
_BLOCKED_RE = re.compile(r'\b(?:block(?:ed)?|stuck|can\'t\s+(?:start|do|work))\b', re.IGNORECASE)
# A question opens with one of these (or ends in "?"). The loose verb match must
# not fire on those — "is T5 done?" / "who is working on T5?" are READS, not
# commands, and must never mutate a task's status.
_INTERROGATIVE_RE = re.compile(
    r'^\s*(?:who|what|when|where|why|which|whose|whom|is|are|was|were|do|does|'
    r'did|has|have|had|can|could|should|would|will|am|any|anyone|anybody)\b',
    re.IGNORECASE,
)


def _detect_task_action(question: str) -> tuple[str, str] | None:
    """Return (task_id, new_status) if the question is a status update, else None."""
    task_ids = _TASK_ID_RE.findall(question)
    if not task_ids:
        return None
    task_id = task_ids[0].upper()

    # Explicit "mark T4 as X" takes priority
    mark_match = re.search(r'\bmark\s+(?:[A-Z0-9]+-)?T\d+\s+as\s+([\w\s]+)', question, re.IGNORECASE)
    if mark_match:
        label = mark_match.group(1).strip().lower()
        if any(w in label for w in ("complet", "done", "finish")):
            return task_id, "completed"
        if any(w in label for w in ("progress", "start", "active")):
            return task_id, "in_progress"
        if any(w in label for w in ("block", "stuck")):
            return task_id, "blocked"
        if any(w in label for w in ("upcoming", "todo")):
            return task_id, "upcoming"

    # Below is the loose verb match. Only let it fire on a statement or command
    # ("T5 is done", "finished T5"), never on a question ("is T5 done?", "who is
    # working on T5?") — otherwise merely asking about a task mutated its status.
    # The explicit "mark ... as" command above is exempt (already returned).
    if question.strip().endswith("?") or _INTERROGATIVE_RE.match(question):
        return None

    if _COMPLETED_RE.search(question):
        return task_id, "completed"
    if _IN_PROGRESS_RE.search(question):
        return task_id, "in_progress"
    if _BLOCKED_RE.search(question):
        return task_id, "blocked"
    return None


def _detect_status_from_question(question: str) -> str | None:
    """Return the target status from a statement/command, or None if not detectable."""
    mark_match = re.search(r'\bmark\s+.+?\s+as\s+([\w\s]+)', question, re.IGNORECASE)
    if mark_match:
        label = mark_match.group(1).strip().lower()
        if any(w in label for w in ("complet", "done", "finish")):
            return "completed"
        if any(w in label for w in ("progress", "start", "active")):
            return "in_progress"
        if any(w in label for w in ("block", "stuck")):
            return "blocked"
        if any(w in label for w in ("upcoming", "todo")):
            return "upcoming"
    if _COMPLETED_RE.search(question):
        return "completed"
    if _IN_PROGRESS_RE.search(question):
        return "in_progress"
    if _BLOCKED_RE.search(question):
        return "blocked"
    return None


def _has_update_intent(question: str) -> bool:
    """Return True if the question looks like a status update command (not a question)."""
    if question.strip().endswith("?") or _INTERROGATIVE_RE.match(question):
        return False
    return _detect_status_from_question(question) is not None


_CONFIRMATION_RE = re.compile(
    r'^\s*(?:yes|yeah|yep|yup|sure|go ahead|do it|confirm|ok|okay|sounds good|correct|right)\s*[.!]?\s*$',
    re.IGNORECASE,
)


_GITHUB_UPDATE_KEYWORDS = {
    "update kanban", "update my kanban", "check my github", "check github",
    "update from github", "sync github", "github update", "sync my kanban",
    "update tasks from github", "check my commits",
}


def _needs_github_update(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _GITHUB_UPDATE_KEYWORDS)


def _infer_status_from_event(event_type: str) -> str | None:
    """Map a normalized event type to a task status, or None if not actionable."""
    et = event_type.lower()
    if "merge" in et:
        return "completed"
    if "push" in et or "pull" in et or "pr" in et:
        return "in_progress"
    return None


def update_tasks_from_github(
    github_username: str,
    api_key: str,
    project_id: str | None = None,
) -> list[dict]:
    """Fetch recent GitHub events for a user, map to tasks, and update Neo4j.

    Returns a list of {task_id, title, new_status, event_summary, success} dicts
    for every task that was updated, so the done event can report what changed.
    Capped at 5 events to keep Gemini calls bounded.
    """
    events = _safe_fetch_live_events() or []
    user_events = [
        e for e in events
        if e.get("platform") == "github"
        and str(e.get("actor", "")).lower() == github_username.lower()
    ]
    if not user_events:
        return []

    all_tasks = get_all_tasks()
    tasks = [t for t in all_tasks if not project_id or t.get("project_id") == project_id]
    if not tasks:
        return []

    client = genai.Client(api_key=api_key)
    updates: list[dict] = []

    for event in user_events[:5]:
        try:
            link = link_event_to_task(event, tasks, client)
            task_id = link.get("linked_task_id")
            if not task_id:
                continue
            new_status = _infer_status_from_event(event.get("event_type", ""))
            if not new_status:
                continue
            result = patch_task_status(task_id, new_status)
            updates.append({
                "task_id": task_id,
                "title": link.get("linked_task_title"),
                "new_status": new_status,
                "event_summary": event.get("action_summary", ""),
                "success": result is not None,
            })
        except Exception:
            continue

    return updates


_REPO_KEYWORDS = {"open the repo", "open repo", "go to the repo", "open github", "github repo", "open the github", "repo link", "github link"}

# Explicit project-switch intent (separate from general nav keywords)
_SWITCH_KEYWORDS = {"switch to", "switch project", "change project", "change to"}


def _detect_repo_redirect(question: str, project_id: str | None, projects: list[dict]) -> dict | None:
    """Return an open_url action if the question asks to open the GitHub repo."""
    q = question.lower()
    if not any(_phrase_present(kw, q) for kw in _REPO_KEYWORDS):
        return None

    # Fuzzy-match project name from question first
    for p in projects:
        name = p.get("name", "")
        if name and name.lower() in q:
            url = p.get("github_repo_url")
            if url:
                return {"type": "open_url", "url": url}

    # Fall back to current project_id
    if project_id:
        for p in projects:
            if p.get("id") == project_id:
                url = p.get("github_repo_url")
                if url:
                    return {"type": "open_url", "url": url}

    return None


def _detect_project_switch(question: str, project_names: dict[str, str]) -> dict | None:
    """Return a switch_project action if the question targets a specific project by name.

    Triggers on explicit switch phrasing ("switch to X") OR general nav phrasing
    ("take me to X", "open X") when X matches a known project name. The project
    name match is the disambiguator — "take me to the dashboard" has no project
    name and falls through to _detect_navigation, while "take me to PantryPal"
    returns switch_project.

    Matching is word-level so "open the amazon project workflow" matches a project
    named "Amazon Clone" via the word "amazon", without requiring the full name.
    Short words (≤3 chars) are skipped to avoid false positives on "the", "app" etc.
    If the question also contains a project-page keyword (e.g. "workflow"), the
    destination is included so the frontend can navigate directly to that page.
    """
    q = question.lower()
    has_intent = (any(_phrase_present(kw, q) for kw in _SWITCH_KEYWORDS)
                  or any(_phrase_present(kw, q) for kw in _NAV_KEYWORDS))
    if not has_intent:
        return None

    for pid, pname in project_names.items():
        if not pname:
            continue
        name_words = [w for w in pname.lower().split() if len(w) > 3]
        full_name = pname.lower()
        if _phrase_present(full_name, q) or (name_words and any(_phrase_present(w, q) for w in name_words)):
            action: dict = {"type": "switch_project", "project_id": pid}
            # Include destination if a project-page keyword is also present
            for keyword, dest in _PROJECT_DEST_MAP.items():
                if _nav_match(keyword, q):
                    action["destination"] = dest
                    break
            return action

    return None


_NAV_KEYWORDS = {"take me to", "open", "go to", "show me the", "navigate to", "show me my", "i want to see"}

# Global pages — no project_id needed
_GLOBAL_DEST_MAP = {
    "dashboard": "dashboard",
    "home": "dashboard",
    "my projects": "projects",
    "all projects": "projects",
    "projects page": "projects",
    "todo": "todo",
    "to do": "todo",
    "my tasks": "todo",
    "calendar": "calendar",
    "schedule": "calendar",
    "archive": "archive",
    "archived": "archive",
    "profile": "profile",
    "my profile": "profile",
    "account": "profile",
    "settings": "settings",
    "workspaces": "workspaces",
    "integrations": "workspaces",
    "connected platforms": "workspaces",
    "connections": "workspaces",
    "help": "help",
}

# Project-specific pages — need project_id
_PROJECT_DEST_MAP = {
    "workflow": "workflow",
    "kanban": "workflow",
    "board": "workflow",
    "graph": "workflow",
    "flow": "workflow",
    "task list": "tasks",
    "team": "team",
    "members": "team",
    "activity": "activity",
    "events": "activity",
    "event log": "activity",
    "blueprint": "blueprint",
    "modify": "blueprint",
}


_FUZZY_STOP = {"the", "a", "an", "to", "me", "my", "i", "and", "or", "of",
               "for", "is", "in", "on", "take", "show", "open", "go", "want"}


def _fuzzy_project_match(question: str, project_names: dict[str, str]) -> tuple[str, str] | None:
    """Return (project_id, project_name) if a question word is a substring of a
    project name — catches CamelCase names like 'PantryPal' when user says 'pantry'."""
    q_words = [w for w in re.findall(r'\b\w+\b', question.lower())
               if len(w) > 3 and w not in _FUZZY_STOP]
    for pid, pname in project_names.items():
        if not pname:
            continue
        pname_lower = pname.lower()
        for word in q_words:
            if word in pname_lower and word != pname_lower:
                return pid, pname
    return None


def _find_matched_word(question: str, project_name: str) -> str:
    """Return the question word that partially matched the project name."""
    pname_lower = project_name.lower()
    for word in re.findall(r'\b\w+\b', question.lower()):
        if len(word) > 3 and word not in _FUZZY_STOP and word in pname_lower and word != pname_lower:
            return word
    return question


def _phrase_present(phrase: str, q: str) -> bool:
    """True if `phrase` appears in q as whole words (word-boundary match).

    Avoids the substring false positives the old `in` checks produced — "open"
    matching "reopened", "team" matching "esteem", "flow" matching "workflow",
    "board" matching "dashboard".
    """
    phrase = phrase.strip()
    if not phrase:
        return False
    return re.search(rf"\b{re.escape(phrase)}\b", q) is not None


def _nav_match(keyword: str, q: str) -> bool:
    """Return True if keyword or its singular/plural form appears in q as whole words."""
    if _phrase_present(keyword, q):
        return True
    if keyword.endswith("s") and _phrase_present(keyword[:-1], q):
        return True
    if not keyword.endswith("s") and _phrase_present(keyword + "s", q):
        return True
    return False


def _detect_navigation(
    question: str,
    project_id: str | None,
    project_names: dict[str, str],
) -> dict | None:
    """Return an action dict if the question is a navigation request, else None."""
    q = question.lower()
    if not any(_phrase_present(kw, q) for kw in _NAV_KEYWORDS):
        return None

    # Global pages first — no project_id needed
    for keyword, dest in _GLOBAL_DEST_MAP.items():
        if _nav_match(keyword, q):
            return {"type": "navigate", "destination": dest}

    # Project-specific pages
    destination = None
    for keyword, dest in _PROJECT_DEST_MAP.items():
        if _nav_match(keyword, q):
            destination = dest
            break
    if not destination:
        return None

    # Fuzzy-match project name from question, fall back to current project_id
    target_project_id = project_id
    for pid, pname in project_names.items():
        if pname and pname.lower() in q:
            target_project_id = pid
            break

    return {"type": "navigate", "destination": destination, "project_id": target_project_id}


_GRAPH_KEYWORDS = {"block", "depend", "skill", "assign", "who is", "who can", "owner", "relationship", "working on", "work on", "assigned to", "responsible"}
_EVENT_KEYWORDS = {"recent", "today", "yesterday", "commit", "push", "discord", "did", "doing", "worked", "activity", "update", "lately", "last week", "this week"}
_CAPACITY_KEYWORDS = {
    "how much can i", "how many can i", "realistically finish", "realistically complete",
    "get done today", "finish today", "finish this week", "accomplish today",
    "complete today", "my workload", "my capacity", "how productive",
    "how busy am i", "what can i finish", "what can i get done", "what can i realistically",
}


def _needs_graph(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _GRAPH_KEYWORDS)


def _needs_events(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _EVENT_KEYWORDS)


def _needs_capacity_planning(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _CAPACITY_KEYWORDS)


# Answers a question, running the three independent retrievals concurrently.
def answer_question(
    question: str,
    api_key: str,
    conversation_history: list[dict] = None,
    project_id: str | None = None,
) -> str:
    """End-to-end Clover answer with the retrieval steps parallelised.

    The semantic search, the graph build, and the live-events fetch don't depend
    on each other — only the final Gemini synthesis needs all three. Running them
    in a thread pool (they're all I/O-bound) collapses their latency to the slowest
    single step instead of their sum, then ask_clover does the one LLM call.

    Graph and events are only fetched when the question actually needs them —
    skipping an unnecessary HTTP call or Neo4j query saves real time.
    """
    want_graph = _needs_graph(question)
    want_events = _needs_events(question)
    # If neither keyword set matched, fetch both — better to have too much context
    # than too little for an ambiguous question.
    if not want_graph and not want_events:
        want_graph = want_events = True

    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks_future = pool.submit(search_top_tasks, question, api_key, project_id)
        graph_future = pool.submit(fetch_graph) if want_graph else None
        events_future = pool.submit(_safe_fetch_live_events) if want_events else None
        names_future = pool.submit(fetch_project_names)

        relevant_tasks = tasks_future.result()
        graph = graph_future.result() if graph_future else None
        live_events = events_future.result() if events_future else None
        project_names = names_future.result()

    return ask_clover(
        question,
        relevant_tasks,
        api_key,
        conversation_history,
        project_id=project_id,
        graph=graph,
        live_events=live_events,
        project_names=project_names,
    )


# Runs Clover from the command line: asks a question and prints the answer.
def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Ask Clover a project question: ").strip()

    if not question:
        raise RuntimeError("Question cannot be empty.")

    conversation_history: list[dict] = []
    answer = answer_question(question, api_key, conversation_history)

    print(f"\nQuestion: {question}\n")
    print("Clover:")
    print(answer)

    conversation_history.append({"question": question, "answer": answer})
    conversation_history = conversation_history[-5:]


if __name__ == "__main__":
    main()
