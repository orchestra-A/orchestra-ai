# Orchestra — Contracts & API Reference

*Single source of truth for all three repos.*

## Task Data Shape

**Produced by:** `blueprint.py`, `assign.py`  
**Consumed by:** `assign.py`, `skill_gap.py`, `dependency_query.py`, `search.py`, `commit_intel.py`

Each task in the AI-generated roadmap follows this structure:

```json
{
  "id": "T1",
  "title": "string",
  "track": "string",
  "description": "string",
  "status": "todo | in_progress | blocked | completed",
  "assigned_to": "string",
  "dependencies": [],
  "priority": "high | medium | low",
  "project_id": "string",
  "created_at": "2026-05-28T10:00:00Z",
  "updated_at": "2026-05-28T10:00:00Z",
  "platform": "string"
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique task identifier (e.g. `T1`, `T2`) |
| `title` | string | Short, human-readable task name |
| `description` | string | Detailed explanation of what the task involves |
| `status` | string | Current task status. One of: `todo`, `in_progress`, `completed`, `blocked` |
| `assigned_to` | string | Name of the team member assigned to this task (added by `assign.py`) |
| `platform` | string | Target platform or surface area for the task (e.g. `web`, `mobile`, `backend`) |
| `priority` | string | Task priority level (e.g. `low`, `medium`, `high`) |
| `project_id` | string | Identifier linking the task to its parent project |
| `created_at` | string | ISO 8601 datetime when the task was created |
| `updated_at` | string | ISO 8601 datetime when the task was last updated |

### Full Document Shape

Both scripts output a top-level JSON object:

```json
{
  "project_name": "string",
  "summary": "string",
  "tasks": [ ... ]
}
```

- `blueprint.py` writes to `blueprint.json` (tasks without `assigned_to`)
- `assign.py` writes to `assigned.json` (tasks with `assigned_to` populated)
- `summary` is a short, plain-English explanation of how the work was broken down. It is produced by `blueprint.py` and surfaced in the `POST /blueprint` response. Because `assign.py` regenerates the JSON, the API re-attaches `summary` after assignment — consumers should treat it as optional and render it if present.

---

## 2. Normalized Event Format

**Produced by:** `normalizer.py` (Arnav)  
**Consumed by:** `commit_intel.py`

All platform events (GitHub, Slack, etc.) are normalized into this single format before downstream processing:

```json
{
  "platform": "github",
  "actor": "arnav",
  "event_type": "pushed code",
  "timestamp": "2026-05-28T10:00:00",
  "action_summary": "Fixed login bug in auth.py"
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `platform` | string | Source platform (e.g. `github`, `slack`, `jira`) |
| `actor` | string | Team member who performed the action (lowercase) |
| `event_type` | string | What happened (e.g. `pushed code`, `opened PR`, `merged PR`) |
| `timestamp` | string | ISO 8601 datetime of the event |
| `action_summary` | string | Human-readable description — typically the commit message or event summary |

---

## 3. Neo4j Node Format

**Produced by:** Naman's ingestion pipeline  
**Returned by:** `query.py`

Tasks stored in the Neo4j knowledge graph use this node structure:

```json
{
  "node_type": "Task",
  "id": "T1",
  "title": "string",
  "track": "string",
  "assigned_to": "member_name",
  "status": "todo | in_progress | completed | blocked"
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `node_type` | string | Always `"Task"` for task nodes |
| `id` | string | Task identifier — must match the AI Output Format `id` |
| `title` | string | Task title — must match the AI Output Format `title` |
| `track` | string | Work category — must match the AI Output Format `track` |
| `assigned_to` | string | Assigned team member — must match the AI Output Format `assigned_to` |
| `status` | string | Current task status. One of: `todo`, `in_progress`, `completed`, `blocked` |

### Relationship Types

Task dependency relationships in Neo4j:

```
(Task A)-[:DEPENDS_ON]->(Task B)
```

Meaning: Task A cannot start until Task B is complete.

---

## Frontend API Routing

| Action | Server | Endpoint |
|--------|--------|----------|
| Generate roadmap | AI | `POST /blueprint` |
| Get all tasks | AI | `GET /tasks` |
| Get graph data | AI | `GET /graph` |
| Ask Clover | AI | `POST /clover` |
| Search tasks | AI | `GET /search` |
| Update task status | AI | `PATCH /tasks/{id}/status` |
| Onboard developer | AI | `POST /onboarding` |
| Get team skills | AI | `GET /team` |
| Login + Signup | Backend | auth endpoints |
| Real-time updates | Backend | `WS /ws` |
| View live events | Backend | `GET /events` |

## Database Ownership

- **Neo4j** — tasks, people, skills, relationships — owned by AI Server
- **ChromaDB** — vector embeddings for semantic search — owned by AI Server
- **PostgreSQL** — user accounts, login sessions, OAuth tokens — owned by Backend

## Authentication

All AI Server endpoints require the x-api-key header (value shared privately with the team)

## Live URLs

| Service | URL |
|---------|-----|
| AI Server | https://orchestra-ai-36zm.onrender.com |
| Backend | https://orchestra-backend-2v5a.onrender.com |

## 4. Manual Skills Endpoint

**Provided by:** `POST /team/manual` (`main.py` → `graph_query.merge_developer_skills`)
**Consumed by:** the frontend skills form (manual fallback for `onboarding.py`)

A human override for adding or correcting a developer's skills directly, without
depending on GitHub-based AI inference. Requires the `x-api-key` header.

**Request body:**

```json
{
  "name": "Naman",
  "skills": ["Python", "Neo4j", "FastAPI"]
}
```

**Response** — the developer's full skill set after the merge:

```json
{
  "name": "Naman",
  "skills": ["FastAPI", "Neo4j", "Python"]
}
```

### Behavior — merge, never overwrite

- If the `Developer` node exists, the new skills are **added** to its existing
  `HAS_SKILL` set; current skills are never removed.
- If no `Developer` node exists, one is created with the given skills.
- Skills are stored as `Skill` nodes joined by `(Developer)-[:HAS_SKILL]->(Skill)`
  — the same representation `onboarding.py` and `ingest.py` use, so manually-added
  skills are visible to `/team`, `/blueprint`, `/assign`, and `/graph`. `MERGE`
  makes the write additive and duplicate-free (no APOC required).
- Validation (no AI step to catch bad input): empty `name` or empty `skills`
  returns `400`.

---

## 5. Golden Rule

> **If you change any data format defined in this document, you must update this file first and notify the team in Discord before merging.**

This applies to:

- Adding, removing, or renaming fields
- Changing field types or allowed values
- Changing file names or output paths
- Modifying relationship types in Neo4j

Failure to follow this rule will break downstream scripts and integrations that depend on these contracts.

---

*Last updated: June 2026 — Orchestra + Clover Team*
