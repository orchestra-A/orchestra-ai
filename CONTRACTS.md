# Orchestra + Clover — Data Contracts

This document defines the shared data formats used across the Orchestra + Clover system. All services, scripts, and integrations must conform to these contracts to ensure interoperability between team components.

---

## 1. AI Output Format

**Produced by:** `blueprint.py`, `assign.py`  
**Consumed by:** `assign.py`, `skill_gap.py`, `dependency_query.py`, `search.py`, `commit_intel.py`

Each task in the AI-generated roadmap follows this structure:

```json
{
  "id": "T1",
  "title": "string",
  "description": "string",
  "status": "todo",
  "assigned_to": "member_name",
  "platform": "string",
  "priority": "string",
  "project_id": "string",
  "created_at": "2026-05-28T10:00:00",
  "updated_at": "2026-05-28T10:00:00"
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
  "tasks": [ ... ]
}
```

- `blueprint.py` writes to `blueprint.json` (tasks without `assigned_to`)
- `assign.py` writes to `assigned.json` (tasks with `assigned_to` populated)

---

## 2. Normalized Event Format

**Produced by:** `normalizer.py` (Arnav)  
**Consumed by:** `commit_intel.py`

All platform events (GitHub, Slack, etc.) are normalized into this single format before downstream processing:

```json
{
  "platform": "github",
  "member": "arnav",
  "action": "pushed code",
  "timestamp": "2026-05-28T10:00:00",
  "details": "Fixed login bug in auth.py"
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `platform` | string | Source platform (e.g. `github`, `slack`, `jira`) |
| `member` | string | Team member who performed the action (lowercase) |
| `action` | string | What happened (e.g. `pushed code`, `opened PR`, `merged PR`) |
| `timestamp` | string | ISO 8601 datetime of the event |
| `details` | string | Human-readable description — typically the commit message or event summary |

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
  "status": "pending | in_progress | complete"
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
| `status` | string | Current task status. One of: `pending`, `in_progress`, `complete` |

### Relationship Types

Task dependency relationships in Neo4j:

```
(Task A)-[:DEPENDS_ON]->(Task B)
```

Meaning: Task A cannot start until Task B is complete.

---

## 4. Golden Rule

> **If you change any data format defined in this document, you must update this file first and notify the team in Discord before merging.**

This applies to:

- Adding, removing, or renaming fields
- Changing field types or allowed values
- Changing file names or output paths
- Modifying relationship types in Neo4j

Failure to follow this rule will break downstream scripts and integrations that depend on these contracts.

---

*Last updated: May 2026 — Orchestra + Clover Team*
