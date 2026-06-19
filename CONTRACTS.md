# Orchestra — Contracts & API Reference

*Single source of truth for all three repos.*

## Task Data Shape

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

All AI Server endpoints require header `x-api-key: orchestra-secret-2026`

## Live URLs

| Service | URL |
|---------|-----|
| AI Server | https://orchestra-ai-36zm.onrender.com |
| Backend | https://orchestra-backend-2v5a.onrender.com |
