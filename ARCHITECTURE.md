# Orchestra AI — System Architecture

## 1 — Overview

| Repo | Owner | Deployed At |
|------|-------|-------------|
| `orchestra-frontend` | Prince, Isha | — |
| `orchestra-backend` | Arnav, Prakash | `https://orchestra-backend-2v5a.onrender.com` |
| `orchestra-ai` | Mitaali, Naman | `https://orchestra-ai-36zm.onrender.com` |

## 2 — System Diagram

```
                    ┌─────────────────────┐
                    │  orchestra-frontend │
                    └──────┬───────┬──────┘
           GET /tasks,    │       │    GET /graph,
           WebSocket /ws  │       │    POST /clover
                          ▼       ▼
            ┌─────────────────┐   ┌─────────────────┐
            │ orchestra-backend│   │  orchestra-ai   │
            └────────┬─────────┘   └────────┬────────┘
                     │ PATCH /tasks/{id}/status
                     │ (on PR merge)          │
                     └───────────────────────►│
                                              │
                     GET /events              │
                     ◄─────────────────────────┘
```

## 3 — Connection Details

1. **Frontend → Backend: `GET /tasks`** — Frontend fetches the task list from the backend API for display in the UI.
2. **Frontend → Backend: WebSocket `/ws`** — Frontend maintains a live connection to receive real-time Discord and GitHub events.
3. **Frontend → AI Server: `GET /graph`** — Frontend fetches the knowledge graph (nodes and edges) from the AI server for visualization.
4. **Frontend → AI Server: `POST /clover`** — Frontend sends natural-language questions to Clover, the conversational project assistant.
5. **Backend → AI Server: `PATCH /tasks/{id}/status`** — When a PR is merged, the backend updates the corresponding task status in the AI server's Neo4j graph.
6. **AI Server → Backend: `GET /events`** — The AI server pulls live activity events from the backend to enrich Clover and commit-intelligence context.

## 4 — Databases

| Database | Location | Purpose |
|----------|----------|---------|
| Neo4j | AI server | Knowledge graph (tasks, developers, skills, dependencies) |
| ChromaDB | AI server | Vector search for semantic task and commit retrieval |
| PostgreSQL (Neon) | Backend | Event storage and authentication |

---

*Orchestra AI Team — VIT Bhopal 2027*
