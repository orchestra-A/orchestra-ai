# Orchestra + Clover

**AI-powered project manager** — turn a raw app idea into a dependency-locked engineering roadmap, auto-assign work to your team, and ask Clover anything about the project in plain English.

---

## What It Does

- **Roadmap generation** — Takes a raw app idea and produces a structured, dependency-aware engineering roadmap with full task metadata.
- **Smart assignment** — Auto-assigns tasks to team members based on skill profiles and task type.
- **Clover AI assistant** — Answers natural language questions about the project using ChromaDB semantic search and a Neo4j knowledge graph for rich project context.

---

## Pipeline Overview

```
App idea → blueprint.py → assign.py → assigned.json
                ↓              ↓
           skills.json    Neo4j graph (ingest)
                ↓              ↓
         search.py / clover.py / commit_intel.py / skill_gap.py / dependency_query.py
                ↓
            main.py (FastAPI)
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `blueprint.py` | App idea → structured JSON roadmap with all task fields (`id`, `title`, `track`, `description`, `dependencies`, `status`, `priority`, `project_id`, timestamps, `platform`) |
| `skills.py` | Collects team member skill profiles interactively and saves to `skills.json` |
| `assign.py` | Auto-assigns tasks to the best-fit team member using Gemini |
| `search.py` | RAG semantic search over tasks via ChromaDB + Gemini embeddings |
| `commit_intel.py` | Links GitHub commits to roadmap tasks automatically and stores enriched events in ChromaDB |
| `skill_gap.py` | Detects missing skills per task and flags gaps in `skill_gap_report.json` |
| `dependency_query.py` | Natural language dependency chain querying with recursive blocker tracing |
| `clover.py` | Conversational AI assistant — RAG retrieval + Gemini answers with task IDs, titles, and assignee names |
| `main.py` | FastAPI server exposing the pipeline as REST endpoints (CORS enabled for frontend) |

Supporting modules: `ingest.py`, `query.py`, and `graph_query.py` handle Neo4j ingestion and graph queries.

---

## API Endpoints

Start the server:

```bash
python main.py
```

Interactive docs: http://localhost:8000/docs

Deployed API: https://orchestra-ai-production.up.railway.app

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns API status and timestamp |
| `GET` | `/team` | Returns team skill profiles from `skills.json` |
| `GET` | `/project` | Returns project summary with task counts by status, person, and track |
| `POST` | `/blueprint` | Generate roadmap from `{"idea": "string"}` |
| `POST` | `/assign` | Assign tasks from `{"tasks": [...], "skills": {...}}` |
| `GET` | `/search` | Top 3 semantic matches — `?question=...` |
| `POST` | `/clover` | Conversational answer — `{"question": "string"}` → `{"answer": "string"}` |
| `GET` | `/standup` | Generates daily standup summary |
| `GET` | `/replan` | Suggests re-planning for blocked tasks |
| `GET` | `/commit-intel` | Fetches live events from Discord and GitHub |
| `POST` | `/onboarding` | Scans GitHub profile and auto-assigns tasks |
| `GET` | `/tasks` | All tasks from the Neo4j graph |
| `GET` | `/graph` | ReactFlow-ready `nodes` + `edges` for the project graph |

### Examples

```bash
curl -X POST http://localhost:8000/blueprint \
  -H "Content-Type: application/json" \
  -d '{"idea": "An AI app that suggests recipes from fridge photos"}'

curl -X POST http://localhost:8000/clover \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is working on frontend tasks?"}'

curl "http://localhost:8000/search?question=authentication%20backend"

curl http://localhost:8000/graph
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| AI | [google-genai](https://pypi.org/project/google-genai/) (`gemini-2.5-flash`, `models/gemini-embedding-001`) |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |
| API | FastAPI + Uvicorn |
| Vector search | ChromaDB |
| Knowledge graph | Neo4j |

---

## Neo4j Graph

The project knowledge graph stores:

- **30** task nodes
- **5** developer nodes
- Relationships: `ASSIGNED_TO`, `DEPENDS_ON`, `HAS_SKILL`

`GET /graph` returns the full graph as ReactFlow-ready JSON for Prince's frontend. `GET /tasks` returns all tasks in the agreed contract shape for downstream integrations.

---

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:

```env
GEMINI_API_KEY=your_api_key_here

# Neo4j (for /tasks, /graph, ingest)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

4. Run individual scripts or the API:

```bash
python blueprint.py "Your app idea here"
python skills.py
python assign.py
python clover.py "What tasks are blocked?"
python main.py
```

Data contracts for all shared formats are documented in [`CONTRACTS.md`](CONTRACTS.md).

---

## Current Status

**Weeks 1–3 complete.** Full AI pipeline is live — roadmap generation, assignment, RAG search, commit intelligence, skill gap analysis, dependency queries, Clover assistant, and FastAPI + Neo4j integration.

---

*Orchestra + Clover Team — 2026*
