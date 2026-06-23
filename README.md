# Orchestra + Clover

Orchestra is an AI-powered project manager that turns a raw app idea into a dependency-locked engineering roadmap, auto-assigns tasks from team skill profiles, and tracks progress across GitHub and Discord. Clover is the conversational assistant that answers questions like "what did Arnav push last night?" using RAG search and a Neo4j knowledge graph.

## Team

| Name | Role | GitHub |
|------|------|--------|
| Mitaali Singh | Lead · PM · AI Developer | |
| Naman Gupta | Knowledge Graph Engineer | @Naman-GG |
| Arnav Tripathi | Infrastructure Engineer | @ArnavXT |
| Sarvagya Prakash | Data Pipeline Engineer | @SarvagyaPrakash |
| Prince Negi | Interactive Canvas Specialist | |
| Isha Mahadev | Interface Developer | @IshaMahadev |

## Architecture

- **Frontend → Backend: `GET /tasks`** — `https://orchestra-backend-2v5a.onrender.com/tasks`
- **Frontend → Backend: WebSocket `/ws`** — `wss://orchestra-backend-2v5a.onrender.com/ws`
- **Frontend → AI Server: `GET /graph`** — `https://orchestra-ai-36zm.onrender.com/graph`
- **Frontend → AI Server: `POST /clover`** — `https://orchestra-ai-36zm.onrender.com/clover`
- **Backend → AI Server: `PATCH /tasks/{id}/status`** — `https://orchestra-ai-36zm.onrender.com/tasks/{id}/status`
- **AI Server → Backend: `GET /events`** — `https://orchestra-backend-2v5a.onrender.com/events`

## Live URLs

| Service | URL |
|---------|-----|
| AI Server | https://orchestra-ai-36zm.onrender.com |
| Backend | https://orchestra-backend-2v5a.onrender.com |
| Docs | https://orchestra-ai-36zm.onrender.com/docs |

## Repos

| Repo | Description |
|------|-------------|
| `orchestra-ai` | AI server — roadmap generation, task assignment, Clover assistant, knowledge graph, and semantic search |
| `orchestra-backend` | Backend API — event ingestion from GitHub and Discord, authentication, and real-time WebSocket feed |
| `orchestra-frontend` | Web application — task dashboard, interactive graph canvas, and Clover chat interface |

## Stack

Python · Gemini API · ChromaDB · Neo4j · FastAPI · PostgreSQL · React · ReactFlow
