# Nexus

Nexus is an AI Operating System for knowledge work.

The long-term goal is to help a user turn goals into structured work: planning, research, writing, memory, scheduling, and tool execution.

The current implementation is Phase 3:

```text
User Goal -> Planner Agent -> Human Approval -> Memory Retrieval -> Parallel Research Agents -> Writer Agent -> Memory Storage -> Final Markdown Output
```

## Current Status

Phase 1 created the first working vertical slice. Phase 2 added human approval, web research, scraping, parallel agent execution, and durable LangGraph checkpointing. Phase 3 adds persistent project memory with pgvector and OpenAI embeddings.

Nexus can currently:

- Accept a user goal from the Streamlit UI or FastAPI API.
- Create a project and workflow run in PostgreSQL.
- Ask a planner agent to create a structured research plan.
- Pause for human approval before research starts.
- Resume or reject the graph from the approval step.
- Retrieve relevant project memory before research starts.
- Run research subtasks concurrently.
- Search the web with Tavily.
- Scrape source pages with Playwright.
- Fall back to aiohttp and BeautifulSoup when browser scraping fails.
- Score research result relevance with the configured OpenRouter model.
- Ask a writer agent to turn the plan and research into a final markdown report.
- Store research chunks and final-output summaries as vector memory.
- Search project memory from the UI or API.
- Save workflow state and agent messages.
- Let the frontend poll workflow status and render progress.

Detailed phase notes are kept in:

- `docs/phases/phase-1-foundation.md`
- `docs/phases/phase-2-planner-research-approval.md`
- `docs/phases/phase-3-persistent-memory.md`

Every major phase or code-flow change should be recorded in `docs/phases/`.

## Architecture Flow

```text
Streamlit Frontend
    |
    | POST /workflows
    v
FastAPI Backend
    |
    | create project + workflow_run
    v
PostgreSQL
    |
    | invoke graph until approval interrupt
    v
LangGraph Workflow
    |
    | planner node
    v
PlannerAgent
    |
    | OpenRouter chat completion
    v
WorkflowPlan
    |
    | interrupt for human approval
    v
Human Approval
    |
    | POST /workflows/{run_id}/approve
    v
Memory Retrieval
    |
    | pgvector search + model reranking
    v
Parallel Research
    |
    | Tavily search + Playwright scrape + relevance scoring
    v
ResearchResult[]
    |
    | writer node
    v
WriterAgent
    |
    | OpenRouter chat completion
    v
Final Markdown Output
    |
    | store memory + update workflow_run state
    v
PostgreSQL
    |
    | GET /workflows/{run_id}/status
    v
Streamlit Frontend
```

## Tech Stack

- Python 3.11+
- FastAPI for the async backend API
- LangGraph for graph-based agent orchestration
- LangGraph PostgresSaver for durable graph checkpointing
- OpenRouter for model access through an OpenAI-compatible chat completions API
- OpenAI embeddings with `text-embedding-3-small`
- pgvector for vector memory
- tiktoken for token-aware chunking
- Tavily for web search
- Playwright for browser-based scraping
- aiohttp for scrape fallback
- BeautifulSoup and lxml for content extraction
- PostgreSQL for workflow persistence
- asyncpg for async database access
- Pydantic v2 for schemas and settings
- pydantic-settings for environment configuration
- LangSmith tracing decorators on agent methods
- Streamlit for the UI
- Docker Compose for local services
- Pytest for tests

## Project Structure

```text
backend/
  main.py                         FastAPI application entrypoint
  config.py                       Environment settings
  agents/
    base.py                       BaseAgent and OpenRouter model call helper
    planner.py                    PlannerAgent
    memory_agent.py               MemoryAgent
    research.py                   ResearchAgent
    writer.py                     WriterAgent
  api/routes/
    workflows.py                  Workflow API routes
  db/
    checkpointer.py               LangGraph PostgresSaver setup
    connection.py                 asyncpg pool and query helpers
    migrations/001_initial.sql    Initial database schema
    migrations/002_phase2_statuses.sql
    migrations/002_vector_memory.sql
  memory/
    chunker.py                    Token-aware chunking
    embeddings.py                 OpenAI embedding helpers
    store.py                      pgvector memory store
  graphs/
    research_graph.py             LangGraph workflow definition
  observability/
    token_tracker.py              Token usage logging
  schemas/
    api.py                        API request/response models
    workflow.py                   Workflow state and agent models
  tests/
    test_phase1.py
    test_phase2.py

frontend/
  app.py                          Streamlit app

docs/phases/
  phase-1-foundation.md
  phase-2-planner-research-approval.md
  phase-3-persistent-memory.md

docker-compose.yml
requirements.txt
.env.example
main.py
```

## Backend API

### Health Check

```http
GET /health
```

### Create Workflow

```http
POST /workflows
```

Creates the project and workflow run, invokes the graph until the approval interrupt, and returns the generated plan.

Request body:

```json
{
  "project_name": "Research Brief",
  "goal": "Write a concise brief about using pgvector for semantic search."
}
```

Response:

```json
{
  "run_id": "uuid",
  "status": "awaiting_approval",
  "plan": {
    "title": "Plan title",
    "goal": "User goal",
    "subtasks": []
  },
  "output": null
}
```

### Approve Workflow

```http
POST /workflows/{run_id}/approve
```

Resumes the graph from the human approval interrupt and starts research in the background.

### Reject Workflow

```http
POST /workflows/{run_id}/reject
```

Resumes the graph with rejection and marks the workflow as rejected.

### Get Workflow Status

```http
GET /workflows/{run_id}/status
```

Returns status, plan, research results, final output, and serialized workflow state.

### Projects

```http
GET /projects
POST /projects
GET /projects/{project_id}/runs
GET /projects/{project_id}/memory?query=...
```

Project memory search returns the top relevant memory chunks.

## Database Schema

The app stores product workflow data in:

- `projects`
- `workflow_runs`
- `agent_messages`
- `memory_chunks`
- `project_summaries`

LangGraph checkpoint tables are created by `PostgresSaver` during startup.

## Agent Flow

### BaseAgent

Shared base class for model-backed agents.

It handles:

- Model name and system prompt setup.
- Async OpenRouter calls.
- LangSmith tracing.
- Token usage logging.

### PlannerAgent

Input:

- User goal.

Output:

- `WorkflowPlan` with title, goal, and prioritized `ResearchTask` items.

### ResearchAgent

Input:

- One `ResearchTask`.

Output:

- Sorted `ResearchResult` records.

It searches Tavily, scrapes top URLs, deduplicates by URL, and scores source relevance.

### MemoryAgent

Input:

- Project ID.
- Query or workflow goal.

Output:

- Formatted relevant memory context.

It retrieves project memory, reranks chunks, and stores summaries after workflow completion.

### WriterAgent

Input:

- User goal.
- Approved workflow plan.
- Research results.

Output:

- Final markdown report.

## LangGraph Workflow

Defined in:

```text
backend/graphs/research_graph.py
```

Current graph:

```text
START -> planner -> human_approval -> memory_retrieval -> parallel_research -> writer -> memory_storage -> END
```

Rejected workflows route from `human_approval` directly to `END`.

## Environment Variables

Create a `.env` file from `.env.example`.

For running the backend locally with Postgres exposed on `localhost:5432`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-sonnet-4
OPENROUTER_APP_NAME=Nexus
OPENROUTER_SITE_URL=http://localhost:8501
OPENAI_API_KEY=your_openai_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TAVILY_API_KEY=your_tavily_key_here

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nexus
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=nexus-phase3
ENVIRONMENT=development
API_BASE_URL=http://localhost:8000
```

When running the backend inside Docker Compose, use the Docker service hostname:

```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/nexus
```

## Local Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install Playwright browser binaries:

```powershell
python -m playwright install chromium
```

Start Postgres:

```powershell
docker compose up -d postgres
```

Start the backend:

```powershell
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```powershell
streamlit run frontend/app.py
```

Open:

```text
Backend:  http://localhost:8000
Health:   http://localhost:8000/health
Frontend: http://localhost:8501
```

## Docker Compose

Run all services:

```powershell
docker compose up
```

Services:

- `postgres`: PostgreSQL with pgvector image
- `app`: FastAPI backend
- `streamlit`: Streamlit frontend

## Tests

Run Phase 1 compatibility tests:

```powershell
python -m pytest backend/tests/test_phase1.py -q
```

Run Phase 2 tests:

```powershell
python -m pytest backend/tests/test_phase2.py -q
```

Run Phase 3 tests:

```powershell
python -m pytest backend/tests/test_phase3.py -q
```

Tests mock model, embedding, Tavily, and database boundaries where appropriate.

## Documentation Rule

For every major change in project behavior, architecture, workflow, or code flow:

1. Update this README if the change affects how the project works or runs.
2. Add or update a phase document under `docs/phases/`.
3. Keep the architecture flow current.
4. Keep setup commands and environment variables current.
