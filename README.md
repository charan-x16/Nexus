# Nexus

Nexus is an AI Operating System for knowledge work.

The long-term goal is to help a user turn goals into structured work: planning, research, writing, memory, scheduling, and tool execution. The current implementation is Phase 1, which focuses on the smallest useful vertical slice:

```text
User Goal -> Planner Agent -> Writer Agent -> Final Markdown Output
```

Phase 1 gives the project a working backend, frontend, database layer, agent layer, LangGraph workflow, and tests.

## Current Status

Phase 1 is complete.

Nexus can currently:

- Accept a user goal from the Streamlit UI or FastAPI API.
- Create a project and workflow run in PostgreSQL.
- Run a LangGraph workflow.
- Ask a planner agent to convert the goal into a structured task plan.
- Ask a writer agent to turn the plan into a final markdown report.
- Save workflow state and agent messages.
- Let the frontend poll workflow status and render the final output.

Detailed phase notes are kept in:

- `docs/phases/phase-1-foundation.md`

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
    | background task starts graph
    v
LangGraph Workflow
    |
    | planner node
    v
PlannerAgent
    |
    | OpenRouter chat completion
    v
TaskPlan
    |
    | writer node
    v
WriterAgent
    |
    | OpenRouter chat completion
    v
Final Markdown Output
    |
    | update workflow_run state
    v
PostgreSQL
    |
    | GET /workflows/{run_id}
    v
Streamlit Frontend
```

## Tech Stack

- Python 3.11+
- FastAPI for the async backend API
- LangGraph for graph-based agent orchestration
- OpenRouter for model access through an OpenAI-compatible chat completions API
- PostgreSQL for workflow persistence
- `asyncpg` for async database access
- Pydantic v2 for schemas and settings
- `pydantic-settings` for environment configuration
- LangSmith tracing decorators on agent methods
- Streamlit for the first UI
- Docker Compose for local Postgres, backend, and frontend services
- Pytest for tests

## Project Structure

```text
backend/
  main.py                       FastAPI application entrypoint
  config.py                     Environment settings
  agents/
    base.py                     BaseAgent and OpenRouter model call helper
    planner.py                  PlannerAgent
    writer.py                   WriterAgent
  api/routes/
    workflows.py                Workflow API routes
  db/
    connection.py               asyncpg pool and query helpers
    migrations/001_initial.sql  Initial database schema
  graphs/
    research_graph.py           LangGraph workflow definition
  observability/
    token_tracker.py            Token usage logging
  schemas/
    api.py                      API request/response models
    workflow.py                 Workflow state and agent models
  tests/
    test_phase1.py              Phase 1 tests

frontend/
  app.py                        Streamlit app

docs/phases/
  phase-1-foundation.md         Phase 1 implementation log

docker-compose.yml              Local services
requirements.txt                Pinned Python dependencies
.env.example                    Example environment variables
main.py                         Convenience backend launcher
```

## Backend API

### Health Check

```http
GET /health
```

Returns backend health and environment.

### Create Workflow

```http
POST /workflows
```

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
  "status": "queued",
  "output": null
}
```

### Get Workflow Status

```http
GET /workflows/{run_id}
```

Returns the workflow status, serialized state, and final markdown output once complete.

## Database Schema

The initial migration creates:

- `projects`
  - Stores project name and goal.
- `workflow_runs`
  - Stores run status and serialized workflow state.
- `agent_messages`
  - Stores messages produced by agents during a workflow run.

Migration file:

```text
backend/db/migrations/001_initial.sql
```

## Agent Flow

### BaseAgent

`BaseAgent` is the shared base class for all agents.

It handles:

- Model name and system prompt setup.
- Async OpenRouter calls.
- LangSmith tracing.
- Token usage logging.

### PlannerAgent

Input:

- User goal.

Output:

- `TaskPlan` with title, description, subtasks, and estimated steps.

### WriterAgent

Input:

- User goal.
- Planner output.
- Available research context.

Output:

- Final markdown report.

## LangGraph Workflow

The Phase 1 graph is defined in:

```text
backend/graphs/research_graph.py
```

Current graph:

```text
START -> planner -> writer -> END
```

The graph is compiled with `MemorySaver` as the checkpointer for now.

## Environment Variables

Create a `.env` file from `.env.example`.

For running backend locally with Postgres exposed on `localhost:5432`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-sonnet-4
OPENROUTER_APP_NAME=Nexus
OPENROUTER_SITE_URL=http://localhost:8501

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nexus
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=nexus-phase1
ENVIRONMENT=development
API_BASE_URL=http://localhost:8000
```

When running the backend inside Docker Compose, the app uses the Docker service hostname:

```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/nexus
```

## Local Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start Postgres with Docker:

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

Run Phase 1 tests:

```powershell
python -m pytest backend/tests/test_phase1.py -q
```

The Phase 1 tests mock model calls, so they do not require a real OpenRouter API call.

## Documentation Rule

For every major change in project behavior, architecture, workflow, or code flow:

1. Update this README if the change affects how the project works or runs.
2. Add or update a phase document under `docs/phases/`.
3. Keep the architecture flow current.
4. Keep setup commands and environment variables current.

This keeps the project understandable as Nexus grows from a Phase 1 vertical slice into a larger AI operating system.
