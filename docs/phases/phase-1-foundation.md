# Phase 1: Foundation and Vertical Slice

## Goal

Build the first working foundation for Nexus, an AI Operating System for knowledge work, with one complete end-to-end workflow:

User Goal -> Planner Agent -> Writer Agent -> Final Markdown Output

## What Was Implemented

### Backend Foundation

- Created the backend package under `backend/`.
- Added FastAPI application entrypoint in `backend/main.py`.
- Added application settings in `backend/config.py` using `pydantic-settings`.
- Added `.env` loading support.
- Added CORS middleware.
- Added health check endpoint:
  - `GET /health`

### Database Layer

- Added async PostgreSQL connection pooling using `asyncpg`.
- Added DB lifecycle helpers:
  - `init_pool()`
  - `close_pool()`
  - `get_pool()`
  - `execute_query()`
  - `fetch_rows()`
  - `run_migrations()`
- Added initial migration in `backend/db/migrations/001_initial.sql`.
- Created tables:
  - `projects`
  - `workflow_runs`
  - `agent_messages`

### Schemas

- Added workflow schemas in `backend/schemas/workflow.py`.
- Added API schemas in `backend/schemas/api.py`.
- Implemented:
  - `WorkflowState`
  - `TaskPlan`
  - `AgentMessage`
  - `WorkflowCreateRequest`
  - `WorkflowCreateResponse`
  - `WorkflowStatusResponse`

### Agent Layer

- Added abstract `BaseAgent` in `backend/agents/base.py`.
- Integrated OpenRouter using its OpenAI-compatible chat completions API.
- Configured default OpenRouter model:
  - `anthropic/claude-sonnet-4`
- Added LangSmith tracing decorators on agent methods.
- Added token usage logging through `backend/observability/token_tracker.py`.
- Implemented `PlannerAgent`:
  - Takes a user goal.
  - Calls OpenRouter.
  - Parses JSON into `TaskPlan`.
  - Stores the plan in workflow state.
- Implemented `WriterAgent`:
  - Takes the user goal, plan, and available research context.
  - Calls OpenRouter.
  - Produces final markdown output.
  - Stores the final output in workflow state.

### LangGraph Workflow

- Added graph definition in `backend/graphs/research_graph.py`.
- Implemented nodes:
  - `planner`
  - `writer`
- Implemented graph flow:
  - `START -> planner -> writer -> END`
- Compiled graph with `MemorySaver` checkpointer.
- Exported `compiled_graph`.

### Workflow API

- Added workflow routes in `backend/api/routes/workflows.py`.
- Implemented:
  - `POST /workflows`
  - `GET /workflows/{run_id}`
- `POST /workflows`:
  - Accepts `goal` and `project_name`.
  - Creates a project record.
  - Creates a workflow run record.
  - Starts graph execution in a FastAPI background task.
  - Returns the run ID and queued status.
- `GET /workflows/{run_id}`:
  - Returns workflow status.
  - Returns final output when complete.
  - Returns serialized workflow state.

### Frontend

- Created Streamlit frontend under `frontend/`.
- Added `frontend/app.py`.
- Implemented UI for:
  - Entering project name.
  - Entering user goal.
  - Starting a workflow.
  - Polling workflow status every 2 seconds.
  - Rendering final markdown output.

### DevOps and Runtime

- Added `.env.example`.
- Added `requirements.txt` with pinned dependencies.
- Added `docker-compose.yml`.
- Docker Compose services:
  - `postgres` using `ankane/pgvector`
  - `app` running FastAPI
  - `streamlit` running the frontend
- Updated root `main.py` to launch the FastAPI app with Uvicorn.
- Updated `pyproject.toml` with project dependencies and pytest settings.

### Tests

- Added Phase 1 tests in `backend/tests/test_phase1.py`.
- Tests cover:
  - `PlannerAgent` creates a valid `TaskPlan`.
  - `WriterAgent` produces non-empty output.
  - LangGraph runs end-to-end with mocked OpenRouter model calls.

## Verification

The focused Phase 1 test suite passed:

```bash
python -m pytest backend/tests/test_phase1.py -q
```

Result:

```text
3 passed
```

## Important Files

- `backend/main.py`
- `backend/config.py`
- `backend/db/connection.py`
- `backend/db/migrations/001_initial.sql`
- `backend/agents/base.py`
- `backend/agents/planner.py`
- `backend/agents/writer.py`
- `backend/graphs/research_graph.py`
- `backend/api/routes/workflows.py`
- `backend/schemas/workflow.py`
- `backend/schemas/api.py`
- `backend/observability/token_tracker.py`
- `frontend/app.py`
- `backend/tests/test_phase1.py`
- `.env.example`
- `requirements.txt`
- `docker-compose.yml`

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Run the frontend:

```bash
streamlit run frontend/app.py
```

Run with Docker Compose:

```bash
docker compose up
```

## Phase 1 Status

Phase 1 is complete. Nexus now has a working foundation with a real vertical slice from user goal to agent-generated markdown output.
