# Phase 2: Planner Approval and Research

## Goal

Build a production-quality planning and research workflow:

```text
User Goal -> Planner -> Human Approval -> Parallel Research -> Writer -> Final Output
```

## What Changed

### Planner

- Rewrote `PlannerAgent`.
- Planner now returns a structured `WorkflowPlan`.
- `WorkflowPlan` contains `title`, `goal`, and `subtasks`.
- Each subtask is a `ResearchTask` with `id`, `description`, `search_queries`, `priority`, and `status`.
- Planner output is parsed with Pydantic.
- Planner retries once when model JSON cannot be parsed.

### Human Approval

- Added a LangGraph human approval interrupt.
- Workflow now pauses after planning.
- User can approve or reject the plan from the UI/API.
- Approving resumes the graph.
- Rejecting marks the workflow as rejected.

### Research

- Added `ResearchAgent`.
- Research uses Tavily for web search.
- Research scrapes source pages with Playwright.
- If Playwright fails, scraping falls back to `aiohttp` and BeautifulSoup.
- Pages are cleaned by removing navigation, headers, footers, scripts, forms, ads, and other low-value page elements.
- Scraped content is truncated to 4000 characters.
- Tavily rate limits and temporary failures use exponential backoff.
- Empty search and scrape results are handled gracefully.
- Results are deduplicated by URL.
- The model scores relevance from 1 to 10.

### Parallel Execution

- Added `parallel_research_node`.
- One `ResearchAgent` runs per research subtask.
- Research tasks run concurrently with `asyncio.gather()`.
- Research results are collected and sorted by relevance score.
- Intermediate research progress is persisted to workflow state when each subtask completes.

### Graph

- Replaced Phase 1 graph with:

```text
START -> planner -> human_approval -> parallel_research -> writer -> END
```

- Rejected workflows route from approval to `END`.
- Added Postgres-backed LangGraph checkpointing through `PostgresSaver`.
- Graph compilation is lazy so the app can set up the checkpointer during FastAPI startup.

### API

- `POST /workflows` creates a workflow, runs planning, pauses at approval, and returns the plan.
- `POST /workflows/{run_id}/approve` resumes graph execution.
- `POST /workflows/{run_id}/reject` rejects the workflow.
- `GET /workflows/{run_id}/status` returns status, plan, research results, final output, and state.

### UI

- Streamlit now shows the generated plan before research runs.
- Subtasks render as expandable sections.
- Added approve and reject buttons.
- Approved workflows poll for research progress.
- Research results appear as the backend persists them.
- Final markdown output renders after completion.

## Important Files

- `backend/agents/planner.py`
- `backend/agents/research.py`
- `backend/agents/writer.py`
- `backend/graphs/research_graph.py`
- `backend/db/checkpointer.py`
- `backend/api/routes/workflows.py`
- `backend/schemas/workflow.py`
- `frontend/app.py`
- `backend/tests/test_phase2.py`

## Phase 2 Status

Phase 2 adds plan approval, parallel web research, source scraping, relevance scoring, and durable graph checkpointing.
