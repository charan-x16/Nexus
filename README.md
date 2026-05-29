# Nexus

Nexus is an AI Operating System for knowledge work.

The long-term goal is to help a user turn goals into structured work: planning, research, writing, memory, scheduling, and tool execution.

The current implementation is Phase 5:

```text
User Goal -> Planner Agent -> Human Approval -> Background Workflow Runner -> Memory Retrieval -> Parallel Research Agents -> Critic Agent -> Targeted Research Loop -> Writer Agent -> Memory Storage -> Structured Final Report
```

## Current Status

Phase 1 created the first working vertical slice. Phase 2 added human approval, web research, scraping, parallel agent execution, and durable LangGraph checkpointing. Phase 3 added persistent project memory with pgvector and OpenAI embeddings. Phase 4 added a critic reflection loop and structured reports with inline citations. Phase 5 adds async background execution, scheduled monitoring jobs, token/cost tracking, Prometheus metrics, and a multipage observability dashboard.

Nexus can currently:

- Accept a user goal from the Streamlit UI or FastAPI API.
- Create a project and workflow run in PostgreSQL.
- Ask a planner agent to create a structured research plan.
- Pause for human approval before research starts.
- Resume, reject, or cancel the graph from the approval step.
- Continue approved workflows through an async background queue.
- Retrieve relevant project memory before research starts.
- Run research subtasks concurrently.
- Critique research for contradictions, weak evidence, missing context, and unverified claims.
- Run targeted follow-up research for high-severity critic findings.
- Search the web with Tavily.
- Scrape source pages with Playwright.
- Fall back to aiohttp and BeautifulSoup when browser scraping fails.
- Score research result relevance with the configured OpenRouter model.
- Ask a writer agent to produce a structured final report with inline citations.
- Store research chunks and final-output summaries as vector memory.
- Search project memory from the UI or API.
- Record token usage and estimated model cost per run and agent.
- Estimate workflow cost before research execution starts.
- Schedule monitoring jobs for project topics.
- Create monitoring alerts when new relevant information is found.
- Expose Prometheus metrics for runs, tokens, and cost.
- Fetch final reports as JSON or markdown.
- Save workflow state and agent messages.
- Let the multipage frontend poll workflow status, show monitoring alerts, and render observability charts.

Detailed phase notes are kept in:

- `docs/phases/phase-1-foundation.md`
- `docs/phases/phase-2-planner-research-approval.md`
- `docs/phases/phase-3-persistent-memory.md`
- `docs/phases/phase-4-critic-structured-reports.md`
- `docs/phases/phase-5-background-observability-monitoring.md`

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
Background Workflow Runner
    |
    | resumes graph from checkpoint
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
    | critic review + targeted research loop
    v
CriticReport[]
    |
    | writer node
    v
WriterAgent
    |
    | OpenRouter chat completion
    v
Structured Final Report
    |
    | store memory + update workflow_run state
    v
PostgreSQL
    |
    | token usage + run cost + metrics
    v
Observability Dashboard
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
- APScheduler for recurring monitoring jobs
- Prometheus client for metrics export
- Rich for CLI/background task logging
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
    critic.py                     CriticAgent
    research.py                   ResearchAgent
    writer.py                     WriterAgent
  api/routes/
    monitoring.py                  Monitoring job and alert routes
    observability.py               Token, cost, trace, and metrics routes
    projects.py                    Project and memory routes
    workflows.py                  Workflow API routes
  db/
    checkpointer.py               LangGraph PostgresSaver setup
    connection.py                 asyncpg pool and query helpers
    migrations/001_initial.sql    Initial database schema
    migrations/002_phase2_statuses.sql
    migrations/002_vector_memory.sql
    migrations/003_observability.sql
    migrations/003_phase4_statuses.sql
    migrations/004_phase5_statuses.sql
  memory/
    chunker.py                    Token-aware chunking
    embeddings.py                 OpenAI embedding helpers
    store.py                      pgvector memory store
  graphs/
    research_graph.py             LangGraph workflow definition
  observability/
    cost_estimator.py              Pre-run workflow cost estimates
    token_tracker.py               Token usage and cost persistence
  monitoring/
    scheduler.py                   APScheduler monitoring jobs
  tasks/
    background_runner.py           Async workflow execution queue
  schemas/
    api.py                        API request/response models
    workflow.py                   Workflow state and agent models
  tests/
    test_phase1.py
    test_phase2.py
    test_phase3.py
    test_phase4.py
    test_phase5.py

frontend/
  dashboard.py                     Streamlit multipage entrypoint
  app.py                           Legacy single-page Streamlit app
  api_client.py                    Shared frontend API client
  pages/1_Workflows.py
  pages/2_Dashboard.py
  pages/3_Monitoring.py
  pages/4_Memory.py

docs/phases/
  phase-1-foundation.md
  phase-2-planner-research-approval.md
  phase-3-persistent-memory.md
  phase-4-critic-structured-reports.md
  phase-5-background-observability-monitoring.md

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
  "cost_estimate": {
    "min_usd": "0.075600",
    "max_usd": "0.156600",
    "estimated_usd": "0.108000",
    "breakdown_by_agent": {
      "planner": "0.014400",
      "research": "0.064800",
      "critic": "0.028800",
      "writer": "0.036000"
    }
  },
  "output": null
}
```

### Approve Workflow

```http
POST /workflows/{run_id}/approve
```

Resumes the graph from the human approval interrupt and starts research in the background.

### Cancel Workflow

```http
POST /workflows/{run_id}/cancel
```

Cancels an active background workflow and marks it as `cancelled`.

### Reject Workflow

```http
POST /workflows/{run_id}/reject
```

Resumes the graph with rejection and marks the workflow as rejected.

### Get Workflow Status

```http
GET /workflows/{run_id}/status
GET /workflows/{run_id}/report
GET /workflows/{run_id}/report.md
```

Returns status, plan, research results, critic reports, final report, final output, and serialized workflow state.

### Projects

```http
GET /projects
POST /projects
GET /projects/{project_id}/runs
GET /projects/{project_id}/memory?query=...
```

Project memory search returns the top relevant memory chunks.

### Monitoring

```http
POST /monitoring/jobs
GET /monitoring/jobs
DELETE /monitoring/jobs/{job_id}
GET /monitoring/alerts
GET /monitoring/alerts/{job_id}
```

Monitoring jobs run on cron schedules and create alerts when new relevant findings appear.

### Observability

```http
GET /observability/runs/{run_id}/tokens
GET /observability/projects/{project_id}/cost
GET /observability/runs/{run_id}/trace
GET /observability/dashboard
GET /metrics
```

Observability routes expose token usage, cost summaries, LangSmith trace links, and Prometheus metrics.

## Database Schema

The app stores product workflow data in:

- `projects`
- `workflow_runs`
- `agent_messages`
- `memory_chunks`
- `project_summaries`
- `token_usage`
- `monitoring_jobs`
- `monitoring_alerts`

LangGraph checkpoint tables are created by `PostgresSaver` during startup.

## Agent Flow

### BaseAgent

Shared base class for model-backed agents.

It handles:

- Model name and system prompt setup.
- Async OpenRouter calls.
- LangSmith tracing.
- Token usage logging and database-backed cost tracking when `run_id` is available.

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

### CriticAgent

Input:

- User goal.
- Workflow plan.
- Research results.

Output:

- `CriticReport` with findings and pass/fail recommendation.

It drives the targeted research loop before writing.

### WriterAgent

Input:

- User goal.
- Approved workflow plan.
- Research results.

Output:

- `FinalReport` JSON with inline citations and a confidence score.

## LangGraph Workflow

Defined in:

```text
backend/graphs/research_graph.py
```

Current graph:

```text
START -> planner -> human_approval -> memory_retrieval -> parallel_research -> critic -> [targeted_research -> critic]* -> writer -> memory_storage -> END
```

Rejected workflows route from `human_approval` directly to `END`.

Approved workflows are resumed by `WorkflowRunner` in `backend/tasks/background_runner.py`, so long-running research and writing work does not block the approval API request.

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
LANGSMITH_PROJECT=nexus-phase5
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
streamlit run frontend/dashboard.py
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

Run Phase 4 tests:

```powershell
python -m pytest backend/tests/test_phase4.py -q
```

Run Phase 5 tests:

```powershell
python -m pytest backend/tests/test_phase5.py -q
```

Tests mock model, embedding, Tavily, and database boundaries where appropriate.

## Documentation Rule

For every major change in project behavior, architecture, workflow, or code flow:

1. Update this README if the change affects how the project works or runs.
2. Add or update a phase document under `docs/phases/`.
3. Keep the architecture flow current.
4. Keep setup commands and environment variables current.
