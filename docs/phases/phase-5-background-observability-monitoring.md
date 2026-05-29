# Phase 5: Background Execution, Monitoring, and Observability

## Goal

Move approved workflows into an async background runner, add scheduled monitoring jobs, and make model usage visible through token/cost tracking and dashboards.

New runtime flow:

```text
Planner -> Human Approval -> WorkflowRunner Queue -> Memory Retrieval -> Parallel Research -> Critic Loop -> Writer -> Memory Storage
```

New monitoring flow:

```text
Monitoring Job -> APScheduler -> Tavily Search -> Memory Comparison -> Alert Summary -> monitoring_alerts
```

## What Changed

- Added `token_usage`, `monitoring_jobs`, and `monitoring_alerts` tables.
- Added workflow status support for `cancelled`.
- Added `TokenTracker` with per-agent cost tracking.
- Integrated token tracking into `BaseAgent._call_model()`.
- Added `CostEstimate` and pre-run cost estimation for workflow creation responses.
- Added `WorkflowRunner` with an async queue, active run tracking, DB status updates, cancellation, and retrying DB writes.
- Updated workflow approval to enqueue the resumed graph instead of running it inside the HTTP request.
- Added `MonitoringScheduler` using APScheduler.
- Added monitoring APIs for jobs and alerts.
- Added observability APIs for token summaries, project cost summaries, LangSmith trace search links, and Prometheus metrics.
- Added a multipage Streamlit dashboard:
  - Workflows
  - Observability
  - Monitoring
  - Memory
- Updated Docker Compose to run `frontend/dashboard.py`.
- Added `backend/tests/test_phase5.py`.

## New Backend Files

```text
backend/api/routes/monitoring.py
backend/api/routes/observability.py
backend/monitoring/scheduler.py
backend/observability/cost_estimator.py
backend/tasks/background_runner.py
backend/db/migrations/003_observability.sql
backend/db/migrations/004_phase5_statuses.sql
```

## New Frontend Files

```text
frontend/dashboard.py
frontend/api_client.py
frontend/pages/1_Workflows.py
frontend/pages/2_Dashboard.py
frontend/pages/3_Monitoring.py
frontend/pages/4_Memory.py
```

## Phase 5 Status

Implementation is in place. Python syntax checks pass. Install the updated requirements before running the backend because Phase 5 adds APScheduler.
