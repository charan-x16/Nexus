from decimal import Decimal
from uuid import uuid4

import pytest

from backend.monitoring.scheduler import MonitoringScheduler
from backend.observability.cost_estimator import estimate_workflow_cost
from backend.observability.token_tracker import TokenTracker
from backend.schemas.workflow import ResearchTask, SearchResult, WorkflowPlan
from backend.tasks.background_runner import WorkflowJob, WorkflowRunner


def phase5_plan(task_count: int = 3) -> WorkflowPlan:
    return WorkflowPlan(
        title="Phase 5 plan",
        goal="Track observability for a research workflow.",
        subtasks=[
            ResearchTask(
                id=f"task-{index}",
                description=f"Research task {index}",
                search_queries=[f"query {index}"],
                priority=index,
            )
            for index in range(1, task_count + 1)
        ],
    )


@pytest.mark.asyncio
async def test_token_tracker_records_and_calculates_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_execute(query: str, *args):
        calls.append((query, args))
        return "INSERT 0 1"

    monkeypatch.setattr("backend.observability.token_tracker.execute_query", fake_execute)
    tracker = TokenTracker()
    cost = await tracker.record(
        run_id=uuid4(),
        agent_name="PlannerAgent",
        model="anthropic/claude-sonnet-4",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert cost == Decimal("0.018000")
    assert calls
    assert calls[0][1][3] == 1000
    assert calls[0][1][4] == 1000


def test_cost_estimator_produces_reasonable_estimate_for_three_task_plan() -> None:
    estimate = estimate_workflow_cost(phase5_plan(3))

    assert estimate.estimated_usd > Decimal("0")
    assert estimate.max_usd >= estimate.estimated_usd >= estimate.min_usd
    assert estimate.breakdown_by_agent["research"] > estimate.breakdown_by_agent["planner"]


@pytest.mark.asyncio
async def test_workflow_runner_submits_and_tracks_job(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = WorkflowRunner()
    completed: list[str] = []

    async def fake_run(job: WorkflowJob) -> None:
        completed.append(job.run_id)

    monkeypatch.setattr(runner, "_run_with_tracking", fake_run)
    await runner.start()
    run_id = str(uuid4())
    await runner.submit(run_id=run_id, goal="Test goal", project_id=str(uuid4()))
    await runner.run_queue.join()
    await runner.stop()

    assert completed == [run_id]
    assert run_id not in runner.active_runs


@pytest.mark.asyncio
async def test_monitoring_scheduler_runs_job_and_produces_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    project_id = uuid4()
    inserted_queries: list[str] = []

    async def fake_fetch_rows(query: str, *args):
        if "FROM monitoring_jobs" in query:
            return [
                {
                    "id": job_id,
                    "project_id": project_id,
                    "topic": "LangGraph releases",
                    "search_queries": ["LangGraph release notes"],
                    "schedule_cron": "0 * * * *",
                    "last_run_at": None,
                    "next_run_at": None,
                    "is_active": True,
                    "created_at": None,
                }
            ]
        return []

    async def fake_execute(query: str, *args):
        inserted_queries.append(query)
        return "OK"

    async def fake_retrieve(self, project_id: str, query: str, top_k: int = 20):
        return []

    async def fake_search(self, query: str):
        return [
            SearchResult(
                url="https://example.com/langgraph",
                title="LangGraph update",
                content="LangGraph announced a relevant update.",
            )
        ]

    async def fake_score(self, task, query, result, content):
        return 8.0

    async def fake_summary(topic: str, content: str) -> str:
        return "New relevant LangGraph update found."

    monkeypatch.setattr("backend.monitoring.scheduler.fetch_rows", fake_fetch_rows)
    monkeypatch.setattr("backend.monitoring.scheduler.execute_query", fake_execute)
    monkeypatch.setattr("backend.monitoring.scheduler.MemoryStore.retrieve", fake_retrieve)
    monkeypatch.setattr("backend.monitoring.scheduler.ResearchAgent.tavily_search", fake_search)
    monkeypatch.setattr("backend.monitoring.scheduler.ResearchAgent._score_relevance", fake_score)

    scheduler = MonitoringScheduler()
    monkeypatch.setattr(scheduler._summary_agent, "run", fake_summary)
    await scheduler.run_monitoring_job(str(job_id))

    assert any("INSERT INTO monitoring_alerts" in query for query in inserted_queries)
    assert any("UPDATE monitoring_jobs" in query for query in inserted_queries)


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint_returns_valid_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.api.routes import observability

    async def fake_fetch_rows(query: str, *args):
        return [
            {
                "total_runs": 2,
                "total_tokens": 42,
                "total_cost_usd": Decimal("0.123456"),
            }
        ]

    monkeypatch.setattr(observability, "fetch_rows", fake_fetch_rows)
    response = await observability.prometheus_metrics(_pool=object())

    assert response.status_code == 200
    assert b"nexus_total_runs" in response.body
    assert b"nexus_total_tokens" in response.body
