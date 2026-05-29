import pytest

from middleware.rate_limiter import RateLimit, SlidingWindowRateLimiter
from middleware.request_queue import WorkflowExecutionQueue
from workflows.competitor_analysis import (
    CompetitorAnalysisInput,
    build_competitor_tasks,
    detect_outdated_data,
)
from workflows.market_research import (
    MarketResearchInput,
    build_goal,
    build_market_research_plan,
)
from backend.schemas.workflow import ResearchResult


def test_market_research_plan_has_required_tasks() -> None:
    request = MarketResearchInput(
        company_name="Acme",
        industry="SaaS",
        geography="India",
    )
    plan = build_market_research_plan(request)
    task_ids = {task.id for task in plan.subtasks}

    assert "Market Overview" in build_goal(request)
    assert {
        "market-size",
        "key-players",
        "market-trends",
        "regulatory-environment",
        "customer-segments",
    }.issubset(task_ids)


def test_competitor_tasks_are_generated_per_competitor_and_focus_area() -> None:
    request = CompetitorAnalysisInput(
        your_company="Us",
        competitors=["A", "B"],
        focus_areas=["product", "pricing"],
    )
    tasks = build_competitor_tasks(request)

    assert len(tasks) == 4
    assert {task.id for task in tasks} == {
        "a-product",
        "a-pricing",
        "b-product",
        "b-pricing",
    }


def test_outdated_competitor_data_is_flagged() -> None:
    report = detect_outdated_data(
        [
            ResearchResult(
                task_id="a-pricing",
                query="A pricing",
                url="https://example.com",
                title="Old Pricing",
                content="This pricing page was last updated on 2023-01-01.",
                relevance_score=8,
            )
        ]
    )

    assert report.findings
    assert report.findings[0].severity == "medium"


@pytest.mark.asyncio
async def test_workflow_execution_queue_limits_concurrency() -> None:
    queue = WorkflowExecutionQueue(max_concurrent=1)
    first = await queue.reserve("run-1")
    second = await queue.reserve("run-2")

    assert first.status == "researching"
    assert second.status == "queued"
    assert second.position == 1

    await queue.release("run-1")
    await queue.wait_for_slot("run-2")
    snapshot = await queue.snapshot()
    assert snapshot["active_runs"] == ["run-2"]


def test_sliding_window_rate_limiter_blocks_after_limit() -> None:
    limiter = SlidingWindowRateLimiter()
    limit = RateLimit(max_requests=2, window_seconds=60)

    assert limiter.check("ip", limit)[0] is True
    assert limiter.check("ip", limit)[0] is True
    allowed, retry_after = limiter.check("ip", limit)
    assert allowed is False
    assert retry_after > 0
