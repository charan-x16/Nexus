import argparse
import asyncio
import json
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.agents.research import ResearchAgent
from backend.db.checkpointer import setup_checkpointer
from backend.db.connection import close_pool, execute_query, init_pool, run_migrations
from backend.memory.store import MemoryStore
from backend.monitoring.scheduler import MonitoringJob, monitoring_scheduler
from backend.schemas.workflow import ResearchTask
from backend.config import settings


class TrendMonitorInput(BaseModel):
    topic: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    schedule: str = "0 9 * * 1"


async def setup_trend_monitor(request: TrendMonitorInput) -> dict[str, str | int]:
    await init_pool()
    await run_migrations()
    setup_checkpointer(settings.DATABASE_URL)

    project_id = uuid4()
    run_id = uuid4()
    job_id = uuid4()
    await execute_query(
        """
        INSERT INTO projects (id, name, goal)
        VALUES ($1, $2, $3)
        """,
        project_id,
        f"Trend Monitor: {request.topic}",
        f"Monitor trends for {request.topic}",
    )
    await execute_query(
        """
        INSERT INTO workflow_runs (id, project_id, status, state)
        VALUES ($1, $2, 'completed', '{}'::jsonb)
        """,
        run_id,
        project_id,
    )

    task = ResearchTask(
        id="trend-baseline",
        description=f"Establish a baseline for trend monitoring on {request.topic}.",
        search_queries=[f"{request.topic} {keyword}" for keyword in request.keywords],
        priority=1,
    )
    agent = ResearchAgent(goal=f"Trend baseline for {request.topic}")
    baseline_results = await agent.run(task)
    if baseline_results:
        await MemoryStore().store_research_results(
            run_id=str(run_id),
            project_id=str(project_id),
            results=baseline_results,
        )

    job = MonitoringJob(
        id=str(job_id),
        project_id=str(project_id),
        topic=request.topic,
        search_queries=request.keywords,
        schedule_cron=request.schedule,
    )
    await monitoring_scheduler.add_job(job)
    await close_pool()
    return {
        "project_id": str(project_id),
        "baseline_run_id": str(run_id),
        "monitoring_job_id": str(job_id),
        "baseline_results": len(baseline_results),
        "schedule": request.schedule,
    }


def parse_args() -> TrendMonitorInput:
    parser = argparse.ArgumentParser(description="Create a Nexus trend monitoring workflow.")
    parser.add_argument("--topic", required=True, help="Topic to monitor.")
    parser.add_argument("--keywords", required=True, help="Comma-separated keywords.")
    parser.add_argument(
        "--schedule",
        default="0 9 * * 1",
        help="Cron schedule. Default is weekly Monday at 09:00 UTC.",
    )
    args = parser.parse_args()
    return TrendMonitorInput(
        topic=args.topic,
        keywords=[item.strip() for item in args.keywords.split(",") if item.strip()],
        schedule=args.schedule,
    )


async def async_main() -> None:
    result = await setup_trend_monitor(parse_args())
    print(json.dumps(result, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
