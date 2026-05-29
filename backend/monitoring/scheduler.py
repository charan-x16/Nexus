import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

from backend.agents.base import BaseAgent
from backend.agents.research import ResearchAgent
from backend.config import settings
from backend.db.connection import execute_query, fetch_rows
from backend.memory.store import MemoryStore
from backend.schemas.workflow import ResearchTask, SearchResult


WebhookCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
console = Console()


@dataclass(frozen=True)
class MonitoringJob:
    id: str
    project_id: str
    topic: str
    search_queries: list[str]
    schedule_cron: str
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    is_active: bool = True
    created_at: datetime | None = None


class _MonitoringSummaryAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=(
                "You summarize newly discovered monitoring findings. Be brief, "
                "specific, and cite source URLs in prose when useful."
            ),
        )

    async def run(self, topic: str, content: str) -> str:
        try:
            return await self._call_model(
                [
                    {
                        "role": "user",
                        "content": (
                            "Write a concise monitoring alert summary in 3 to 5 "
                            "sentences.\n\n"
                            f"Topic: {topic}\n\n"
                            f"New finding:\n{content[:5000]}"
                        ),
                    }
                ],
                max_tokens=350,
                temperature=0.1,
            )
        except Exception:
            return content[:700].strip()


class MonitoringScheduler:
    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.webhooks: list[WebhookCallback] = []
        self._summary_agent = _MonitoringSummaryAgent()

    async def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
        jobs = await self._load_active_jobs()
        for job in jobs:
            self._schedule_runtime_job(job)
        console.log(f"[monitoring] scheduled {len(jobs)} active jobs")

    async def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        console.log("[monitoring] stopped")

    async def add_job(self, job: MonitoringJob) -> MonitoringJob:
        trigger = CronTrigger.from_crontab(job.schedule_cron, timezone="UTC")
        await _execute_with_retry(
            """
            INSERT INTO monitoring_jobs (
                id,
                project_id,
                topic,
                search_queries,
                schedule_cron,
                next_run_at,
                is_active
            )
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, TRUE)
            """,
            UUID(job.id),
            UUID(job.project_id),
            job.topic,
            json.dumps(job.search_queries),
            job.schedule_cron,
            job.next_run_at,
        )
        self._schedule_runtime_job(job, trigger=trigger)
        scheduled = self.scheduler.get_job(job.id)
        next_run_at = getattr(scheduled, "next_run_time", None) if scheduled else None
        await _execute_with_retry(
            """
            UPDATE monitoring_jobs
            SET next_run_at = $2
            WHERE id = $1
            """,
            UUID(job.id),
            next_run_at,
        )
        return job

    async def deactivate_job(self, job_id: str) -> None:
        scheduler_job = self.scheduler.get_job(job_id)
        if scheduler_job is not None:
            scheduler_job.remove()
        await _execute_with_retry(
            """
            UPDATE monitoring_jobs
            SET is_active = FALSE,
                next_run_at = NULL
            WHERE id = $1
            """,
            UUID(str(job_id)),
        )

    async def run_monitoring_job(self, job_id: str) -> None:
        rows = await fetch_rows(
            """
            SELECT id, project_id, topic, search_queries, schedule_cron,
                   last_run_at, next_run_at, is_active, created_at
            FROM monitoring_jobs
            WHERE id = $1
            """,
            UUID(str(job_id)),
        )
        if not rows:
            return
        job = _job_from_row(rows[0])
        if not job.is_active:
            return

        known_urls = await self._known_memory_urls(job)
        research_agent = ResearchAgent(goal=job.topic)
        task = ResearchTask(
            id=f"monitoring-{job.id}",
            description=f"Monitor updates for {job.topic}",
            search_queries=job.search_queries,
            priority=1,
        )
        best_finding: dict[str, Any] | None = None
        for query in job.search_queries:
            results = await research_agent.tavily_search(query)
            for result in results[:5]:
                if result.url in known_urls:
                    continue
                score = await self._score_result(research_agent, task, query, result)
                if score < 0.6:
                    continue
                finding = {
                    "query": query,
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "relevance_score": score,
                }
                if best_finding is None or score > best_finding["relevance_score"]:
                    best_finding = finding

        if best_finding is not None:
            summary = await self._summary_agent.run(
                topic=job.topic,
                content=(
                    f"Title: {best_finding['title']}\n"
                    f"URL: {best_finding['url']}\n"
                    f"Content: {best_finding['content']}"
                ),
            )
            await _execute_with_retry(
                """
                INSERT INTO monitoring_alerts (
                    job_id,
                    summary,
                    new_findings,
                    relevance_score
                )
                VALUES ($1, $2, $3, $4)
                """,
                UUID(job.id),
                summary,
                json.dumps(best_finding),
                best_finding["relevance_score"],
            )
            await self._dispatch_webhooks(
                {
                    "job_id": job.id,
                    "project_id": job.project_id,
                    "topic": job.topic,
                    "summary": summary,
                    "new_findings": best_finding,
                }
            )

        scheduled = self.scheduler.get_job(job.id)
        next_run_at = getattr(scheduled, "next_run_time", None) if scheduled else None
        await _execute_with_retry(
            """
            UPDATE monitoring_jobs
            SET last_run_at = NOW(),
                next_run_at = $2
            WHERE id = $1
            """,
            UUID(job.id),
            next_run_at,
        )

    def register_webhook(self, callback: WebhookCallback) -> None:
        self.webhooks.append(callback)

    async def _load_active_jobs(self) -> list[MonitoringJob]:
        rows = await fetch_rows(
            """
            SELECT id, project_id, topic, search_queries, schedule_cron,
                   last_run_at, next_run_at, is_active, created_at
            FROM monitoring_jobs
            WHERE is_active = TRUE
            """
        )
        return [_job_from_row(row) for row in rows]

    def _schedule_runtime_job(
        self,
        job: MonitoringJob,
        trigger: CronTrigger | None = None,
    ) -> None:
        trigger = trigger or CronTrigger.from_crontab(job.schedule_cron, timezone="UTC")
        self.scheduler.add_job(
            self.run_monitoring_job,
            trigger=trigger,
            args=[job.id],
            id=job.id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    async def _known_memory_urls(self, job: MonitoringJob) -> set[str]:
        try:
            chunks = await MemoryStore().retrieve(
                project_id=job.project_id,
                query=job.topic,
                top_k=20,
            )
            return {chunk.source_url for chunk in chunks if chunk.source_url}
        except Exception:
            return set()

    async def _score_result(
        self,
        research_agent: ResearchAgent,
        task: ResearchTask,
        query: str,
        result: SearchResult,
    ) -> float:
        score = await research_agent._score_relevance(
            task=task,
            query=query,
            result=result,
            content=result.content,
        )
        return max(0.0, min(1.0, score / 10))

    async def _dispatch_webhooks(self, payload: dict[str, Any]) -> None:
        for webhook in self.webhooks:
            result = webhook(payload)
            if asyncio.iscoroutine(result):
                await result


monitoring_scheduler = MonitoringScheduler()


def _job_from_row(row: Any) -> MonitoringJob:
    search_queries = row["search_queries"]
    if isinstance(search_queries, str):
        parsed = json.loads(search_queries)
        search_queries = parsed if isinstance(parsed, list) else []
    return MonitoringJob(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        topic=str(row["topic"]),
        search_queries=[str(item) for item in search_queries],
        schedule_cron=str(row["schedule_cron"]),
        last_run_at=row["last_run_at"],
        next_run_at=row["next_run_at"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
    )


async def _execute_with_retry(query: str, *args: Any) -> str:
    for attempt in range(3):
        try:
            return await execute_query(query, *args)
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.5 * (2**attempt))
    raise RuntimeError("Database command failed after retries.")
