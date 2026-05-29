import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.db.connection import fetch_rows, get_pool
from backend.monitoring.scheduler import MonitoringJob, monitoring_scheduler
from backend.schemas.api import (
    MonitoringAlertResponse,
    MonitoringJobCreateRequest,
    MonitoringJobResponse,
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post(
    "/jobs",
    response_model=MonitoringJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_monitoring_job(
    request: MonitoringJobCreateRequest,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> MonitoringJobResponse:
    await _ensure_project_exists(request.project_id)
    job = MonitoringJob(
        id=str(uuid4()),
        project_id=str(request.project_id),
        topic=request.topic,
        search_queries=request.search_queries,
        schedule_cron=request.schedule_cron,
    )
    try:
        await monitoring_scheduler.add_job(job)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cron schedule: {exc}",
        ) from exc
    row = await _load_job(job.id)
    return _job_response(row)


@router.get("/jobs", response_model=list[MonitoringJobResponse])
async def list_monitoring_jobs(
    project_id: UUID | None = Query(default=None),
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[MonitoringJobResponse]:
    if project_id is None:
        rows = await fetch_rows(
            """
            SELECT id, project_id, topic, search_queries, schedule_cron,
                   last_run_at, next_run_at, is_active, created_at
            FROM monitoring_jobs
            ORDER BY created_at DESC
            """
        )
    else:
        rows = await fetch_rows(
            """
            SELECT id, project_id, topic, search_queries, schedule_cron,
                   last_run_at, next_run_at, is_active, created_at
            FROM monitoring_jobs
            WHERE project_id = $1
            ORDER BY created_at DESC
            """,
            project_id,
        )
    return [_job_response(row) for row in rows]


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_monitoring_job(
    job_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> None:
    await _load_job(str(job_id))
    await monitoring_scheduler.deactivate_job(str(job_id))


@router.get("/alerts", response_model=list[MonitoringAlertResponse])
async def list_recent_alerts(
    project_id: UUID | None = Query(default=None),
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[MonitoringAlertResponse]:
    since = datetime.now(timezone.utc) - timedelta(days=7)
    if project_id is None:
        rows = await fetch_rows(
            """
            SELECT alerts.id, alerts.job_id, alerts.summary, alerts.new_findings,
                   alerts.relevance_score, alerts.created_at
            FROM monitoring_alerts AS alerts
            WHERE alerts.created_at >= $1
            ORDER BY alerts.created_at DESC
            """,
            since,
        )
    else:
        rows = await fetch_rows(
            """
            SELECT alerts.id, alerts.job_id, alerts.summary, alerts.new_findings,
                   alerts.relevance_score, alerts.created_at
            FROM monitoring_alerts AS alerts
            JOIN monitoring_jobs AS jobs ON jobs.id = alerts.job_id
            WHERE alerts.created_at >= $1
              AND jobs.project_id = $2
            ORDER BY alerts.created_at DESC
            """,
            since,
            project_id,
        )
    return [_alert_response(row) for row in rows]


@router.get("/alerts/{job_id}", response_model=list[MonitoringAlertResponse])
async def list_job_alerts(
    job_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[MonitoringAlertResponse]:
    await _load_job(str(job_id))
    rows = await fetch_rows(
        """
        SELECT id, job_id, summary, new_findings, relevance_score, created_at
        FROM monitoring_alerts
        WHERE job_id = $1
        ORDER BY created_at DESC
        """,
        job_id,
    )
    return [_alert_response(row) for row in rows]


async def _ensure_project_exists(project_id: UUID) -> None:
    rows = await fetch_rows(
        """
        SELECT id
        FROM projects
        WHERE id = $1
        """,
        project_id,
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )


async def _load_job(job_id: str) -> Any:
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Monitoring job not found.",
        )
    return rows[0]


def _job_response(row: Any) -> MonitoringJobResponse:
    search_queries = row["search_queries"]
    if isinstance(search_queries, str):
        decoded = json.loads(search_queries)
        search_queries = decoded if isinstance(decoded, list) else []
    return MonitoringJobResponse(
        id=row["id"],
        project_id=row["project_id"],
        topic=row["topic"],
        search_queries=[str(item) for item in search_queries],
        schedule_cron=row["schedule_cron"],
        last_run_at=row["last_run_at"],
        next_run_at=row["next_run_at"],
        is_active=row["is_active"],
        created_at=row["created_at"],
    )


def _alert_response(row: Any) -> MonitoringAlertResponse:
    return MonitoringAlertResponse(
        id=row["id"],
        job_id=row["job_id"],
        summary=row["summary"],
        new_findings=row["new_findings"],
        relevance_score=float(row["relevance_score"] or 0),
        created_at=row["created_at"],
    )
