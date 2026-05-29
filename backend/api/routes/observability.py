from decimal import Decimal
from typing import Any
from urllib.parse import quote
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from backend.config import settings
from backend.db.connection import fetch_rows, get_pool
from backend.observability.token_tracker import token_tracker
from backend.schemas.api import ProjectCostSummary, TokenUsageRunSummary

router = APIRouter(prefix="/observability", tags=["observability"])
metrics_router = APIRouter(tags=["metrics"])

TOTAL_RUNS = Gauge("nexus_total_runs", "Total workflow runs recorded by Nexus.")
TOTAL_TOKENS = Gauge("nexus_total_tokens", "Total model tokens recorded by Nexus.")
TOTAL_COST_USD = Gauge("nexus_total_cost_usd", "Total estimated model cost in USD.")


@router.get("/runs/{run_id}/tokens", response_model=TokenUsageRunSummary)
async def get_run_tokens(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> TokenUsageRunSummary:
    return TokenUsageRunSummary.model_validate(
        await token_tracker.get_run_summary(run_id)
    )


@router.get("/projects/{project_id}/cost", response_model=ProjectCostSummary)
async def get_project_cost(
    project_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> ProjectCostSummary:
    return ProjectCostSummary.model_validate(
        await token_tracker.get_project_summary(project_id)
    )


@router.get("/runs/{run_id}/trace")
async def get_run_trace(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, str]:
    project = quote(settings.LANGSMITH_PROJECT)
    return {
        "run_id": str(run_id),
        "trace_url": (
            "https://smith.langchain.com/"
            f"?project={project}&q={quote(str(run_id))}"
        ),
    }


@router.get("/dashboard")
async def get_observability_dashboard(
    _pool: asyncpg.Pool = Depends(get_pool),
) -> dict[str, Any]:
    month_rows = await fetch_rows(
        """
        SELECT
            COUNT(DISTINCT workflow_runs.id)::INT AS total_runs,
            COALESCE(SUM(token_usage.cost_usd), 0)::NUMERIC(10, 6) AS total_cost
        FROM workflow_runs
        LEFT JOIN token_usage ON token_usage.run_id = workflow_runs.id
        WHERE workflow_runs.created_at >= date_trunc('month', NOW())
        """
    )
    cost_rows = await fetch_rows(
        """
        SELECT
            projects.id AS project_id,
            projects.name AS project_name,
            COALESCE(SUM(token_usage.cost_usd), 0)::NUMERIC(10, 6) AS total_cost
        FROM projects
        JOIN workflow_runs ON workflow_runs.project_id = projects.id
        LEFT JOIN token_usage ON token_usage.run_id = workflow_runs.id
        WHERE workflow_runs.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY projects.id, projects.name
        ORDER BY total_cost DESC
        """
    )
    recent_rows = await fetch_rows(
        """
        SELECT
            workflow_runs.id,
            workflow_runs.project_id,
            projects.name AS project_name,
            workflow_runs.status,
            workflow_runs.created_at,
            workflow_runs.updated_at
        FROM workflow_runs
        JOIN projects ON projects.id = workflow_runs.project_id
        ORDER BY workflow_runs.updated_at DESC
        LIMIT 20
        """
    )
    month = month_rows[0] if month_rows else None
    total_runs = int(month["total_runs"] or 0) if month else 0
    total_cost = Decimal(str(month["total_cost"] or "0")) if month else Decimal("0")
    avg_cost = total_cost / Decimal(total_runs) if total_runs else Decimal("0")
    return {
        "total_runs_this_month": total_runs,
        "total_cost_this_month": total_cost.quantize(Decimal("0.000001")),
        "avg_cost_per_run_this_month": avg_cost.quantize(Decimal("0.000001")),
        "cost_by_project_last_30_days": [
            {
                "project_id": str(row["project_id"]),
                "project_name": row["project_name"],
                "total_cost": Decimal(str(row["total_cost"] or "0")).quantize(
                    Decimal("0.000001")
                ),
            }
            for row in cost_rows
        ],
        "recent_runs": [
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "project_name": row["project_name"],
                "status": row["status"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in recent_rows
        ],
    }


@metrics_router.get("/metrics")
async def prometheus_metrics(
    _pool: asyncpg.Pool = Depends(get_pool),
) -> Response:
    summary = await _metrics_summary()
    TOTAL_RUNS.set(summary["total_runs"])
    TOTAL_TOKENS.set(summary["total_tokens"])
    TOTAL_COST_USD.set(float(summary["total_cost_usd"]))
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _metrics_summary() -> dict[str, Any]:
    rows = await fetch_rows(
        """
        SELECT
            (SELECT COUNT(*)::INT FROM workflow_runs) AS total_runs,
            COALESCE(SUM(input_tokens + output_tokens), 0)::INT AS total_tokens,
            COALESCE(SUM(cost_usd), 0)::NUMERIC(10, 6) AS total_cost_usd
        FROM token_usage
        """
    )
    row = rows[0] if rows else None
    return {
        "total_runs": int(row["total_runs"] or 0) if row else 0,
        "total_tokens": int(row["total_tokens"] or 0) if row else 0,
        "total_cost_usd": Decimal(str(row["total_cost_usd"] or "0"))
        if row
        else Decimal("0"),
    }
