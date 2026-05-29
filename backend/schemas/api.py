from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.observability.cost_estimator import CostEstimate
from backend.schemas.workflow import CriticReport, FinalReport, ResearchResult, WorkflowPlan


class WorkflowCreateRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    project_name: str = Field(min_length=1, max_length=200)
    project_id: UUID | None = None


class WorkflowCreateResponse(BaseModel):
    run_id: UUID
    status: str
    plan: WorkflowPlan | None = None
    cost_estimate: CostEstimate | None = None
    output: str | None = None


class WorkflowDecisionResponse(BaseModel):
    run_id: UUID
    status: str


class WorkflowStatusResponse(BaseModel):
    run_id: UUID
    status: str
    plan: WorkflowPlan | None = None
    research_results: list[ResearchResult] = Field(default_factory=list)
    critic_reports: list[CriticReport] = Field(default_factory=list)
    final_report: FinalReport | None = None
    final_output: str | None = None
    state: dict[str, Any]


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=8000)


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    goal: str
    created_at: str


class WorkflowRunResponse(BaseModel):
    id: UUID
    project_id: UUID
    status: str
    state: dict[str, Any]
    created_at: str
    updated_at: str


class MonitoringJobCreateRequest(BaseModel):
    project_id: UUID
    topic: str = Field(min_length=1, max_length=500)
    search_queries: list[str] = Field(min_length=1, max_length=20)
    schedule_cron: str = Field(min_length=9, max_length=120)


class MonitoringJobResponse(BaseModel):
    id: UUID
    project_id: UUID
    topic: str
    search_queries: list[str]
    schedule_cron: str
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    is_active: bool
    created_at: datetime


class MonitoringAlertResponse(BaseModel):
    id: UUID
    job_id: UUID
    summary: str
    new_findings: str
    relevance_score: float
    created_at: datetime


class TokenUsageAgentSummary(BaseModel):
    agent_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Decimal


class TokenUsageRunSummary(BaseModel):
    run_id: UUID
    total_input: int
    total_output: int
    total_tokens: int
    total_cost: Decimal
    by_agent: list[TokenUsageAgentSummary]


class ProjectCostSummary(BaseModel):
    project_id: UUID
    total_runs: int
    total_cost: Decimal
    avg_cost_per_run: Decimal
