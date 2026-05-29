from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.schemas.workflow import ResearchResult, WorkflowPlan


class WorkflowCreateRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    project_name: str = Field(min_length=1, max_length=200)
    project_id: UUID | None = None


class WorkflowCreateResponse(BaseModel):
    run_id: UUID
    status: str
    plan: WorkflowPlan | None = None
    output: str | None = None


class WorkflowDecisionResponse(BaseModel):
    run_id: UUID
    status: str


class WorkflowStatusResponse(BaseModel):
    run_id: UUID
    status: str
    plan: WorkflowPlan | None = None
    research_results: list[ResearchResult] = Field(default_factory=list)
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
