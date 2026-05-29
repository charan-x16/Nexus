from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    url: str = Field(min_length=1)
    title: str = Field(default="", max_length=500)
    content: str = Field(default="", max_length=4000)


class ResearchTask(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)
    priority: int = Field(ge=1)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"


class WorkflowPlan(BaseModel):
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    subtasks: list[ResearchTask] = Field(min_length=1)


class ResearchResult(BaseModel):
    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = Field(default="", max_length=500)
    content: str = Field(default="", min_length=1, max_length=4000)
    relevance_score: float = Field(ge=0, le=10)


class TaskPlan(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    subtasks: list[str] = Field(min_length=1)
    estimated_steps: int = Field(ge=1)


class AgentMessage(BaseModel):
    agent: str = Field(min_length=1)
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowState(TypedDict, total=False):
    goal: str
    plan: WorkflowPlan | None
    research_results: list[ResearchResult]
    draft: str | None
    final_output: str | None
    messages: list[AgentMessage]
    run_id: str
    status: str
    awaiting_approval: bool


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def serialize_workflow_state(state: WorkflowState) -> dict[str, Any]:
    return {key: to_jsonable(value) for key, value in state.items()}
