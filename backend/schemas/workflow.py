from datetime import datetime, timezone
from typing import Any, TypedDict

from pydantic import BaseModel, Field


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
    plan: TaskPlan | None
    research_results: list[str]
    draft: str | None
    final_output: str | None
    messages: list[AgentMessage]
    run_id: str
    status: str


def serialize_workflow_state(state: WorkflowState) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, BaseModel):
            serialized[key] = value.model_dump(mode="json")
        elif isinstance(value, list):
            serialized[key] = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized
