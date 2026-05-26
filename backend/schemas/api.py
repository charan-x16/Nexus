from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class WorkflowCreateRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    project_name: str = Field(min_length=1, max_length=200)


class WorkflowCreateResponse(BaseModel):
    run_id: UUID
    status: str
    output: str | None = None


class WorkflowStatusResponse(BaseModel):
    run_id: UUID
    status: str
    final_output: str | None = None
    state: dict[str, Any]
