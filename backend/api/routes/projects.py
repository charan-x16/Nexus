import json
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.db.connection import execute_query, fetch_rows, get_pool
from backend.memory.store import MemoryStore
from backend.schemas.api import ProjectCreateRequest, ProjectResponse, WorkflowRunResponse
from backend.schemas.workflow import MemoryChunk

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[ProjectResponse]:
    rows = await fetch_rows(
        """
        SELECT id, name, goal, created_at
        FROM projects
        ORDER BY created_at DESC
        """
    )
    return [
        ProjectResponse(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: ProjectCreateRequest,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> ProjectResponse:
    project_id = uuid4()
    await execute_query(
        """
        INSERT INTO projects (id, name, goal)
        VALUES ($1, $2, $3)
        """,
        project_id,
        request.name,
        request.goal,
    )
    rows = await fetch_rows(
        """
        SELECT id, name, goal, created_at
        FROM projects
        WHERE id = $1
        """,
        project_id,
    )
    row = rows[0]
    return ProjectResponse(
        id=row["id"],
        name=row["name"],
        goal=row["goal"],
        created_at=row["created_at"].isoformat(),
    )


@router.get("/{project_id}/memory", response_model=list[MemoryChunk])
async def search_project_memory(
    project_id: UUID,
    query: str = Query(min_length=1),
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[MemoryChunk]:
    await _ensure_project_exists(project_id)
    store = MemoryStore()
    chunks = await store.retrieve(project_id=str(project_id), query=query, top_k=10)
    return await store.rerank(query=query, chunks=chunks, top_k=5)


@router.get("/{project_id}/runs", response_model=list[WorkflowRunResponse])
async def list_project_runs(
    project_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> list[WorkflowRunResponse]:
    await _ensure_project_exists(project_id)
    rows = await fetch_rows(
        """
        SELECT id, project_id, status, state, created_at, updated_at
        FROM workflow_runs
        WHERE project_id = $1
        ORDER BY created_at DESC
        """,
        project_id,
    )
    return [
        WorkflowRunResponse(
            id=row["id"],
            project_id=row["project_id"],
            status=row["status"],
            state=_decode_state(row["state"]),
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
        )
        for row in rows
    ]


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")


def _decode_state(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}
