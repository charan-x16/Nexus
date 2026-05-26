import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from backend.db.connection import execute_query, fetch_rows, get_pool
from backend.graphs.research_graph import compiled_graph
from backend.schemas.api import (
    WorkflowCreateRequest,
    WorkflowCreateResponse,
    WorkflowStatusResponse,
)
from backend.schemas.workflow import AgentMessage, WorkflowState, serialize_workflow_state

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_workflow(
    request: WorkflowCreateRequest,
    background_tasks: BackgroundTasks,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowCreateResponse:
    project_id = uuid4()
    run_id = uuid4()
    initial_state: WorkflowState = {
        "goal": request.goal,
        "plan": None,
        "research_results": [],
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": str(run_id),
        "status": "queued",
    }

    await execute_query(
        """
        INSERT INTO projects (id, name, goal)
        VALUES ($1, $2, $3)
        """,
        project_id,
        request.project_name,
        request.goal,
    )
    await execute_query(
        """
        INSERT INTO workflow_runs (id, project_id, status, state)
        VALUES ($1, $2, $3, $4::jsonb)
        """,
        run_id,
        project_id,
        "queued",
        json.dumps(serialize_workflow_state(initial_state)),
    )

    background_tasks.add_task(_run_workflow_background, run_id, initial_state)
    return WorkflowCreateResponse(run_id=run_id, status="queued", output=None)


@router.get("/{run_id}", response_model=WorkflowStatusResponse)
async def get_workflow(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowStatusResponse:
    rows = await fetch_rows(
        """
        SELECT id, status, state
        FROM workflow_runs
        WHERE id = $1
        """,
        run_id,
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found.")

    row = rows[0]
    state_value = _decode_state(row["state"])
    return WorkflowStatusResponse(
        run_id=row["id"],
        status=row["status"],
        final_output=state_value.get("final_output"),
        state=state_value,
    )


async def _run_workflow_background(run_id: UUID, initial_state: WorkflowState) -> None:
    await _update_workflow_status(run_id, "running", initial_state | {"status": "running"})
    try:
        final_state = await compiled_graph.ainvoke(
            initial_state | {"status": "running"},
            config={"configurable": {"thread_id": str(run_id)}},
        )
        final_state["status"] = "completed"
        await _update_workflow_status(run_id, "completed", final_state)
        await _persist_agent_messages(run_id, final_state.get("messages", []))
    except Exception as exc:
        failed_state: WorkflowState = dict(initial_state)
        failed_state["status"] = "failed"
        failed_state["final_output"] = f"Workflow failed: {exc}"
        failed_state["messages"] = list(initial_state.get("messages", [])) + [
            AgentMessage(
                agent="system",
                role="error",
                content=str(exc),
                timestamp=datetime.now(timezone.utc),
            )
        ]
        await _update_workflow_status(run_id, "failed", failed_state)


async def _update_workflow_status(
    run_id: UUID,
    status_value: str,
    state: WorkflowState,
) -> None:
    await execute_query(
        """
        UPDATE workflow_runs
        SET status = $2,
            state = $3::jsonb,
            updated_at = NOW()
        WHERE id = $1
        """,
        run_id,
        status_value,
        json.dumps(serialize_workflow_state(state)),
    )


async def _persist_agent_messages(
    run_id: UUID,
    messages: list[AgentMessage],
) -> None:
    for message in messages:
        await execute_query(
            """
            INSERT INTO agent_messages (id, run_id, agent_name, role, content, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid4(),
            run_id,
            message.agent,
            message.role,
            message.content,
            message.timestamp,
        )


def _decode_state(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}
