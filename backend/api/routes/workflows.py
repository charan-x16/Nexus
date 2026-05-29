import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from langgraph.types import Command

from backend.db.connection import execute_query, fetch_rows, get_pool
from backend.graphs.research_graph import get_compiled_graph
from backend.schemas.api import (
    WorkflowCreateRequest,
    WorkflowCreateResponse,
    WorkflowDecisionResponse,
    WorkflowStatusResponse,
)
from backend.schemas.workflow import (
    AgentMessage,
    ResearchResult,
    WorkflowPlan,
    WorkflowState,
    serialize_workflow_state,
)

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_workflow(
    request: WorkflowCreateRequest,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowCreateResponse:
    project_id = request.project_id or uuid4()
    run_id = uuid4()
    initial_state: WorkflowState = {
        "project_id": str(project_id),
        "goal": request.goal,
        "plan": None,
        "research_results": [],
        "memory_context": "",
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": str(run_id),
        "status": "planning",
        "awaiting_approval": False,
    }

    if request.project_id is None:
        await execute_query(
            """
            INSERT INTO projects (id, name, goal)
            VALUES ($1, $2, $3)
            """,
            project_id,
            request.project_name,
            request.goal,
        )
    else:
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
    await execute_query(
        """
        INSERT INTO workflow_runs (id, project_id, status, state)
        VALUES ($1, $2, $3, $4::jsonb)
        """,
        run_id,
        project_id,
        "planning",
        json.dumps(serialize_workflow_state(initial_state)),
    )

    config = {"configurable": {"thread_id": str(run_id)}}
    try:
        graph = get_compiled_graph()
        graph_result = await graph.ainvoke(initial_state, config=config)
        state = await _graph_state_or_result(graph, config, graph_result)
        state["status"] = "awaiting_approval"
        state["awaiting_approval"] = True
        await _update_workflow_status(run_id, "awaiting_approval", state)
        return WorkflowCreateResponse(
            run_id=run_id,
            status="awaiting_approval",
            plan=_coerce_plan(state.get("plan")),
            output=None,
        )
    except Exception as exc:
        failed_state = _failed_state(initial_state, exc)
        await _update_workflow_status(run_id, "failed", failed_state)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/{run_id}/approve", response_model=WorkflowDecisionResponse)
async def approve_workflow(
    run_id: UUID,
    background_tasks: BackgroundTasks,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowDecisionResponse:
    state = await _load_state(run_id)
    if state.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workflow is not awaiting approval.",
        )

    state["awaiting_approval"] = False
    state["status"] = "researching"
    await _update_workflow_status(run_id, "researching", state)
    background_tasks.add_task(_resume_approved_workflow, run_id)
    return WorkflowDecisionResponse(run_id=run_id, status="researching")


@router.post("/{run_id}/reject", response_model=WorkflowDecisionResponse)
async def reject_workflow(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowDecisionResponse:
    state = await _load_state(run_id)
    if state.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workflow is not awaiting approval.",
        )

    config = {"configurable": {"thread_id": str(run_id)}}
    graph = get_compiled_graph()
    graph_result = await graph.ainvoke(Command(resume={"approved": False}), config=config)
    state = await _graph_state_or_result(graph, config, graph_result)
    state["awaiting_approval"] = False
    state["status"] = "rejected"
    await _update_workflow_status(run_id, "rejected", state)
    return WorkflowDecisionResponse(run_id=run_id, status="rejected")


@router.get("/{run_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowStatusResponse:
    state = await _load_state(run_id)
    return WorkflowStatusResponse(
        run_id=run_id,
        status=state.get("status", "unknown"),
        plan=_coerce_plan(state.get("plan")),
        research_results=_coerce_research_results(state.get("research_results", [])),
        final_output=state.get("final_output"),
        state=serialize_workflow_state(state),
    )


@router.get("/{run_id}", response_model=WorkflowStatusResponse)
async def get_workflow(
    run_id: UUID,
    _pool: asyncpg.Pool = Depends(get_pool),
) -> WorkflowStatusResponse:
    return await get_workflow_status(run_id, _pool)


async def _resume_approved_workflow(run_id: UUID) -> None:
    config = {"configurable": {"thread_id": str(run_id)}}
    try:
        graph = get_compiled_graph()
        graph_result = await graph.ainvoke(Command(resume={"approved": True}), config=config)
        final_state = await _graph_state_or_result(graph, config, graph_result)
        final_state["awaiting_approval"] = False
        final_state["status"] = "completed"
        await _update_workflow_status(run_id, "completed", final_state)
        await _persist_agent_messages(run_id, final_state.get("messages", []))
    except Exception as exc:
        current_state = await _load_state(run_id)
        failed_state = _failed_state(current_state, exc)
        await _update_workflow_status(run_id, "failed", failed_state)


async def _graph_state_or_result(
    graph: Any,
    config: dict[str, Any],
    graph_result: Any,
) -> WorkflowState:
    try:
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            return dict(values)
    except Exception:
        pass

    if isinstance(graph_result, dict):
        return dict(graph_result)
    return {}


async def _load_state(run_id: UUID) -> WorkflowState:
    rows = await fetch_rows(
        """
        SELECT state
        FROM workflow_runs
        WHERE id = $1
        """,
        run_id,
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found.")
    return _decode_state(rows[0]["state"])


async def _update_workflow_status(
    run_id: UUID,
    status_value: str,
    state: WorkflowState,
) -> None:
    state["status"] = status_value
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
    messages: list[AgentMessage] | list[dict[str, Any]],
) -> None:
    for raw_message in messages:
        message = (
            raw_message
            if isinstance(raw_message, AgentMessage)
            else AgentMessage.model_validate(raw_message)
        )
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


def _decode_state(value: Any) -> WorkflowState:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}


def _coerce_plan(value: Any) -> WorkflowPlan | None:
    if value is None:
        return None
    if isinstance(value, WorkflowPlan):
        return value
    return WorkflowPlan.model_validate(value)


def _coerce_research_results(value: Any) -> list[ResearchResult]:
    if not isinstance(value, list):
        return []
    return [
        item if isinstance(item, ResearchResult) else ResearchResult.model_validate(item)
        for item in value
    ]


def _failed_state(state: WorkflowState, exc: Exception) -> WorkflowState:
    failed_state: WorkflowState = dict(state)
    failed_state["status"] = "failed"
    failed_state["awaiting_approval"] = False
    failed_state["final_output"] = f"Workflow failed: {exc}"
    failed_state["messages"] = list(state.get("messages", [])) + [
        AgentMessage(
            agent="system",
            role="error",
            content=str(exc),
            timestamp=datetime.now(timezone.utc),
        )
    ]
    return failed_state
