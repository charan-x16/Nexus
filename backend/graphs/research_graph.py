import asyncio
import json
from typing import Any
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.db.checkpointer import get_checkpointer
from backend.db.connection import execute_query
from backend.schemas.workflow import (
    ResearchResult,
    ResearchTask,
    WorkflowPlan,
    WorkflowState,
    serialize_workflow_state,
)

planner_agent = PlannerAgent()
writer_agent = WriterAgent()
_compiled_graph: Any | None = None


async def planner_node(state: WorkflowState) -> WorkflowState:
    updated_state = await planner_agent.run(state)
    await _persist_graph_state(updated_state)
    return updated_state


def human_approval_node(state: WorkflowState) -> WorkflowState:
    plan = _coerce_plan(state.get("plan"))
    approval = interrupt(
        {
            "run_id": state.get("run_id"),
            "status": "awaiting_approval",
            "plan": plan.model_dump(mode="json") if plan is not None else None,
        }
    )
    approved = approval.get("approved") if isinstance(approval, dict) else bool(approval)
    updated_state: WorkflowState = dict(state)
    updated_state["awaiting_approval"] = False
    updated_state["status"] = "approved" if approved else "rejected"
    return updated_state


async def parallel_research_node(state: WorkflowState) -> WorkflowState:
    plan = _coerce_plan(state.get("plan"))
    if plan is None:
        raise ValueError("Research requires an approved workflow plan.")

    goal = state.get("goal", plan.goal)
    collected: list[ResearchResult] = []
    lock = asyncio.Lock()

    async def run_one_task(task: ResearchTask) -> list[ResearchResult]:
        agent = ResearchAgent(goal=goal)
        results = await agent.run(task)
        async with lock:
            collected.extend(results)
            progress_state: WorkflowState = dict(state)
            progress_state["awaiting_approval"] = False
            progress_state["status"] = "researching"
            progress_state["research_results"] = sorted(
                collected,
                key=lambda item: item.relevance_score,
                reverse=True,
            )
            await _persist_graph_state(progress_state)
        return results

    task_results = await asyncio.gather(
        *(run_one_task(task) for task in sorted(plan.subtasks, key=lambda item: item.priority)),
        return_exceptions=True,
    )
    flattened: list[ResearchResult] = []
    for result in task_results:
        if isinstance(result, Exception):
            continue
        flattened.extend(result)

    updated_state: WorkflowState = dict(state)
    updated_state["awaiting_approval"] = False
    updated_state["status"] = "researching"
    updated_state["research_results"] = sorted(
        flattened,
        key=lambda item: item.relevance_score,
        reverse=True,
    )
    await _persist_graph_state(updated_state)
    return updated_state


async def writer_node(state: WorkflowState) -> WorkflowState:
    updated_state = await writer_agent.run(state)
    await _persist_graph_state(updated_state)
    return updated_state


def route_after_approval(state: WorkflowState) -> str:
    return "rejected" if state.get("status") == "rejected" else "approved"


def build_graph(checkpointer: Any) -> Any:
    workflow = StateGraph(WorkflowState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("parallel_research", parallel_research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "human_approval")
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "approved": "parallel_research",
            "rejected": END,
        },
    )
    workflow.add_edge("parallel_research", "writer")
    workflow.add_edge("writer", END)
    return workflow.compile(checkpointer=checkpointer)


def get_compiled_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(get_checkpointer())
    return _compiled_graph


class LazyCompiledGraph:
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await get_compiled_graph().ainvoke(*args, **kwargs)

    async def aget_state(self, *args: Any, **kwargs: Any) -> Any:
        return await get_compiled_graph().aget_state(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(get_compiled_graph(), name)


compiled_graph = LazyCompiledGraph()


def _coerce_plan(value: Any) -> WorkflowPlan | None:
    if value is None:
        return None
    if isinstance(value, WorkflowPlan):
        return value
    return WorkflowPlan.model_validate(value)


async def _persist_graph_state(state: WorkflowState) -> None:
    run_id = state.get("run_id")
    if not run_id:
        return
    try:
        await execute_query(
            """
            UPDATE workflow_runs
            SET status = $2,
                state = $3::jsonb,
                updated_at = NOW()
            WHERE id = $1
            """,
            UUID(str(run_id)),
            state.get("status", "running"),
            json.dumps(serialize_workflow_state(state)),
        )
    except Exception:
        return
