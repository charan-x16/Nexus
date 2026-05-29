import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.graphs.research_graph import build_graph
from backend.schemas.workflow import ResearchResult, WorkflowPlan, WorkflowState


def sample_state() -> WorkflowState:
    return {
        "goal": "Create a short brief on using pgvector for semantic search.",
        "plan": None,
        "research_results": [],
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": "",
        "status": "planning",
        "awaiting_approval": False,
    }


@pytest.mark.asyncio
async def test_planner_agent_outputs_valid_workflow_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return json.dumps(
            {
                "title": "pgvector Semantic Search Brief",
                "goal": "Create a short brief on using pgvector for semantic search.",
                "subtasks": [
                    {
                        "id": "task-1",
                        "description": "Define evaluation criteria.",
                        "search_queries": ["pgvector semantic search evaluation"],
                        "priority": 1,
                        "status": "pending",
                    }
                ],
            }
        )

    monkeypatch.setattr(PlannerAgent, "_call_model", fake_call_model)

    result = await PlannerAgent().run(sample_state())

    assert isinstance(result["plan"], WorkflowPlan)
    assert result["awaiting_approval"] is True
    assert result["plan"].subtasks[0].id == "task-1"


@pytest.mark.asyncio
async def test_writer_agent_produces_non_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return "# pgvector Semantic Search Brief\n\nUse pgvector when embeddings belong close to relational data."

    monkeypatch.setattr(WriterAgent, "_call_model", fake_call_model)
    state = sample_state()
    state["plan"] = WorkflowPlan(
        title="pgvector Semantic Search Brief",
        goal=state["goal"],
        subtasks=[
            {
                "id": "task-1",
                "description": "Frame the goal.",
                "search_queries": ["pgvector semantic search"],
                "priority": 1,
                "status": "pending",
            }
        ],
    )
    state["research_results"] = [
        ResearchResult(
            task_id="task-1",
            query="pgvector semantic search",
            url="https://example.com/pgvector",
            title="pgvector",
            content="pgvector stores embeddings in PostgreSQL.",
            relevance_score=8,
        )
    ]

    result = await WriterAgent().run(state)

    assert result["final_output"]
    assert "pgvector" in result["final_output"]


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_after_mock_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_planner_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["plan"] = WorkflowPlan(
            title="Mock Plan",
            goal=state["goal"],
            subtasks=[
                {
                    "id": "task-1",
                    "description": "Plan",
                    "search_queries": ["LangGraph tutorial"],
                    "priority": 1,
                    "status": "pending",
                }
            ],
        )
        updated["awaiting_approval"] = True
        updated["status"] = "awaiting_approval"
        return updated

    async def fake_research_run(self, task):
        return [
            ResearchResult(
                task_id=task.id,
                query=task.search_queries[0],
                url="https://example.com",
                title="Example",
                content="A useful research result.",
                relevance_score=7,
            )
        ]

    async def fake_writer_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["final_output"] = "# Mock Report"
        updated["status"] = "completed"
        return updated

    monkeypatch.setattr(PlannerAgent, "run", fake_planner_run)
    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    monkeypatch.setattr(WriterAgent, "run", fake_writer_run)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "phase1-compat"}}
    paused_state = await graph.ainvoke(sample_state(), config=config)

    assert paused_state["status"] == "awaiting_approval"
    assert paused_state["awaiting_approval"] is True

    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["final_output"] == "# Mock Report"
    assert final_state["status"] == "completed"
