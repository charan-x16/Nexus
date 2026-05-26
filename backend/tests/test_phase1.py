import json
from uuid import uuid4

import pytest

from backend.agents.planner import PlannerAgent
from backend.agents.writer import WriterAgent
from backend.graphs.research_graph import compiled_graph
from backend.schemas.workflow import TaskPlan, WorkflowState


def sample_state() -> WorkflowState:
    return {
        "goal": "Create a short brief on using pgvector for semantic search.",
        "plan": None,
        "research_results": [],
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": str(uuid4()),
        "status": "running",
    }


@pytest.mark.asyncio
async def test_planner_agent_outputs_valid_task_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return json.dumps(
            {
                "title": "pgvector Semantic Search Brief",
                "description": "Plan a concise technical brief for evaluating pgvector.",
                "subtasks": [
                    "Define the evaluation goal",
                    "Outline implementation considerations",
                    "Summarize tradeoffs and next steps",
                ],
                "estimated_steps": 3,
            }
        )

    monkeypatch.setattr(PlannerAgent, "_call_model", fake_call_model)

    result = await PlannerAgent().run(sample_state())

    assert isinstance(result["plan"], TaskPlan)
    assert result["plan"].estimated_steps == 3
    assert len(result["plan"].subtasks) == 3


@pytest.mark.asyncio
async def test_writer_agent_produces_non_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return "# pgvector Semantic Search Brief\n\nUse pgvector when embeddings belong close to relational data."

    monkeypatch.setattr(WriterAgent, "_call_model", fake_call_model)
    state = sample_state()
    state["plan"] = TaskPlan(
        title="pgvector Semantic Search Brief",
        description="Evaluate pgvector for semantic search.",
        subtasks=["Frame the goal", "Compare tradeoffs", "Recommend next steps"],
        estimated_steps=3,
    )

    result = await WriterAgent().run(state)

    assert result["final_output"]
    assert "pgvector" in result["final_output"]


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_with_mock_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_planner_call(*args, **kwargs) -> str:
        return json.dumps(
            {
                "title": "Mock Plan",
                "description": "A complete mock plan for the vertical slice.",
                "subtasks": ["Plan", "Write"],
                "estimated_steps": 2,
            }
        )

    async def fake_writer_call(*args, **kwargs) -> str:
        return "# Mock Report\n\nThe planner and writer completed the vertical slice."

    monkeypatch.setattr(PlannerAgent, "_call_model", fake_planner_call)
    monkeypatch.setattr(WriterAgent, "_call_model", fake_writer_call)

    run_id = str(uuid4())
    final_state = await compiled_graph.ainvoke(
        sample_state() | {"run_id": run_id},
        config={"configurable": {"thread_id": run_id}},
    )

    assert isinstance(final_state["plan"], TaskPlan)
    assert final_state["final_output"]
    assert final_state["status"] == "completed"
