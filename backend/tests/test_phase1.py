import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.critic import CriticAgent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.graphs.research_graph import build_graph
from backend.agents.memory_agent import MemoryAgent
from backend.memory.store import MemoryStore
from backend.schemas.workflow import CriticReport, FinalReport, ResearchResult, WorkflowPlan, WorkflowState


def sample_state() -> WorkflowState:
    return {
        "project_id": "00000000-0000-0000-0000-000000000001",
        "goal": "Create a short brief on using pgvector for semantic search.",
        "plan": None,
        "research_results": [],
        "memory_context": "",
        "critic_reports": [],
        "critic_iteration": 0,
        "final_report": None,
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
        return json.dumps(
            {
                "title": "pgvector Semantic Search Brief",
                "executive_summary": "pgvector keeps embeddings close to PostgreSQL data [1].",
                "sections": [
                    {
                        "title": "Recommendation",
                        "content": "Use pgvector when embeddings belong close to relational data [1].",
                        "citations": [
                            {
                                "index": 1,
                                "url": "https://example.com/pgvector",
                                "title": "pgvector",
                                "quote": "pgvector stores embeddings in PostgreSQL.",
                            }
                        ],
                    }
                ],
                "all_citations": [
                    {
                        "index": 1,
                        "url": "https://example.com/pgvector",
                        "title": "pgvector",
                        "quote": "pgvector stores embeddings in PostgreSQL.",
                    }
                ],
                "confidence_score": 0.8,
                "generated_at": "2026-05-29T00:00:00Z",
            }
        )

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

    assert isinstance(result, FinalReport)
    assert "pgvector" in result.executive_summary


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

    async def fake_critic_run(self, state: WorkflowState) -> CriticReport:
        report = CriticReport(
            passed=True,
            findings=[],
            recommendation="Research is acceptable.",
            iteration=state.get("critic_iteration", 1),
        )
        state["critic_reports"] = [report]
        return report

    async def fake_retrieve_context(*args, **kwargs) -> str:
        return ""

    async def fake_store_research_results(*args, **kwargs) -> None:
        return None

    async def fake_summarise_and_store(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(PlannerAgent, "run", fake_planner_run)
    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    monkeypatch.setattr(CriticAgent, "run", fake_critic_run)
    monkeypatch.setattr(WriterAgent, "run", fake_writer_run)
    monkeypatch.setattr(MemoryAgent, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(MemoryStore, "store_research_results", fake_store_research_results)
    monkeypatch.setattr(MemoryAgent, "summarise_and_store", fake_summarise_and_store)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "phase1-compat"}}
    paused_state = await graph.ainvoke(sample_state(), config=config)

    assert paused_state["status"] == "awaiting_approval"
    assert paused_state["awaiting_approval"] is True

    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["final_output"] == "# Mock Report"
    assert final_state["status"] == "completed"
