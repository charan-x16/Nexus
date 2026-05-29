import asyncio
import time

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr

import backend.agents.research as research_module
from backend.agents.memory_agent import MemoryAgent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.graphs.research_graph import build_graph, parallel_research_node
from backend.memory.store import MemoryStore
from backend.schemas.workflow import (
    ResearchResult,
    ResearchTask,
    SearchResult,
    WorkflowPlan,
    WorkflowState,
)


def phase2_state() -> WorkflowState:
    return {
        "project_id": "00000000-0000-0000-0000-000000000001",
        "goal": "Research LangGraph tutorials.",
        "plan": WorkflowPlan(
            title="LangGraph Research",
            goal="Research LangGraph tutorials.",
            subtasks=[
                ResearchTask(
                    id=f"task-{index}",
                    description=f"Research task {index}",
                    search_queries=[f"LangGraph tutorial {index}"],
                    priority=index,
                )
                for index in range(1, 4)
            ],
        ),
        "research_results": [],
        "memory_context": "",
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": "",
        "status": "approved",
        "awaiting_approval": False,
    }


@pytest.mark.asyncio
async def test_research_agent_tavily_search_returns_results_for_langgraph_tutorial(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncTavilyClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def search(self, **kwargs):
            assert kwargs["query"] == "LangGraph tutorial"
            assert kwargs["search_depth"] == "advanced"
            return {
                "results": [
                    {
                        "url": "https://langchain-ai.github.io/langgraph/",
                        "title": "LangGraph Tutorial",
                        "content": "Build stateful agents with LangGraph.",
                    }
                ]
            }

    mocker.patch.object(research_module, "AsyncTavilyClient", FakeAsyncTavilyClient)
    monkeypatch.setattr(
        research_module.settings,
        "TAVILY_API_KEY",
        SecretStr("fake-tavily-key"),
    )

    results = await ResearchAgent().tavily_search("LangGraph tutorial")

    assert results == [
        SearchResult(
            url="https://langchain-ai.github.io/langgraph/",
            title="LangGraph Tutorial",
            content="Build stateful agents with LangGraph.",
        )
    ]


@pytest.mark.asyncio
async def test_parallel_research_node_runs_three_tasks_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_research_run(self, task: ResearchTask):
        await asyncio.sleep(0.1)
        return [
            ResearchResult(
                task_id=task.id,
                query=task.search_queries[0],
                url=f"https://example.com/{task.id}",
                title=task.description,
                content="Research content.",
                relevance_score=task.priority,
            )
        ]

    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    started = time.perf_counter()

    updated_state = await parallel_research_node(phase2_state())

    elapsed = time.perf_counter() - started
    assert elapsed < 0.25
    assert len(updated_state["research_results"]) == 3


@pytest.mark.asyncio
async def test_human_approval_interrupt_pauses_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_planner_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["plan"] = phase2_state()["plan"]
        updated["awaiting_approval"] = True
        updated["status"] = "awaiting_approval"
        return updated

    async def fake_research_run(self, task: ResearchTask):
        return [
            ResearchResult(
                task_id=task.id,
                query=task.search_queries[0],
                url=f"https://example.com/{task.id}",
                title=task.description,
                content="Research content.",
                relevance_score=8,
            )
        ]

    async def fake_writer_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["final_output"] = "# Approved Report"
        updated["status"] = "completed"
        return updated

    async def fake_retrieve_context(*args, **kwargs) -> str:
        return ""

    async def fake_store_research_results(*args, **kwargs) -> None:
        return None

    async def fake_summarise_and_store(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(PlannerAgent, "run", fake_planner_run)
    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    monkeypatch.setattr(WriterAgent, "run", fake_writer_run)
    monkeypatch.setattr(MemoryAgent, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(MemoryStore, "store_research_results", fake_store_research_results)
    monkeypatch.setattr(MemoryAgent, "summarise_and_store", fake_summarise_and_store)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "approval-test"}}
    paused_state = await graph.ainvoke(phase2_state() | {"plan": None}, config=config)

    assert paused_state["status"] == "awaiting_approval"
    assert paused_state["awaiting_approval"] is True

    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["status"] == "completed"
    assert final_state["final_output"] == "# Approved Report"
