from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import backend.memory.embeddings as embeddings_module
import backend.memory.store as store_module
from backend.agents.critic import CriticAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent
from backend.graphs.research_graph import build_graph
from backend.memory.chunker import chunk_text, estimate_tokens
from backend.memory.store import MemoryStore
from backend.schemas.workflow import CriticReport, MemoryChunk, ResearchResult, WorkflowPlan, WorkflowState


def phase3_state() -> WorkflowState:
    return {
        "project_id": "00000000-0000-0000-0000-000000000001",
        "goal": "Research durable memory for AI agents.",
        "plan": None,
        "research_results": [],
        "memory_context": "",
        "critic_reports": [],
        "critic_iteration": 0,
        "final_report": None,
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": "00000000-0000-0000-0000-000000000002",
        "status": "planning",
        "awaiting_approval": False,
    }


def test_chunk_text_splits_and_respects_token_limits() -> None:
    text = "\n\n".join(
        [
            "Persistent memory lets an agent reuse project context.",
            "Vector search finds related notes even when wording differs.",
            "Summaries keep long-running projects understandable.",
        ]
        * 80
    )

    chunks = chunk_text(text, chunk_size=64, overlap=8)

    assert len(chunks) > 1
    assert all(estimate_tokens(chunk) <= 64 for chunk in chunks)


@pytest.mark.asyncio
async def test_embed_texts_returns_1536_dim_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEmbeddings:
        async def create(self, **kwargs):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(index)] * 1536)
                    for index, _ in enumerate(kwargs["input"])
                ]
            )

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(embeddings_module, "_get_client", lambda: FakeClient())

    vectors = await embeddings_module.embed_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(vector) == 1536 for vector in vectors)


@pytest.mark.asyncio
async def test_memory_store_store_and_retrieve_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict] = []

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

    async def fake_embed_single(text: str) -> list[float]:
        return [0.1] * 1536

    async def fake_execute_query(query: str, *args):
        rows.append(
            {
                "content": args[2],
                "metadata": {
                    "task_id": "task-1",
                    "query": "agent memory",
                    "title": "Memory",
                    "relevance_score": 9,
                },
                "source_url": args[6],
                "score": 0.91,
            }
        )
        return "INSERT 0 1"

    async def fake_fetch_rows(query: str, *args):
        return rows[: args[2]]

    monkeypatch.setattr(store_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(store_module, "embed_single", fake_embed_single)
    monkeypatch.setattr(store_module, "execute_query", fake_execute_query)
    monkeypatch.setattr(store_module, "fetch_rows", fake_fetch_rows)

    result = ResearchResult(
        task_id="task-1",
        query="agent memory",
        url="https://example.com/memory",
        title="Memory",
        content="Persistent vector memory helps agents reuse prior research.",
        relevance_score=9,
    )
    store = MemoryStore()

    await store.store_research_results(
        run_id="00000000-0000-0000-0000-000000000002",
        project_id="00000000-0000-0000-0000-000000000001",
        results=[result],
    )
    retrieved = await store.retrieve(
        project_id="00000000-0000-0000-0000-000000000001",
        query="agent memory",
        top_k=5,
    )

    assert retrieved
    assert retrieved[0].source_url == "https://example.com/memory"
    assert "vector memory" in retrieved[0].content


@pytest.mark.asyncio
async def test_reranking_improves_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return '{"1": 2, "2": 9}'

    monkeypatch.setattr(store_module._MemoryReranker, "_call_model", fake_call_model)
    chunks = [
        MemoryChunk(content="General project notes", source_url="a", score=0.8),
        MemoryChunk(content="Detailed vector memory notes", source_url="b", score=0.2),
    ]

    reranked = await MemoryStore().rerank("vector memory", chunks, top_k=2)

    assert reranked[0].source_url == "b"
    assert reranked[0].score == 9


@pytest.mark.asyncio
async def test_full_graph_stores_memory_rows_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_rows: list[ResearchResult] = []

    async def fake_planner_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["plan"] = WorkflowPlan(
            title="Memory Plan",
            goal=state["goal"],
            subtasks=[
                {
                    "id": "task-1",
                    "description": "Research vector memory.",
                    "search_queries": ["pgvector memory"],
                    "priority": 1,
                    "status": "pending",
                }
            ],
        )
        updated["awaiting_approval"] = True
        updated["status"] = "awaiting_approval"
        return updated

    async def fake_retrieve_context(*args, **kwargs) -> str:
        return "Existing summary about durable agent memory."

    async def fake_research_run(self, task):
        return [
            ResearchResult(
                task_id=task.id,
                query=task.search_queries[0],
                url="https://example.com/vector-memory",
                title="Vector Memory",
                content="Vector memory stores chunks for semantic retrieval.",
                relevance_score=9,
            )
        ]

    async def fake_writer_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["final_output"] = "# Memory Report"
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

    async def fake_store_research_results(self, run_id, project_id, results):
        stored_rows.extend(results)

    async def fake_summarise_and_store(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(PlannerAgent, "run", fake_planner_run)
    monkeypatch.setattr(MemoryAgent, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    monkeypatch.setattr(CriticAgent, "run", fake_critic_run)
    monkeypatch.setattr(WriterAgent, "run", fake_writer_run)
    monkeypatch.setattr(MemoryStore, "store_research_results", fake_store_research_results)
    monkeypatch.setattr(MemoryAgent, "summarise_and_store", fake_summarise_and_store)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "phase3-memory"}}
    paused_state = await graph.ainvoke(phase3_state(), config=config)
    assert paused_state["status"] == "awaiting_approval"

    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["status"] == "completed"
    assert stored_rows
