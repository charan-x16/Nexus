import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agents.critic import CriticAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.planner import PlannerAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent, calculate_confidence_score
from backend.graphs.research_graph import build_graph
from backend.memory.store import MemoryStore
from backend.schemas.workflow import (
    CriticFinding,
    CriticReport,
    FinalReport,
    ResearchResult,
    WorkflowPlan,
    WorkflowState,
)


def phase4_state() -> WorkflowState:
    return {
        "project_id": "00000000-0000-0000-0000-000000000001",
        "goal": "Evaluate whether Tool X improves analyst productivity.",
        "plan": WorkflowPlan(
            title="Tool X Evaluation",
            goal="Evaluate whether Tool X improves analyst productivity.",
            subtasks=[
                {
                    "id": "task-1",
                    "description": "Find evidence for productivity impact.",
                    "search_queries": ["Tool X analyst productivity evidence"],
                    "priority": 1,
                    "status": "pending",
                }
            ],
        ),
        "research_results": [
            ResearchResult(
                task_id="task-1",
                query="Tool X analyst productivity evidence",
                url="https://example.com/a",
                title="Positive Study",
                content="Tool X improved analyst productivity by 40 percent.",
                relevance_score=9,
            ),
            ResearchResult(
                task_id="task-1",
                query="Tool X analyst productivity evidence",
                url="https://example.com/b",
                title="Contrary Study",
                content="Tool X did not improve analyst productivity in controlled trials.",
                relevance_score=8,
            ),
        ],
        "memory_context": "",
        "critic_reports": [],
        "critic_iteration": 0,
        "final_report": None,
        "draft": None,
        "final_output": None,
        "messages": [],
        "run_id": "00000000-0000-0000-0000-000000000002",
        "status": "researching",
        "awaiting_approval": False,
    }


@pytest.mark.asyncio
async def test_critic_agent_identifies_planted_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return json.dumps(
            {
                "passed": False,
                "findings": [
                    {
                        "finding_type": "contradiction",
                        "description": "One source reports a 40 percent improvement while another reports no improvement.",
                        "affected_tasks": ["task-1"],
                        "severity": "high",
                    }
                ],
                "recommendation": "Run targeted research for independent confirmation.",
                "iteration": 1,
            }
        )

    monkeypatch.setattr(CriticAgent, "_call_model", fake_call_model)
    state = phase4_state()
    state["critic_iteration"] = 1

    report = await CriticAgent().run(state)

    assert report.passed is False
    assert report.findings[0].finding_type == "contradiction"
    assert state["critic_reports"]


@pytest.mark.asyncio
async def test_reflection_loop_terminates_after_max_three_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targeted_calls = 0

    async def fake_planner_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["plan"] = phase4_state()["plan"]
        updated["awaiting_approval"] = True
        updated["status"] = "awaiting_approval"
        return updated

    async def fake_retrieve_context(*args, **kwargs) -> str:
        return ""

    async def fake_research_run(self, task):
        return phase4_state()["research_results"]

    async def fake_critic_run(self, state: WorkflowState) -> CriticReport:
        report = CriticReport(
            passed=False,
            findings=[
                CriticFinding(
                    finding_type="weak_evidence",
                    description="More confirmation is required.",
                    affected_tasks=["task-1"],
                    severity="high",
                )
            ],
            recommendation="Continue targeted research.",
            iteration=state["critic_iteration"],
        )
        state["critic_reports"] = list(state.get("critic_reports", [])) + [report]
        return report

    async def fake_targeted_research(self, task, findings):
        nonlocal targeted_calls
        targeted_calls += 1
        return [
            ResearchResult(
                task_id=task.id,
                query="targeted confirmation",
                url=f"https://example.com/targeted-{targeted_calls}",
                title="Targeted Evidence",
                content="Additional targeted evidence.",
                relevance_score=7,
            )
        ]

    async def fake_writer_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["final_output"] = "# Report"
        updated["status"] = "completed"
        return updated

    async def fake_store_research_results(*args, **kwargs) -> None:
        return None

    async def fake_summarise_and_store(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(PlannerAgent, "run", fake_planner_run)
    monkeypatch.setattr(MemoryAgent, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(ResearchAgent, "run", fake_research_run)
    monkeypatch.setattr(CriticAgent, "run", fake_critic_run)
    monkeypatch.setattr(ResearchAgent, "targeted_research", fake_targeted_research)
    monkeypatch.setattr(WriterAgent, "run", fake_writer_run)
    monkeypatch.setattr(MemoryStore, "store_research_results", fake_store_research_results)
    monkeypatch.setattr(MemoryAgent, "summarise_and_store", fake_summarise_and_store)

    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "phase4-loop"}}
    await graph.ainvoke(phase4_state() | {"plan": None}, config=config)
    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["critic_iteration"] == 3
    assert len(final_state["critic_reports"]) == 3
    assert targeted_calls == 2
    assert final_state["status"] == "completed"


@pytest.mark.asyncio
async def test_writer_agent_produces_final_report_with_section_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_model(*args, **kwargs) -> str:
        return json.dumps(
            {
                "title": "Tool X Report",
                "executive_summary": "Evidence is mixed for Tool X [1].",
                "sections": [
                    {
                        "title": "Evidence",
                        "content": "One source reports productivity gains [1].",
                        "citations": [
                            {
                                "index": 1,
                                "url": "https://example.com/a",
                                "title": "Positive Study",
                                "quote": "Tool X improved analyst productivity by 40 percent.",
                            }
                        ],
                    }
                ],
                "all_citations": [
                    {
                        "index": 1,
                        "url": "https://example.com/a",
                        "title": "Positive Study",
                        "quote": "Tool X improved analyst productivity by 40 percent.",
                    }
                ],
                "confidence_score": 0.1,
                "generated_at": "2026-05-29T00:00:00Z",
            }
        )

    monkeypatch.setattr(WriterAgent, "_call_model", fake_call_model)

    report = await WriterAgent().run(phase4_state())

    assert isinstance(report, FinalReport)
    assert report.sections
    assert all(section.citations for section in report.sections)


def test_confidence_score_calculation_penalizes_high_severity_findings() -> None:
    score = calculate_confidence_score(
        research_results=phase4_state()["research_results"],
        critic_reports=[
            CriticReport(
                passed=False,
                findings=[
                    CriticFinding(
                        finding_type="unverified_claim",
                        description="Central claim is not verified.",
                        affected_tasks=["task-1"],
                        severity="high",
                    )
                ],
                recommendation="Verify the claim.",
                iteration=1,
            )
        ],
    )

    assert score == 0.75


@pytest.mark.asyncio
async def test_full_pipeline_records_critic_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_planner_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["plan"] = phase4_state()["plan"]
        updated["awaiting_approval"] = True
        updated["status"] = "awaiting_approval"
        return updated

    async def fake_retrieve_context(*args, **kwargs) -> str:
        return ""

    async def fake_research_run(self, task):
        return phase4_state()["research_results"]

    async def fake_critic_run(self, state: WorkflowState) -> CriticReport:
        report = CriticReport(
            passed=True,
            findings=[],
            recommendation="Research is acceptable.",
            iteration=state["critic_iteration"],
        )
        state["critic_reports"] = list(state.get("critic_reports", [])) + [report]
        return report

    async def fake_writer_run(self, state: WorkflowState) -> WorkflowState:
        updated = dict(state)
        updated["final_output"] = "# Report"
        updated["status"] = "completed"
        return updated

    async def fake_store_research_results(*args, **kwargs) -> None:
        return None

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
    config = {"configurable": {"thread_id": "phase4-integration"}}
    await graph.ainvoke(phase4_state() | {"plan": None}, config=config)
    final_state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert final_state["critic_reports"]
    assert final_state["status"] == "completed"
