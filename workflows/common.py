import asyncio
from datetime import datetime, timezone
from typing import Any

from backend.agents.critic import CriticAgent
from backend.agents.research import ResearchAgent
from backend.agents.writer import WriterAgent, render_final_report_markdown
from backend.schemas.workflow import (
    AgentMessage,
    Citation,
    CriticFinding,
    CriticReport,
    FinalReport,
    ResearchResult,
    ResearchTask,
    WorkflowPlan,
    WorkflowState,
)


async def run_preplanned_pipeline(
    *,
    goal: str,
    plan: WorkflowPlan,
    memory_context: str = "",
    critic_reports: list[CriticReport] | None = None,
    max_critic_iterations: int = 3,
) -> FinalReport:
    state: WorkflowState = {
        "goal": goal,
        "plan": plan,
        "research_results": [],
        "memory_context": memory_context,
        "critic_reports": list(critic_reports or []),
        "critic_iteration": 0,
        "final_report": None,
        "draft": None,
        "final_output": None,
        "messages": [],
        "status": "researching",
        "awaiting_approval": False,
    }

    state["research_results"] = await run_parallel_research(
        goal=goal,
        tasks=plan.subtasks,
        memory_context=memory_context,
    )

    critic = CriticAgent()
    while True:
        state["critic_iteration"] = int(state.get("critic_iteration", 0) or 0) + 1
        report = await critic.run(state)
        if report.passed or int(state.get("critic_iteration", 0) or 0) >= max_critic_iterations:
            break

        high_findings = [finding for finding in report.findings if finding.severity == "high"]
        if not high_findings:
            break

        targeted = await run_targeted_research(
            goal=goal,
            tasks=plan.subtasks,
            findings=high_findings,
            memory_context=memory_context,
        )
        state["research_results"] = merge_research_results(
            state.get("research_results", []),
            targeted,
        )

    writer = WriterAgent()
    report = await writer.run(state)
    state["final_report"] = report
    state["final_output"] = render_final_report_markdown(report)
    state["messages"] = list(state.get("messages", [])) + [
        AgentMessage(
            agent="writer",
            role="assistant",
            content=report.model_dump_json(),
            timestamp=datetime.now(timezone.utc),
        )
    ]
    return report


async def run_parallel_research(
    *,
    goal: str,
    tasks: list[ResearchTask],
    memory_context: str = "",
) -> list[ResearchResult]:
    async def run_task(task: ResearchTask) -> list[ResearchResult]:
        agent = ResearchAgent(goal=goal, memory_context=memory_context)
        return await agent.run(task)

    results = await asyncio.gather(
        *(run_task(task) for task in sorted(tasks, key=lambda item: item.priority)),
        return_exceptions=True,
    )
    flattened: list[ResearchResult] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, list):
            flattened.extend(result)
    return sorted(flattened, key=lambda item: item.relevance_score, reverse=True)


async def run_targeted_research(
    *,
    goal: str,
    tasks: list[ResearchTask],
    findings: list[CriticFinding],
    memory_context: str = "",
) -> list[ResearchResult]:
    async def run_task(task: ResearchTask) -> list[ResearchResult]:
        agent = ResearchAgent(goal=goal, memory_context=memory_context)
        return await agent.targeted_research(task, findings)

    results = await asyncio.gather(
        *(run_task(task) for task in sorted(tasks, key=lambda item: item.priority)),
        return_exceptions=True,
    )
    flattened: list[ResearchResult] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, list):
            flattened.extend(result)
    return flattened


def merge_research_results(
    existing_results: list[ResearchResult] | list[dict[str, Any]],
    new_results: list[ResearchResult] | list[dict[str, Any]],
) -> list[ResearchResult]:
    by_url: dict[str, ResearchResult] = {}
    for raw_result in [*existing_results, *new_results]:
        result = (
            raw_result
            if isinstance(raw_result, ResearchResult)
            else ResearchResult.model_validate(raw_result)
        )
        current = by_url.get(result.url)
        if current is None or result.relevance_score > current.relevance_score:
            by_url[result.url] = result
    return sorted(by_url.values(), key=lambda item: item.relevance_score, reverse=True)


def citation_for_result(index: int, result: ResearchResult) -> Citation:
    return Citation(
        index=index,
        url=result.url,
        title=result.title or result.url,
        quote=result.content.strip().replace("\n", " ")[:260]
        or "Source content collected during research.",
    )


def print_report(report: FinalReport) -> None:
    print(render_final_report_markdown(report))
