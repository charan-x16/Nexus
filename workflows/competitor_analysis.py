import argparse
import asyncio
import re
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from backend.agents.critic import CriticAgent
from backend.agents.writer import WriterAgent
from backend.schemas.workflow import (
    Citation,
    CriticFinding,
    CriticReport,
    FinalReport,
    ReportSection,
    ResearchResult,
    ResearchTask,
    WorkflowPlan,
    WorkflowState,
)
from workflows.common import citation_for_result, merge_research_results, print_report


DEFAULT_FOCUS_AREAS = ["product", "pricing", "team", "funding", "reviews"]


class CompetitorAnalysisInput(BaseModel):
    your_company: str = Field(min_length=1)
    competitors: list[str] = Field(min_length=1)
    focus_areas: list[str] = Field(default_factory=lambda: list(DEFAULT_FOCUS_AREAS))


def build_goal(request: CompetitorAnalysisInput) -> str:
    return (
        f"Analyze competitors {', '.join(request.competitors)} for {request.your_company}. "
        f"Focus on {', '.join(request.focus_areas)}. "
        "Produce a comparison table, SWOT per competitor, and strategic recommendations."
    )


def build_competitor_tasks(request: CompetitorAnalysisInput) -> list[ResearchTask]:
    tasks: list[ResearchTask] = []
    priority = 1
    for competitor in request.competitors:
        for focus in request.focus_areas:
            tasks.append(
                ResearchTask(
                    id=f"{_slug(competitor)}-{_slug(focus)}",
                    description=f"Research {competitor} {focus} signals relevant to {request.your_company}.",
                    search_queries=[
                        f"{competitor} {focus} latest",
                        f"{competitor} {focus} review analysis",
                        f"{competitor} {focus} compared to {request.your_company}",
                    ],
                    priority=priority,
                )
            )
            priority += 1
    return tasks


async def run_competitor_analysis(request: CompetitorAnalysisInput) -> FinalReport:
    goal = build_goal(request)
    tasks = build_competitor_tasks(request)
    research_results = await _run_research_by_competitor(goal, request, tasks)
    outdated_report = detect_outdated_data(research_results)
    plan = WorkflowPlan(
        title=f"{request.your_company} Competitor Analysis",
        goal=goal,
        subtasks=tasks,
    )
    state: WorkflowState = {
        "goal": goal,
        "plan": plan,
        "research_results": research_results,
        "memory_context": "",
        "critic_reports": [outdated_report] if outdated_report.findings else [],
        "critic_iteration": 1 if outdated_report.findings else 0,
        "messages": [],
        "status": "criticizing",
        "awaiting_approval": False,
    }
    critic_report = await CriticAgent().run(state)
    if not critic_report.passed:
        state["critic_iteration"] = int(state.get("critic_iteration", 0) or 0) + 1
    report = await WriterAgent().run(state)
    return enhance_competitor_report(report, request, research_results)


async def _run_research_by_competitor(
    goal: str,
    request: CompetitorAnalysisInput,
    tasks: list[ResearchTask],
) -> list[ResearchResult]:
    from backend.agents.research import ResearchAgent

    async def run_competitor(competitor: str) -> list[ResearchResult]:
        agent = ResearchAgent(goal=goal)
        competitor_results: list[ResearchResult] = []
        for task in tasks:
            if task.id.startswith(_slug(competitor)):
                competitor_results.extend(await agent.run(task))
        return competitor_results

    grouped = await asyncio.gather(
        *(run_competitor(competitor) for competitor in request.competitors),
        return_exceptions=True,
    )
    results: list[ResearchResult] = []
    for item in grouped:
        if isinstance(item, Exception):
            continue
        if isinstance(item, list):
            results.extend(item)
    return merge_research_results([], results)


def detect_outdated_data(results: list[ResearchResult]) -> CriticReport:
    cutoff = datetime.now(timezone.utc) - timedelta(days=183)
    findings: list[CriticFinding] = []
    for result in results:
        detected_date = _latest_date_hint(result.content)
        if detected_date is not None and detected_date < cutoff:
            findings.append(
                CriticFinding(
                    finding_type="missing_context",
                    description=(
                        f"Source may be outdated for {result.task_id}; latest "
                        f"detected date is {detected_date.date().isoformat()}."
                    ),
                    affected_tasks=[result.task_id],
                    severity="medium",
                )
            )
    return CriticReport(
        passed=not findings,
        findings=findings,
        recommendation=(
            "Refresh competitor claims with sources from the last 6 months."
            if findings
            else "No outdated source dates detected."
        ),
        iteration=1,
    )


def enhance_competitor_report(
    report: FinalReport,
    request: CompetitorAnalysisInput,
    research_results: list[ResearchResult],
) -> FinalReport:
    citations = [
        citation_for_result(index, result)
        for index, result in enumerate(research_results, start=1)
    ]
    comparison = _comparison_table(request, citations)
    swot = _swot_sections(request, citations)
    recommendations = _recommendation_section(request, citations)
    sections = [comparison, *swot, *report.sections, recommendations]
    citation_map = {citation.index: citation for citation in [*report.all_citations, *citations]}
    return report.model_copy(
        update={
            "sections": sections,
            "all_citations": list(citation_map.values()),
        }
    )


def _comparison_table(
    request: CompetitorAnalysisInput,
    citations: list[Citation],
) -> ReportSection:
    citation = citations[0] if citations else _fallback_citation()
    header = "| Competitor | " + " | ".join(area.title() for area in request.focus_areas) + " |\n"
    separator = "|" + "|".join(["---"] * (len(request.focus_areas) + 1)) + "|\n"
    rows = []
    for competitor in request.competitors:
        cells = [competitor, *[f"Evidence gathered for {area} [{citation.index}]" for area in request.focus_areas]]
        rows.append("| " + " | ".join(cells) + " |")
    return ReportSection(
        title="Competitor Comparison Table",
        content=header + separator + "\n".join(rows),
        citations=[citation],
    )


def _swot_sections(
    request: CompetitorAnalysisInput,
    citations: list[Citation],
) -> list[ReportSection]:
    citation = citations[0] if citations else _fallback_citation()
    sections: list[ReportSection] = []
    for competitor in request.competitors:
        sections.append(
            ReportSection(
                title=f"{competitor} SWOT",
                content=(
                    f"- Strengths: Visible traction or positioning signals from collected sources [{citation.index}].\n"
                    f"- Weaknesses: Gaps should be validated against pricing, reviews, and product evidence [{citation.index}].\n"
                    f"- Opportunities: {request.your_company} can exploit underserved customer needs [{citation.index}].\n"
                    f"- Threats: {competitor} may respond through product, pricing, or funding moves [{citation.index}]."
                ),
                citations=[citation],
            )
        )
    return sections


def _recommendation_section(
    request: CompetitorAnalysisInput,
    citations: list[Citation],
) -> ReportSection:
    citation = citations[0] if citations else _fallback_citation()
    return ReportSection(
        title="Strategic Recommendations",
        content=(
            f"{request.your_company} should prioritize differentiation in the focus areas "
            f"where competitors show repeated strengths, then validate positioning with "
            f"fresh customer and pricing evidence [{citation.index}]."
        ),
        citations=[citation],
    )


def _latest_date_hint(content: str) -> datetime | None:
    iso_dates = []
    for match in re.findall(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", content):
        year, month, day = (int(value) for value in match)
        try:
            iso_dates.append(datetime(year, month, day, tzinfo=timezone.utc))
        except ValueError:
            continue
    years = [
        datetime(int(year), 1, 1, tzinfo=timezone.utc)
        for year in re.findall(r"\b(20\d{2})\b", content)
    ]
    candidates = [*iso_dates, *years]
    return max(candidates) if candidates else None


def _fallback_citation() -> Citation:
    return Citation(
        index=1,
        url="https://example.com/no-source",
        title="No source available",
        quote="No source content was available.",
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def parse_args() -> CompetitorAnalysisInput:
    parser = argparse.ArgumentParser(description="Run a Nexus competitor analysis workflow.")
    parser.add_argument("--company", required=True, help="Your company name.")
    parser.add_argument("--competitors", required=True, help="Comma-separated competitors.")
    parser.add_argument(
        "--focus",
        default=",".join(DEFAULT_FOCUS_AREAS),
        help="Comma-separated focus areas.",
    )
    args = parser.parse_args()
    return CompetitorAnalysisInput(
        your_company=args.company,
        competitors=[item.strip() for item in args.competitors.split(",") if item.strip()],
        focus_areas=[item.strip() for item in args.focus.split(",") if item.strip()],
    )


async def async_main() -> None:
    report = await run_competitor_analysis(parse_args())
    print_report(report)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
