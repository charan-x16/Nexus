import json
import re
from datetime import datetime, timezone
from typing import Any

from langsmith import traceable
from pydantic import ValidationError

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import (
    Citation,
    CriticReport,
    FinalReport,
    ResearchResult,
    ReportSection,
    WorkflowPlan,
    WorkflowState,
)

WRITER_SYSTEM_PROMPT = (
    "You are an expert analyst and writer. Produce structured research reports "
    "with: executive summary, thematic sections, and inline citations. Every "
    "factual claim must cite a source using [N] notation. Output JSON matching "
    "the FinalReport schema exactly."
)


class WriterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=WRITER_SYSTEM_PROMPT,
        )

    @traceable(name="WriterAgent.run")
    async def run(self, state: WorkflowState) -> FinalReport:
        plan = _coerce_plan(state.get("plan"))
        research_results = _coerce_research_results(state.get("research_results", []))
        critic_reports = _coerce_critic_reports(state.get("critic_reports", []))
        response_text = await self._call_model(
            [_writer_prompt(state, plan, research_results, critic_reports)],
            max_tokens=4500,
            temperature=0.15,
        )
        report = await self._parse_with_single_retry(
            state,
            plan,
            research_results,
            critic_reports,
            response_text,
        )
        confidence_score = calculate_confidence_score(research_results, critic_reports)
        return _normalize_report(report, research_results, confidence_score)

    async def _parse_with_single_retry(
        self,
        state: WorkflowState,
        plan: WorkflowPlan,
        research_results: list[ResearchResult],
        critic_reports: list[CriticReport],
        response_text: str,
    ) -> FinalReport:
        try:
            return FinalReport.model_validate(_parse_json_object(response_text))
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            retry_text = await self._call_model(
                [
                    _writer_prompt(state, plan, research_results, critic_reports),
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not match the FinalReport "
                            "JSON schema. Return corrected JSON only. "
                            f"Parser error: {exc}"
                        ),
                    },
                ],
                max_tokens=4500,
                temperature=0.0,
            )
            return FinalReport.model_validate(_parse_json_object(retry_text))


def calculate_confidence_score(
    research_results: list[ResearchResult],
    critic_reports: list[CriticReport],
) -> float:
    if not research_results:
        return 0.0
    top_scores = sorted(
        [result.relevance_score for result in research_results],
        reverse=True,
    )[:5]
    average_relevance = sum(top_scores) / len(top_scores) / 10
    high_severity_count = sum(
        1
        for report in critic_reports
        for finding in report.findings
        if finding.severity == "high"
    )
    penalty = min(0.6, high_severity_count * 0.1)
    return round(max(0.0, min(1.0, average_relevance - penalty)), 2)


def render_final_report_markdown(report: FinalReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"**Confidence:** {report.confidence_score:.2f}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
    ]
    for section in report.sections:
        lines.extend(["", f"## {section.title}", "", section.content])

    if report.all_citations:
        lines.extend(["", "## References"])
        for citation in sorted(report.all_citations, key=lambda item: item.index):
            quote = citation.quote.replace("\n", " ").strip()
            lines.append(
                f"[{citation.index}] {citation.title or citation.url}. "
                f"{citation.url}. \"{quote}\""
            )
    return "\n".join(lines).strip()


def _writer_prompt(
    state: WorkflowState,
    plan: WorkflowPlan,
    research_results: list[ResearchResult],
    critic_reports: list[CriticReport],
) -> dict[str, str]:
    source_blocks = []
    for index, result in enumerate(research_results, start=1):
        source_blocks.append(
            "\n".join(
                [
                    f"[{index}] {result.title or result.url}",
                    f"URL: {result.url}",
                    f"TASK_ID: {result.task_id}",
                    f"QUERY: {result.query}",
                    f"RELEVANCE_SCORE: {result.relevance_score}",
                    f"QUOTE_CANDIDATE: {result.content[:700]}",
                ]
            )
        )

    return {
        "role": "user",
        "content": (
            "Create a structured research report. Return only JSON matching "
            "this schema:\n"
            "{\n"
            '  "title": "string",\n'
            '  "executive_summary": "string with [N] citations",\n'
            '  "sections": [\n'
            "    {\n"
            '      "title": "string",\n'
            '      "content": "string with [N] citations",\n'
            '      "citations": [{"index": 1, "url": "string", "title": "string", "quote": "string"}]\n'
            "    }\n"
            "  ],\n"
            '  "all_citations": [{"index": 1, "url": "string", "title": "string", "quote": "string"}],\n'
            '  "confidence_score": 0.0,\n'
            '  "generated_at": "ISO-8601 datetime"\n'
            "}\n\n"
            "Rules:\n"
            "- Use only the numbered sources below.\n"
            "- Every factual claim must include an inline [N] citation.\n"
            "- Every section must include at least one citation.\n"
            "- Citation indexes must match the numbered sources.\n"
            "- Quotes must be short excerpts from the cited source.\n\n"
            f"GOAL:\n{state.get('goal', '')}\n\n"
            f"PLAN:\n{plan.model_dump_json(indent=2)}\n\n"
            f"MEMORY CONTEXT:\n{state.get('memory_context', '') or 'No prior memory context.'}\n\n"
            "CRITIC REPORTS:\n"
            f"{json.dumps([report.model_dump(mode='json') for report in critic_reports], indent=2)}\n\n"
            "NUMBERED SOURCES:\n"
            f"{chr(10).join(source_blocks) if source_blocks else 'No sources available.'}"
        ),
    }


def _normalize_report(
    report: FinalReport,
    research_results: list[ResearchResult],
    confidence_score: float,
) -> FinalReport:
    citations_by_index = {
        index: _citation_from_result(index, result)
        for index, result in enumerate(research_results, start=1)
    }
    normalized_sections: list[ReportSection] = []
    used_indexes: set[int] = set()

    for section in report.sections:
        indexes = set(_citation_indexes(section.content))
        indexes.update(citation.index for citation in section.citations)
        valid_indexes = [index for index in sorted(indexes) if index in citations_by_index]
        if not valid_indexes and citations_by_index:
            first_index = next(iter(citations_by_index))
            valid_indexes = [first_index]
            if f"[{first_index}]" not in section.content:
                section_content = f"{section.content.rstrip()} [{first_index}]"
            else:
                section_content = section.content
        else:
            section_content = section.content

        section_citations = [citations_by_index[index] for index in valid_indexes]
        used_indexes.update(valid_indexes)
        normalized_sections.append(
            ReportSection(
                title=section.title,
                content=section_content,
                citations=section_citations,
            )
        )

    for citation in report.all_citations:
        if citation.index in citations_by_index:
            used_indexes.add(citation.index)

    all_citations = [citations_by_index[index] for index in sorted(used_indexes)]
    if not all_citations and citations_by_index:
        all_citations = [next(iter(citations_by_index.values()))]

    return FinalReport(
        title=report.title,
        executive_summary=report.executive_summary,
        sections=normalized_sections,
        all_citations=all_citations,
        confidence_score=confidence_score,
        generated_at=report.generated_at or datetime.now(timezone.utc),
    )


def _citation_from_result(index: int, result: ResearchResult) -> Citation:
    quote = result.content.strip().replace("\n", " ")[:260]
    return Citation(
        index=index,
        url=result.url,
        title=result.title or result.url,
        quote=quote or "Source content was collected during research.",
    )


def _citation_indexes(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", text)]


def _coerce_plan(value: Any) -> WorkflowPlan:
    if value is None:
        raise ValueError("WriterAgent requires a plan in workflow state.")
    return value if isinstance(value, WorkflowPlan) else WorkflowPlan.model_validate(value)


def _coerce_research_results(value: Any) -> list[ResearchResult]:
    if not isinstance(value, list):
        return []
    return [
        item if isinstance(item, ResearchResult) else ResearchResult.model_validate(item)
        for item in value
    ]


def _coerce_critic_reports(value: Any) -> list[CriticReport]:
    if not isinstance(value, list):
        return []
    return [
        item if isinstance(item, CriticReport) else CriticReport.model_validate(item)
        for item in value
    ]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("The model did not return a JSON object.") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model returned JSON, but it was not an object.")
    return parsed
