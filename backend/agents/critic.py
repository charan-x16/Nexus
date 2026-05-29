import json
from typing import Any

from langsmith import traceable
from pydantic import ValidationError

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import (
    CriticFinding,
    CriticReport,
    ResearchResult,
    WorkflowPlan,
    WorkflowState,
)

CRITIC_SYSTEM_PROMPT = (
    "You are a rigorous research critic. Your job is to identify: "
    "(1) contradictions between sources, (2) claims lacking evidence, "
    "(3) missing critical context, (4) unverified or suspicious claims. "
    "You are thorough but fair. You output structured JSON only."
)


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )

    @traceable(name="CriticAgent.run")
    async def run(self, state: WorkflowState) -> CriticReport:
        iteration = max(1, int(state.get("critic_iteration", 1) or 1))
        response_text = await self._call_model(
            [_critic_prompt(state, iteration)],
            max_tokens=1800,
            temperature=0.0,
            run_id=state.get("run_id"),
        )
        report = await self._parse_with_single_retry(state, response_text, iteration)
        report = _normalize_report(report)

        reports = list(state.get("critic_reports", []))
        reports.append(report)
        state["critic_reports"] = reports
        return report

    async def _parse_with_single_retry(
        self,
        state: WorkflowState,
        response_text: str,
        iteration: int,
    ) -> CriticReport:
        try:
            return CriticReport.model_validate(_parse_json_object(response_text))
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            retry_text = await self._call_model(
                [
                    _critic_prompt(state, iteration),
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not match the required "
                            "CriticReport JSON schema. Return corrected JSON only. "
                            f"Parser error: {exc}"
                        ),
                    },
                ],
                max_tokens=1800,
                temperature=0.0,
                run_id=state.get("run_id"),
            )
            return CriticReport.model_validate(_parse_json_object(retry_text))


def _critic_prompt(state: WorkflowState, iteration: int) -> dict[str, str]:
    plan_value = state.get("plan")
    plan = (
        plan_value
        if isinstance(plan_value, WorkflowPlan)
        else WorkflowPlan.model_validate(plan_value)
    )
    research_results = [
        item if isinstance(item, ResearchResult) else ResearchResult.model_validate(item)
        for item in state.get("research_results", [])
    ]

    research_blocks = []
    for index, result in enumerate(research_results, start=1):
        research_blocks.append(
            "\n".join(
                [
                    f"RESULT {index}",
                    f"TASK_ID: {result.task_id}",
                    f"QUERY: {result.query}",
                    f"URL: {result.url}",
                    f"TITLE: {result.title}",
                    f"RELEVANCE_SCORE: {result.relevance_score}",
                    f"CONTENT: {result.content[:2200]}",
                ]
            )
        )

    return {
        "role": "user",
        "content": (
            f"GOAL: {state.get('goal', '')}\n\n"
            f"PLAN: {plan.model_dump_json(indent=2)}\n\n"
            "RESEARCH RESULTS:\n"
            f"{chr(10).join(research_blocks) if research_blocks else 'No research results.'}\n\n"
            "Identify all issues. Return JSON matching this schema exactly:\n"
            "{\n"
            '  "passed": true,\n'
            '  "findings": [\n'
            "    {\n"
            '      "finding_type": "contradiction|weak_evidence|missing_context|unverified_claim",\n'
            '      "description": "specific issue",\n'
            '      "affected_tasks": ["task-1"],\n'
            '      "severity": "low|medium|high"\n'
            "    }\n"
            "  ],\n"
            '  "recommendation": "what to do next",\n'
            f'  "iteration": {iteration}\n'
            "}\n\n"
            "If there are no high-severity findings, set passed=true. "
            "If there is a high-severity contradiction, missing critical context, "
            "or unverified central claim, set passed=false."
        ),
    }


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


def _normalize_report(report: CriticReport) -> CriticReport:
    has_high_severity = any(finding.severity == "high" for finding in report.findings)
    if has_high_severity and report.passed:
        return report.model_copy(update={"passed": False})
    if not has_high_severity and not report.passed:
        return report.model_copy(update={"passed": True})
    return report
