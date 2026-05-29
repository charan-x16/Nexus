import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from backend.agents.base import BaseAgent
from backend.agents.critic import CriticAgent
from backend.agents.planner import PlannerAgent
from backend.agents.writer import WriterAgent
from backend.config import settings
from backend.observability.token_tracker import token_tracker
from backend.schemas.workflow import (
    CriticReport,
    FinalReport,
    ResearchResult,
    ResearchTask,
    WorkflowPlan,
    WorkflowState,
)
from workflows.common import run_preplanned_pipeline


DATASET_DIR = Path(__file__).resolve().parent / "datasets"


class EvalScorer(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=(
                "You are an impartial evaluator. Return strict JSON only with "
                "numeric scores and short rationales."
            ),
        )

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            raise ValueError("EvalScorer.run requires evaluator inputs.")
        return await self.score_planner(*args, **kwargs)

    async def score_planner(self, goal: str, expected: list[str], plan: dict[str, Any]) -> float:
        response = await self._call_model(
            [
                {
                    "role": "user",
                    "content": (
                        "Score this workflow plan from 1 to 10 on completeness, "
                        "specificity, and logical ordering. Return JSON like "
                        '{"score": 8.0, "rationale": "..."}.'
                        f"\n\nGoal: {goal}\nExpected task themes: {expected}\n"
                        f"Plan: {json.dumps(plan, indent=2)}"
                    ),
                }
            ],
            max_tokens=300,
            temperature=0.0,
        )
        try:
            parsed = json.loads(response)
            return float(parsed.get("score", 0))
        except Exception:
            return 0.0


async def planner_run_fn(inputs: dict[str, Any]) -> dict[str, Any]:
    state: WorkflowState = {
        "goal": inputs["goal"],
        "messages": [],
        "status": "planning",
    }
    result = await PlannerAgent().run(state)
    plan = result.get("plan")
    return plan.model_dump(mode="json") if isinstance(plan, WorkflowPlan) else plan


async def planner_accuracy_eval() -> dict[str, Any]:
    dataset = _load_json("planner_dataset.json")
    scorer = EvalScorer()
    scores = []
    for example in dataset:
        plan = await planner_run_fn({"goal": example["goal"]})
        score = await scorer.score_planner(
            goal=example["goal"],
            expected=example["expected_tasks"],
            plan=plan,
        )
        scores.append(score)
    average = sum(scores) / len(scores)
    return {"average_score": average, "passed": average >= 7.0, "scores": scores}


async def critic_recall_eval() -> dict[str, Any]:
    dataset = _load_json("critic_dataset.json")
    recalls = []
    precisions = []
    for bundle in dataset:
        report = await _run_critic_bundle(bundle)
        found_contradictions = sum(
            1 for finding in report.findings if finding.finding_type == "contradiction"
        )
        total_findings = max(1, len(report.findings))
        planted = max(1, int(bundle["planted_contradictions"]))
        recalls.append(min(1.0, found_contradictions / planted))
        precisions.append(found_contradictions / total_findings)
    recall = sum(recalls) / len(recalls)
    precision = sum(precisions) / len(precisions)
    return {
        "precision": precision,
        "recall": recall,
        "passed": recall >= 0.8,
        "bundle_count": len(dataset),
    }


async def writer_citation_eval() -> dict[str, Any]:
    bundles = _load_json("critic_dataset.json")[:5]
    valid = 0
    reports: list[FinalReport] = []
    for bundle in bundles:
        state = _bundle_state(bundle)
        report = await WriterAgent().run(state)
        reports.append(report)
        source_urls = {result.url for result in state["research_results"]}
        section_citations_ok = all(section.citations for section in report.sections)
        citation_urls_ok = all(citation.url in source_urls for citation in report.all_citations)
        summary_words = len(report.executive_summary.split())
        summary_ok = 100 <= summary_words <= 300
        if section_citations_ok and citation_urls_ok and summary_ok:
            valid += 1
    validity = valid / len(bundles)
    return {"citation_validity": validity, "passed": validity == 1.0, "reports": len(reports)}


async def end_to_end_eval() -> dict[str, Any]:
    goals = [
        "Research pgvector adoption in enterprise search.",
        "Analyze AI agent orchestration frameworks for knowledge work.",
        "Create a concise market brief on customer support automation.",
    ]
    run_results = []
    for index, goal in enumerate(goals, start=1):
        start = time.perf_counter()
        plan = WorkflowPlan(
            title=f"Evaluation Plan {index}",
            goal=goal,
            subtasks=[
                ResearchTask(
                    id=f"eval-{index}-overview",
                    description=f"Find authoritative evidence for: {goal}",
                    search_queries=[goal, f"{goal} latest analysis"],
                    priority=1,
                ),
                ResearchTask(
                    id=f"eval-{index}-risks",
                    description=f"Find risks, limitations, and counterpoints for: {goal}",
                    search_queries=[f"{goal} risks", f"{goal} limitations"],
                    priority=2,
                ),
            ],
        )
        report = await run_preplanned_pipeline(goal=goal, plan=plan)
        elapsed = time.perf_counter() - start
        run_results.append(
            {
                "goal": goal,
                "elapsed_seconds": elapsed,
                "completed_under_5_minutes": elapsed <= 300,
                "final_report_present": report is not None,
                "confidence_score": report.confidence_score,
                "passed": elapsed <= 300 and report.confidence_score > 0.4,
                "estimated_cost_usd": str(
                    token_tracker.calculate_cost(
                        settings.OPENROUTER_MODEL,
                        input_tokens=8000,
                        output_tokens=3000,
                    )
                ),
            }
        )
    return {
        "runs": run_results,
        "passed": all(item["passed"] for item in run_results),
    }


def run_langsmith_registered_eval() -> None:
    from langsmith.evaluation import evaluate

    evaluate(
        planner_run_fn,
        data="planner-dataset",
        evaluators=[planner_evaluator],
        experiment_prefix="nexus-planner-accuracy",
    )


def planner_evaluator(run: Any, example: Any) -> dict[str, Any]:
    outputs = getattr(run, "outputs", {}) or {}
    expected = getattr(example, "outputs", {}) or {}
    plan_text = json.dumps(outputs)
    expected_terms = expected.get("expected_tasks", [])
    matched = sum(1 for term in expected_terms if term.lower() in plan_text.lower())
    score = matched / max(1, len(expected_terms))
    return {"key": "planner_task_coverage", "score": score}


async def _run_critic_bundle(bundle: dict[str, Any]) -> CriticReport:
    state = _bundle_state(bundle)
    state["critic_iteration"] = 1
    return await CriticAgent().run(state)


def _bundle_state(bundle: dict[str, Any]) -> WorkflowState:
    results = [ResearchResult.model_validate(item) for item in bundle["research_results"]]
    task_ids = sorted({result.task_id for result in results})
    plan = WorkflowPlan(
        title=f"Evaluation: {bundle['goal']}",
        goal=bundle["goal"],
        subtasks=[
            ResearchTask(
                id=task_id,
                description=f"Evaluate evidence for {task_id}",
                search_queries=[bundle["goal"]],
                priority=index,
            )
            for index, task_id in enumerate(task_ids, start=1)
        ],
    )
    return {
        "goal": bundle["goal"],
        "plan": plan,
        "research_results": results,
        "critic_reports": [],
        "critic_iteration": 0,
        "memory_context": "",
        "messages": [],
        "status": "evaluating",
        "awaiting_approval": False,
    }


def _load_json(filename: str) -> Any:
    return json.loads((DATASET_DIR / filename).read_text(encoding="utf-8"))


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run Nexus LangSmith evaluation suites.")
    parser.add_argument(
        "--suite",
        choices=["planner", "critic", "writer", "e2e", "langsmith", "all"],
        default="all",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    if args.suite in {"planner", "all"}:
        results["planner_accuracy"] = await planner_accuracy_eval()
    if args.suite in {"critic", "all"}:
        results["critic_recall"] = await critic_recall_eval()
    if args.suite in {"writer", "all"}:
        results["writer_citation"] = await writer_citation_eval()
    if args.suite in {"e2e", "all"}:
        results["end_to_end"] = await end_to_end_eval()
    if args.suite == "langsmith":
        run_langsmith_registered_eval()
        results["langsmith"] = {"submitted": True}
    print(json.dumps(results, indent=2, default=str))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
