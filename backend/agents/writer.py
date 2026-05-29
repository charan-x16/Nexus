from langsmith import traceable

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import (
    AgentMessage,
    ResearchResult,
    WorkflowPlan,
    WorkflowState,
)

WRITER_SYSTEM_PROMPT = (
    "You are an expert technical writer that synthesises information into "
    "clear structured reports. Write in polished markdown with useful "
    "headings, concrete recommendations, and no invented citations."
)


class WriterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=WRITER_SYSTEM_PROMPT,
        )

    @traceable(name="WriterAgent.run")
    async def run(self, state: WorkflowState) -> WorkflowState:
        goal = state.get("goal", "").strip()
        plan_value = state.get("plan")
        if plan_value is None:
            raise ValueError("WriterAgent requires a plan in workflow state.")

        plan = (
            plan_value
            if isinstance(plan_value, WorkflowPlan)
            else WorkflowPlan.model_validate(plan_value)
        )
        research_values = state.get("research_results", [])
        research_results = [
            item if isinstance(item, ResearchResult) else ResearchResult.model_validate(item)
            for item in research_values
        ]
        research_context = _format_research_context(research_results)
        memory_context = state.get("memory_context", "")

        final_output = await self._call_model(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a final markdown report for the user goal using "
                        "the plan and available research context.\n\n"
                        f"Goal:\n{goal}\n\n"
                        f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
                        f"Relevant project memory:\n{memory_context or 'No stored memory was found.'}\n\n"
                        f"Research context:\n{research_context}"
                    ),
                }
            ],
            max_tokens=2500,
            temperature=0.2,
        )

        messages = list(state.get("messages", []))
        messages.append(
            AgentMessage(
                agent="writer",
                role="assistant",
                content=final_output,
            )
        )

        updated_state: WorkflowState = dict(state)
        updated_state["plan"] = plan
        updated_state["final_output"] = final_output
        updated_state["messages"] = messages
        updated_state["status"] = "completed"
        return updated_state


def _format_research_context(research_results: list[ResearchResult]) -> str:
    if not research_results:
        return "- No research results were collected."

    lines: list[str] = []
    for index, result in enumerate(research_results[:12], start=1):
        lines.append(
            "\n".join(
                [
                    f"{index}. {result.title or result.url}",
                    f"   URL: {result.url}",
                    f"   Query: {result.query}",
                    f"   Relevance: {result.relevance_score}/10",
                    f"   Excerpt: {result.content[:900]}",
                ]
            )
        )
    return "\n\n".join(lines)
