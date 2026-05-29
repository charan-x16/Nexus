import json
from typing import Any

from langsmith import traceable
from pydantic import ValidationError

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import AgentMessage, WorkflowPlan, WorkflowState

PLANNER_SYSTEM_PROMPT = (
    "You are an expert research planner for knowledge work. Decompose user "
    "goals into prioritized research subtasks. Return valid JSON only. Do "
    "not include markdown, commentary, or code fences."
)


class PlannerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    @traceable(name="PlannerAgent.run")
    async def run(self, state: WorkflowState) -> WorkflowState:
        goal = state.get("goal", "").strip()
        if not goal:
            raise ValueError("Workflow goal is required.")

        response_text = await self._call_model(
            [_planner_prompt(goal)],
            max_tokens=1800,
            temperature=0.1,
            run_id=state.get("run_id"),
        )
        plan = await self._parse_with_single_retry(
            goal=goal,
            response_text=response_text,
            run_id=state.get("run_id"),
        )

        messages = list(state.get("messages", []))
        messages.append(
            AgentMessage(
                agent="planner",
                role="assistant",
                content=plan.model_dump_json(),
            )
        )

        updated_state: WorkflowState = dict(state)
        updated_state["plan"] = plan
        updated_state["messages"] = messages
        updated_state["awaiting_approval"] = True
        updated_state["status"] = "awaiting_approval"
        return updated_state

    async def _parse_with_single_retry(
        self,
        goal: str,
        response_text: str,
        run_id: str | None,
    ) -> WorkflowPlan:
        try:
            return WorkflowPlan.model_validate(_parse_json_object(response_text))
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            retry_text = await self._call_model(
                [
                    _planner_prompt(goal),
                    {
                        "role": "user",
                        "content": (
                            "The previous response could not be parsed as the "
                            "required WorkflowPlan JSON. Return corrected JSON "
                            f"only. Parser error: {exc}"
                        ),
                    },
                ],
                max_tokens=1800,
                temperature=0.0,
                run_id=run_id,
            )
            return WorkflowPlan.model_validate(_parse_json_object(retry_text))


def _planner_prompt(goal: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Create a research workflow plan for the user goal. Return only "
            "JSON matching this schema:\n"
            "{\n"
            '  "title": "short plan title",\n'
            '  "goal": "original user goal",\n'
            '  "subtasks": [\n'
            "    {\n"
            '      "id": "task-1",\n'
            '      "description": "specific research task",\n'
            '      "search_queries": ["precise web search query"],\n'
            '      "priority": 1,\n'
            '      "status": "pending"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Create 3 to 6 subtasks.\n"
            "- Each subtask must have 2 to 4 search queries.\n"
            "- Priority 1 is highest priority.\n"
            "- Use stable ids like task-1, task-2, task-3.\n"
            "- Set every status to pending.\n\n"
            f"User goal: {goal}"
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
