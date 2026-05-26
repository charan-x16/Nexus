import json
from typing import Any

from langsmith import traceable

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import AgentMessage, TaskPlan, WorkflowState

PLANNER_SYSTEM_PROMPT = (
    "You are an expert project planner that decomposes user goals into "
    "structured research and writing tasks. Return concise, valid JSON with "
    "the keys title, description, subtasks, and estimated_steps."
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
            [
                {
                    "role": "user",
                    "content": (
                        "Create a project plan for this user goal. "
                        "Return only a JSON object with this shape: "
                        '{"title": "string", "description": "string", '
                        '"subtasks": ["string"], "estimated_steps": 1}.\n\n'
                        f"Goal: {goal}"
                    ),
                }
            ],
            max_tokens=1200,
            temperature=0.1,
        )

        plan = TaskPlan.model_validate(_parse_json_object(response_text))
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
        updated_state["status"] = "planned"
        return updated_state


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
