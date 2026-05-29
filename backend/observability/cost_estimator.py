from decimal import Decimal

from pydantic import BaseModel, Field

from backend.schemas.workflow import WorkflowPlan


class CostEstimate(BaseModel):
    min_usd: Decimal = Field(ge=0)
    max_usd: Decimal = Field(ge=0)
    estimated_usd: Decimal = Field(ge=0)
    breakdown_by_agent: dict[str, Decimal]


SONNET_INPUT_PER_1K = Decimal("0.003")
SONNET_OUTPUT_PER_1K = Decimal("0.015")


def estimate_workflow_cost(plan: WorkflowPlan) -> CostEstimate:
    task_count = len(plan.subtasks)
    token_breakdown = {
        "planner": 2000,
        "research": 3000 * task_count,
        "critic": 4000,
        "writer": 5000,
    }
    max_token_breakdown = token_breakdown | {"critic": 4000 * 3}
    breakdown_by_agent = {
        agent: _cost_for_total_tokens(tokens)
        for agent, tokens in token_breakdown.items()
    }
    estimated = sum(breakdown_by_agent.values(), Decimal("0")).quantize(
        Decimal("0.000001")
    )
    min_cost = (estimated * Decimal("0.70")).quantize(Decimal("0.000001"))
    max_cost = sum(
        (_cost_for_total_tokens(tokens) for tokens in max_token_breakdown.values()),
        Decimal("0"),
    ).quantize(Decimal("0.000001"))
    return CostEstimate(
        min_usd=min_cost,
        max_usd=max_cost,
        estimated_usd=estimated,
        breakdown_by_agent=breakdown_by_agent,
    )


def _cost_for_total_tokens(total_tokens: int) -> Decimal:
    input_tokens = Decimal(total_tokens) * Decimal("0.65")
    output_tokens = Decimal(total_tokens) * Decimal("0.35")
    cost = (
        input_tokens / Decimal(1000) * SONNET_INPUT_PER_1K
        + output_tokens / Decimal(1000) * SONNET_OUTPUT_PER_1K
    )
    return cost.quantize(Decimal("0.000001"))
