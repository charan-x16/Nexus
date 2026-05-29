from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.db.connection import execute_query, fetch_rows


class TokenUsage(BaseModel):
    agent_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    COST_PER_1K = {
        "claude-sonnet-4-20250514": {"input": Decimal("0.003"), "output": Decimal("0.015")},
        "anthropic/claude-sonnet-4": {"input": Decimal("0.003"), "output": Decimal("0.015")},
    }

    async def record(
        self,
        run_id: str | UUID | None,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        cost_usd = self.calculate_cost(model, input_tokens, output_tokens)
        if not run_id:
            return cost_usd

        await execute_query(
            """
            INSERT INTO token_usage (
                run_id,
                agent_name,
                model,
                input_tokens,
                output_tokens,
                cost_usd
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            UUID(str(run_id)),
            agent_name,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )
        return cost_usd

    async def get_run_summary(self, run_id: str | UUID) -> dict[str, Any]:
        rows = await fetch_rows(
            """
            SELECT
                agent_name,
                COALESCE(SUM(input_tokens), 0)::INT AS input_tokens,
                COALESCE(SUM(output_tokens), 0)::INT AS output_tokens,
                COALESCE(SUM(cost_usd), 0)::NUMERIC(10, 6) AS cost_usd
            FROM token_usage
            WHERE run_id = $1
            GROUP BY agent_name
            ORDER BY agent_name
            """,
            UUID(str(run_id)),
        )
        by_agent = [
            {
                "agent_name": row["agent_name"],
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["input_tokens"] or 0)
                + int(row["output_tokens"] or 0),
                "cost_usd": row["cost_usd"] or Decimal("0"),
            }
            for row in rows
        ]
        total_input = sum(item["input_tokens"] for item in by_agent)
        total_output = sum(item["output_tokens"] for item in by_agent)
        total_cost = sum((Decimal(str(item["cost_usd"])) for item in by_agent), Decimal("0"))
        return {
            "run_id": UUID(str(run_id)),
            "total_input": total_input,
            "total_output": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": total_cost.quantize(Decimal("0.000001")),
            "by_agent": by_agent,
        }

    async def get_project_summary(self, project_id: str | UUID) -> dict[str, Any]:
        rows = await fetch_rows(
            """
            SELECT
                COUNT(DISTINCT workflow_runs.id)::INT AS total_runs,
                COALESCE(SUM(token_usage.cost_usd), 0)::NUMERIC(10, 6) AS total_cost
            FROM workflow_runs
            LEFT JOIN token_usage ON token_usage.run_id = workflow_runs.id
            WHERE workflow_runs.project_id = $1
            """,
            UUID(str(project_id)),
        )
        row = rows[0] if rows else None
        total_runs = int(row["total_runs"] or 0) if row else 0
        total_cost = Decimal(str(row["total_cost"] or "0")) if row else Decimal("0")
        avg_cost = total_cost / Decimal(total_runs) if total_runs else Decimal("0")
        return {
            "project_id": UUID(str(project_id)),
            "total_runs": total_runs,
            "total_cost": total_cost.quantize(Decimal("0.000001")),
            "avg_cost_per_run": avg_cost.quantize(Decimal("0.000001")),
        }

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        rates = self._rates_for_model(model)
        input_cost = Decimal(max(0, input_tokens)) / Decimal(1000) * rates["input"]
        output_cost = Decimal(max(0, output_tokens)) / Decimal(1000) * rates["output"]
        return (input_cost + output_cost).quantize(Decimal("0.000001"))

    def _rates_for_model(self, model: str) -> dict[str, Decimal]:
        if model in self.COST_PER_1K:
            return self.COST_PER_1K[model]
        for known_model, rates in self.COST_PER_1K.items():
            if known_model in model:
                return rates
        return {"input": Decimal("0"), "output": Decimal("0")}


token_tracker = TokenTracker()


def track_token_usage(usage: TokenUsage) -> None:
    print(
        "[token_usage] "
        f"agent={usage.agent_name} "
        f"model={usage.model_name} "
        f"input={usage.input_tokens} "
        f"output={usage.output_tokens} "
        f"total={usage.total_tokens}"
    )
