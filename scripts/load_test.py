import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import aiohttp


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@dataclass
class RunTiming:
    goal: str
    run_id: str
    first_response_seconds: float
    completion_seconds: float
    status: str
    cost_usd: float
    total_tokens: int


async def submit_workflow(
    session: aiohttp.ClientSession,
    base_url: str,
    goal: str,
    index: int,
) -> RunTiming:
    start = time.perf_counter()
    async with session.post(
        f"{base_url}/workflows",
        json={"project_name": f"Load Test {index}", "goal": goal},
    ) as response:
        response.raise_for_status()
        created = await response.json()
    first_response = time.perf_counter() - start
    run_id = created["run_id"]

    async with session.post(f"{base_url}/workflows/{run_id}/approve") as response:
        response.raise_for_status()
        await response.json()

    status = "unknown"
    while time.perf_counter() - start < 1800:
        await asyncio.sleep(5)
        async with session.get(f"{base_url}/workflows/{run_id}/status") as response:
            response.raise_for_status()
            payload = await response.json()
            status = payload["status"]
        if status in {"completed", "failed", "cancelled", "rejected"}:
            break

    cost_usd = 0.0
    total_tokens = 0
    async with session.get(f"{base_url}/observability/runs/{run_id}/tokens") as response:
        if response.status < 400:
            token_payload = await response.json()
            cost_usd = float(token_payload.get("total_cost", 0) or 0)
            total_tokens = int(token_payload.get("total_tokens", 0) or 0)

    return RunTiming(
        goal=goal,
        run_id=run_id,
        first_response_seconds=first_response,
        completion_seconds=time.perf_counter() - start,
        status=status,
        cost_usd=cost_usd,
        total_tokens=total_tokens,
    )


async def run_load_test(goals: list[str], base_url: str) -> list[RunTiming]:
    timeout = aiohttp.ClientTimeout(total=1900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *(
                submit_workflow(session, base_url, goal, index)
                for index, goal in enumerate(goals, start=1)
            )
        )


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile_value / 100) * (len(ordered) - 1)))
    return ordered[index]


def report(results: list[RunTiming]) -> dict[str, Any]:
    completion_latencies = [item.completion_seconds for item in results]
    first_response_latencies = [item.first_response_seconds for item in results]
    failed = [item for item in results if item.status != "completed"]
    return {
        "runs": [item.__dict__ for item in results],
        "time_to_first_response": {
            "p50": statistics.median(first_response_latencies),
            "p95": percentile(first_response_latencies, 95),
            "p99": percentile(first_response_latencies, 99),
        },
        "time_to_completion": {
            "p50": statistics.median(completion_latencies),
            "p95": percentile(completion_latencies, 95),
            "p99": percentile(completion_latencies, 99),
        },
        "total_tokens": sum(item.total_tokens for item in results),
        "total_cost_usd": round(sum(item.cost_usd for item in results), 6),
        "failed_runs": [item.run_id for item in failed],
        "passed": not failed,
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run a concurrent Nexus workflow load test.")
    parser.add_argument("--base-url", default=API_BASE_URL)
    args = parser.parse_args()
    goals = [
        "Create a concise market brief on AI coding assistants.",
        "Research vector database selection criteria for startups.",
        "Analyze customer support automation trends.",
        "Compare agent orchestration frameworks for enterprise teams.",
        "Assess AI governance requirements for SaaS companies.",
    ]
    results = await run_load_test(goals, args.base_url)
    summary = report(results)
    for key, value in summary.items():
        print(f"{key}: {value}")
    if not summary["passed"]:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
