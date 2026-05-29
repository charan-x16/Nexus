import argparse
import asyncio

from pydantic import BaseModel, Field

from backend.schemas.workflow import FinalReport, ResearchTask, WorkflowPlan
from workflows.common import print_report, run_preplanned_pipeline


class MarketResearchInput(BaseModel):
    company_name: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    geography: str = Field(min_length=1)


REQUIRED_SECTIONS = [
    "Market Overview",
    "Key Players",
    "Market Trends",
    "Regulatory Landscape",
    "Opportunities & Risks",
]


def build_goal(request: MarketResearchInput) -> str:
    return (
        f"Conduct comprehensive market research for {request.company_name} "
        f"in the {request.industry} industry in {request.geography}. "
        "The final report must include sections titled: "
        + ", ".join(REQUIRED_SECTIONS)
        + "."
    )


def build_market_research_plan(request: MarketResearchInput) -> WorkflowPlan:
    company = request.company_name
    industry = request.industry
    geography = request.geography
    return WorkflowPlan(
        title=f"{company} {industry} Market Research",
        goal=build_goal(request),
        subtasks=[
            ResearchTask(
                id="market-size",
                description=f"Estimate market size, growth rate, and demand drivers for {industry} in {geography}.",
                search_queries=[
                    f"{industry} market size {geography} latest report",
                    f"{industry} growth forecast {geography} market research",
                    f"{company} {industry} market opportunity {geography}",
                ],
                priority=1,
            ),
            ResearchTask(
                id="key-players",
                description=f"Identify leading competitors, adjacent players, and ecosystem partners for {industry} in {geography}.",
                search_queries=[
                    f"top {industry} companies {geography}",
                    f"{industry} competitive landscape {geography}",
                    f"{company} competitors {industry} {geography}",
                ],
                priority=2,
            ),
            ResearchTask(
                id="market-trends",
                description=f"Find current and emerging trends shaping {industry} adoption in {geography}.",
                search_queries=[
                    f"{industry} trends {geography} 2025 2026",
                    f"{industry} customer adoption trends {geography}",
                    f"{industry} technology trends {geography}",
                ],
                priority=3,
            ),
            ResearchTask(
                id="regulatory-environment",
                description=f"Summarize regulations, compliance requirements, and policy changes affecting {industry} in {geography}.",
                search_queries=[
                    f"{industry} regulation {geography}",
                    f"{industry} compliance requirements {geography}",
                    f"{industry} policy changes {geography} latest",
                ],
                priority=4,
            ),
            ResearchTask(
                id="customer-segments",
                description=f"Identify target customer segments, buying criteria, and unmet needs for {company} in {industry}.",
                search_queries=[
                    f"{industry} customer segments {geography}",
                    f"{industry} buyer needs {geography}",
                    f"{company} target customers {industry}",
                ],
                priority=5,
            ),
        ],
    )


async def run_market_research(request: MarketResearchInput) -> FinalReport:
    goal = build_goal(request)
    plan = build_market_research_plan(request)
    return await run_preplanned_pipeline(goal=goal, plan=plan)


def parse_args() -> MarketResearchInput:
    parser = argparse.ArgumentParser(description="Run a Nexus market research workflow.")
    parser.add_argument("--company", required=True, help="Company name.")
    parser.add_argument("--industry", required=True, help="Industry to research.")
    parser.add_argument("--geo", required=True, help="Geography to focus on.")
    args = parser.parse_args()
    return MarketResearchInput(
        company_name=args.company,
        industry=args.industry,
        geography=args.geo,
    )


async def async_main() -> None:
    report = await run_market_research(parse_args())
    print_report(report)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
