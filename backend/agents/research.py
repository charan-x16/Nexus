import asyncio
import json
import re
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
from langsmith import traceable

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.schemas.workflow import (
    CriticFinding,
    ResearchResult,
    ResearchTask,
    SearchResult,
)

try:
    from tavily import AsyncTavilyClient
except ImportError:  # pragma: no cover - dependency is installed from requirements.
    AsyncTavilyClient = None  # type: ignore[assignment]

try:
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover - dependency is installed from requirements.
    async_playwright = None  # type: ignore[assignment]

RESEARCH_SYSTEM_PROMPT = (
    "You are a careful research analyst. Score source relevance to the "
    "user goal with a single number from 1 to 10. Return only the number."
)


class ResearchAgent(BaseAgent):
    def __init__(self, goal: str = "", memory_context: str = "") -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=RESEARCH_SYSTEM_PROMPT,
        )
        self.goal = goal
        self.memory_context = memory_context

    async def tavily_search(self, query: str) -> list[SearchResult]:
        api_key = (
            settings.TAVILY_API_KEY.get_secret_value()
            if settings.TAVILY_API_KEY is not None
            else None
        )
        if not api_key:
            return []
        if AsyncTavilyClient is None:
            raise RuntimeError("tavily-python is required for Tavily search.")

        client = AsyncTavilyClient(api_key=api_key)
        for attempt in range(4):
            try:
                response = await client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                )
                raw_results = response.get("results", []) if isinstance(response, dict) else []
                return [
                    SearchResult(
                        url=str(item.get("url", "")),
                        title=str(item.get("title", "")),
                        content=str(item.get("content", "")),
                    )
                    for item in raw_results[:5]
                    if item.get("url")
                ]
            except Exception as exc:
                if attempt < 3 and _is_retryable_tavily_error(exc):
                    await asyncio.sleep(2**attempt)
                    continue
                return []
        return []

    async def scrape_page(self, url: str) -> str:
        try:
            return await asyncio.wait_for(self._scrape_with_playwright(url), timeout=30)
        except Exception:
            try:
                return await asyncio.wait_for(self._scrape_with_aiohttp(url), timeout=30)
            except Exception:
                return ""

    @traceable(name="ResearchAgent.run")
    async def run(self, task: ResearchTask) -> list[ResearchResult]:
        return await self._collect_results(task, task.search_queries)

    @traceable(name="ResearchAgent.targeted_research")
    async def targeted_research(
        self,
        task: ResearchTask,
        findings: list[CriticFinding],
    ) -> list[ResearchResult]:
        relevant_findings = [
            finding
            for finding in findings
            if not finding.affected_tasks or task.id in finding.affected_tasks
        ]
        if not relevant_findings:
            return []

        queries = await self._generate_targeted_queries(task, relevant_findings)
        return await self._collect_results(task, queries)

    async def _collect_results(
        self,
        task: ResearchTask,
        queries: list[str],
    ) -> list[ResearchResult]:
        seen_urls: set[str] = set()
        collected: list[ResearchResult] = []

        for query in queries:
            search_results = await self.tavily_search(query)
            for search_result in search_results[:2]:
                if search_result.url in seen_urls:
                    continue
                seen_urls.add(search_result.url)

                scraped_content = await self.scrape_page(search_result.url)
                content = (scraped_content or search_result.content).strip()
                if not content:
                    continue

                relevance_score = await self._score_relevance(
                    task=task,
                    query=query,
                    result=search_result,
                    content=content,
                )
                collected.append(
                    ResearchResult(
                        task_id=task.id,
                        query=query,
                        url=search_result.url,
                        title=search_result.title,
                        content=content[:4000],
                        relevance_score=relevance_score,
                    )
                )

        return sorted(collected, key=lambda item: item.relevance_score, reverse=True)

    async def _generate_targeted_queries(
        self,
        task: ResearchTask,
        findings: list[CriticFinding],
    ) -> list[str]:
        fallback_queries = _fallback_targeted_queries(task, findings)
        try:
            response = await self._call_model(
                [
                    {
                        "role": "user",
                        "content": (
                            "Create focused web search queries to resolve critic "
                            "findings. Include queries that can confirm, refute, "
                            "or contextualize the disputed claims. Return JSON only "
                            "as an array of strings.\n\n"
                            f"Goal: {self.goal}\n"
                            f"Research task: {task.model_dump_json()}\n"
                            "Findings:\n"
                            + "\n".join(
                                f"- {finding.finding_type} ({finding.severity}): "
                                f"{finding.description}"
                                for finding in findings
                            )
                        ),
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )
            parsed = json.loads(response)
            if isinstance(parsed, list):
                queries = [str(item).strip() for item in parsed if str(item).strip()]
                return _unique_queries([*queries, *fallback_queries])[:6]
        except Exception:
            return fallback_queries
        return fallback_queries

    async def _scrape_with_playwright(self, url: str) -> str:
        if async_playwright is None:
            raise RuntimeError("playwright is required for browser scraping.")

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                html = await page.content()
            finally:
                await browser.close()
        return _extract_main_content(html)

    async def _scrape_with_aiohttp(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; NexusResearchBot/0.1; "
                "+https://localhost)"
            )
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status >= 400:
                    return ""
                html = await response.text(errors="ignore")
        return _extract_main_content(html)

    async def _score_relevance(
        self,
        task: ResearchTask,
        query: str,
        result: SearchResult,
        content: str,
    ) -> float:
        try:
            response = await self._call_model(
                [
                    {
                        "role": "user",
                        "content": (
                            "Rate this source from 1 to 10 for relevance.\n\n"
                            f"Goal: {self.goal}\n"
                            f"Research task: {task.description}\n"
                            f"Search query: {query}\n"
                            f"Known project memory:\n{self.memory_context[:1800] or 'None'}\n\n"
                            f"Source title: {result.title}\n"
                            f"Source URL: {result.url}\n"
                            f"Source content:\n{content[:2500]}\n\n"
                            "Prefer sources that add useful new information "
                            "instead of repeating known memory. Return only "
                            "one number from 1 to 10."
                        ),
                    }
                ],
                max_tokens=20,
                temperature=0.0,
            )
            match = re.search(r"\d+(?:\.\d+)?", response)
            if not match:
                return 1.0
            return max(1.0, min(10.0, float(match.group(0))))
        except Exception:
            return 1.0


def _extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for selector in [
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "iframe",
        ".ad",
        ".ads",
        ".advertisement",
        ".cookie",
        ".subscribe",
        '[aria-label*="advertisement"]',
    ]:
        for element in soup.select(selector):
            element.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text[:4000].strip()


def _is_retryable_tavily_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None),
        "status_code",
        None,
    )
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "rate" in message or "timeout" in message or "temporar" in message


def _fallback_targeted_queries(
    task: ResearchTask,
    findings: list[CriticFinding],
) -> list[str]:
    queries: list[str] = []
    for finding in findings:
        issue_terms = finding.description[:140]
        queries.extend(
            [
                f"{task.description} {issue_terms} evidence",
                f"{task.description} {issue_terms} contradiction",
                f"{task.description} {issue_terms} source verification",
            ]
        )
    return _unique_queries(queries)[:6]


def _unique_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique
