import json
import re
from typing import Any
from uuid import UUID

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.db.connection import execute_query, fetch_rows
from backend.memory.chunker import chunk_text
from backend.memory.embeddings import embed_single, embed_texts
from backend.schemas.workflow import MemoryChunk, ResearchResult


class _MemoryReranker(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=(
                "You score memory chunks for relevance to a query. Return valid "
                "JSON only, mapping chunk numbers to relevance scores from 1 to 10."
            ),
        )

    async def run(self, query: str, chunks: list[MemoryChunk]) -> dict[Any, float]:
        response = await self._call_model(
            [{"role": "user", "content": _rerank_prompt(query, chunks)}],
            max_tokens=500,
            temperature=0.0,
        )
        return _parse_scores(response)


class MemoryStore:
    async def store_research_results(
        self,
        run_id: str,
        project_id: str,
        results: list[ResearchResult],
    ) -> None:
        for result in results:
            chunks = chunk_text(result.content)
            if not chunks:
                continue
            embeddings = await embed_texts(chunks)
            for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "task_id": result.task_id,
                    "query": result.query,
                    "title": result.title,
                    "relevance_score": result.relevance_score,
                }
                await execute_query(
                    """
                    INSERT INTO memory_chunks (
                        project_id,
                        run_id,
                        content,
                        embedding,
                        metadata,
                        chunk_index,
                        source_url
                    )
                    VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6, $7)
                    """,
                    UUID(str(project_id)),
                    UUID(str(run_id)),
                    chunk,
                    _vector_literal(embedding),
                    json.dumps(metadata),
                    chunk_index,
                    result.url,
                )

    async def retrieve(
        self,
        project_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[MemoryChunk]:
        query_embedding = await embed_single(query)
        rows = await fetch_rows(
            """
            SELECT
                content,
                metadata,
                source_url,
                1 - (embedding <=> $2::vector) AS score
            FROM memory_chunks
            WHERE project_id = $1
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            UUID(str(project_id)),
            _vector_literal(query_embedding),
            top_k,
        )
        return [
            MemoryChunk(
                content=row["content"],
                metadata=_decode_metadata(row["metadata"]),
                source_url=row["source_url"],
                score=float(row["score"] or 0),
            )
            for row in rows
        ]

    async def rerank(
        self,
        query: str,
        chunks: list[MemoryChunk],
        top_k: int = 5,
    ) -> list[MemoryChunk]:
        if not chunks:
            return []

        reranker = _MemoryReranker()
        try:
            scores = await reranker.run(query, chunks)
        except Exception:
            return chunks[:top_k]

        rescored: list[MemoryChunk] = []
        for index, chunk in enumerate(chunks, start=1):
            score = scores.get(str(index), scores.get(index, chunk.score))
            rescored.append(
                chunk.model_copy(update={"score": max(0.0, min(10.0, float(score)))})
            )
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _rerank_prompt(query: str, chunks: list[MemoryChunk]) -> str:
    chunk_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_blocks.append(
            f"{index}. Source: {chunk.source_url or 'unknown'}\n"
            f"Content: {chunk.content[:1200]}"
        )
    return (
        "Score each memory chunk for relevance to the query from 1 to 10.\n"
        "Return only JSON like {\"1\": 8, \"2\": 3}.\n\n"
        f"Query: {query}\n\n"
        "Chunks:\n"
        + "\n\n".join(chunk_blocks)
    )


def _parse_scores(response: str) -> dict[Any, float]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        return {}
    return {
        key: float(value)
        for key, value in parsed.items()
        if isinstance(value, int | float | str)
        and str(value).replace(".", "", 1).isdigit()
    }
