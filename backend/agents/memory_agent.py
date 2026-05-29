from typing import Any
from uuid import UUID

from langsmith import traceable

from backend.agents.base import BaseAgent
from backend.config import settings
from backend.db.connection import execute_query
from backend.memory.embeddings import embed_single
from backend.memory.store import MemoryStore, _vector_literal
from backend.schemas.workflow import WorkflowState

MEMORY_SYSTEM_PROMPT = (
    "You are an expert at retrieving and summarising relevant past context. "
    "Be concise, faithful to the provided content, and avoid adding facts."
)


class MemoryAgent(BaseAgent):
    def __init__(self, store: MemoryStore | None = None) -> None:
        super().__init__(
            model_name=settings.OPENROUTER_MODEL,
            system_prompt=MEMORY_SYSTEM_PROMPT,
        )
        self.store = store or MemoryStore()

    @traceable(name="MemoryAgent.retrieve_context")
    async def retrieve_context(self, project_id: str, query: str) -> str:
        chunks = await self.store.retrieve(project_id=project_id, query=query, top_k=10)
        reranked = await self.store.rerank(query=query, chunks=chunks, top_k=5)
        if not reranked:
            return ""

        lines = ["Relevant memory context:"]
        for index, chunk in enumerate(reranked, start=1):
            source = chunk.source_url or "stored memory"
            lines.append(
                "\n".join(
                    [
                        f"{index}. Source: {source}",
                        f"Score: {chunk.score}",
                        f"Content: {chunk.content[:1200]}",
                    ]
                )
            )
        return "\n\n".join(lines)

    @traceable(name="MemoryAgent.summarise_and_store")
    async def summarise_and_store(
        self,
        project_id: str,
        run_id: str,
        content: str,
        label: str,
    ) -> None:
        if not content.strip():
            return
        summary = await self._call_model(
            [
                {
                    "role": "user",
                    "content": (
                        "Produce a faithful 3-sentence summary of this content. "
                        "Return only the summary.\n\n"
                        f"Label: {label}\n"
                        f"Run ID: {run_id}\n\n"
                        f"Content:\n{content[:6000]}"
                    ),
                }
            ],
            max_tokens=350,
            temperature=0.1,
        )
        summary_text = f"{label}: {summary.strip()}"
        embedding = await embed_single(summary_text)
        await execute_query(
            """
            INSERT INTO project_summaries (project_id, summary, embedding)
            VALUES ($1, $2, $3::vector)
            """,
            UUID(str(project_id)),
            summary_text,
            _vector_literal(embedding),
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        project_id = state.get("project_id")
        query = state.get("goal", "")
        updated_state: WorkflowState = dict(state)
        updated_state["memory_context"] = (
            await self.retrieve_context(project_id=project_id, query=query)
            if project_id and query
            else ""
        )
        return updated_state
