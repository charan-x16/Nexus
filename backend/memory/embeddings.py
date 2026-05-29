from typing import Any

from backend.config import settings

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - dependency is installed from requirements.
    AsyncOpenAI = None  # type: ignore[assignment]

_client: Any | None = None


def _get_client() -> Any:
    global _client
    if AsyncOpenAI is None:
        raise RuntimeError("openai is required for embeddings.")
    api_key = (
        settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY is not None
        else None
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    clean_texts = [text for text in texts if text.strip()]
    if not clean_texts:
        return []

    embeddings: list[list[float]] = []
    client = _get_client()
    for batch_start in range(0, len(clean_texts), 100):
        batch = clean_texts[batch_start : batch_start + 100]
        response = await client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        for vector in vectors:
            if len(vector) != 1536:
                raise ValueError(
                    f"Expected 1536-dimensional embedding, got {len(vector)}."
                )
        embeddings.extend(vectors)
    return embeddings


async def embed_single(text: str) -> list[float]:
    embeddings = await embed_texts([text])
    if not embeddings:
        raise ValueError("Cannot embed empty text.")
    return embeddings[0]
