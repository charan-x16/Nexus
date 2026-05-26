from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx
from langsmith import traceable

from backend.config import settings
from backend.observability.token_tracker import TokenUsage, track_token_usage
from backend.schemas.workflow import WorkflowState


class BaseAgent(ABC):
    def __init__(self, model_name: str, system_prompt: str) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt

    @abstractmethod
    async def run(self, state: WorkflowState) -> WorkflowState:
        raise NotImplementedError

    @traceable(name="BaseAgent._call_model")
    async def _call_model(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> str:
        api_key = (
            settings.OPENROUTER_API_KEY.get_secret_value()
            if settings.OPENROUTER_API_KEY is not None
            else None
        )
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to call OpenRouter.")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *list(messages),
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.OPENROUTER_SITE_URL,
            "X-Title": settings.OPENROUTER_APP_NAME,
        }

        async with httpx.AsyncClient(
            base_url=settings.OPENROUTER_BASE_URL,
            timeout=90.0,
        ) as client:
            response = await client.post(
                "/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        response_data = response.json()
        usage = response_data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        track_token_usage(
            TokenUsage(
                agent_name=self.__class__.__name__,
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

        choices = response_data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter response did not include any choices.")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            ]
            return "\n".join(text_parts).strip()
        raise RuntimeError("OpenRouter response content was not text.")
