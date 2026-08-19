from __future__ import annotations
from typing import Optional
from .client import LLMClient
from .exceptions import LLMProviderError, LLMError, LLMTimeoutError


class LLMService:
    """Thin orchestration layer between the API and LLM client.

    Keeps API concerns separate from provider implementations and is
    simple to mock in tests.
    """

    def __init__(self, client: LLMClient, default_model: Optional[str] = None):
        self._client = client
        self._default_model = default_model

    async def answer_question(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            raise ValueError("Question cannot be empty or whitespace.")

        try:
            return await self._client.generate(q, model=self._default_model)
        except LLMTimeoutError:
            # preserve timeout semantics for callers
            raise
        except LLMProviderError:
            # propagate provider errors as domain error
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError("Unexpected LLM error") from exc


# Factory to create a default LLMService using OpenAIClient lazily
from functools import lru_cache
from .config import get_settings


@lru_cache()
def get_llm_service() -> LLMService:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    # Import here so tests that don't exercise OpenAI don't require the package
    from .providers.openai_client import OpenAIClient

    client = OpenAIClient(api_key=settings.openai_api_key)
    return LLMService(client, default_model=settings.llm_model)
