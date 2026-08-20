import asyncio
import pytest

from llm.service import LLMService
from llm.exceptions import LLMProviderError, LLMTimeoutError


class _GoodFakeClient:
    async def generate(self, prompt: str, model=None) -> str:
        return f"good:{prompt}"


class _ProviderErrorClient:
    async def generate(self, prompt: str, model=None) -> str:
        raise LLMProviderError("provider down")


class _TimeoutClient:
    async def generate(self, prompt: str, model=None) -> str:
        raise LLMTimeoutError("timed out")


def test_service_success() -> None:
    svc = LLMService(_GoodFakeClient(), default_model="m")
    out = asyncio.run(svc.answer_question("hello"))
    assert out == "good:hello"


def test_service_provider_error_propagates() -> None:
    svc = LLMService(_ProviderErrorClient())
    with pytest.raises(LLMProviderError):
        asyncio.run(svc.answer_question("hello"))


def test_service_timeout_propagates() -> None:
    svc = LLMService(_TimeoutClient())
    with pytest.raises(LLMTimeoutError):
        asyncio.run(svc.answer_question("hello"))


def test_service_empty_question_raises() -> None:
    svc = LLMService(_GoodFakeClient())
    with pytest.raises(ValueError):
        asyncio.run(svc.answer_question(""))
