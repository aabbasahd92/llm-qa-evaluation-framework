from __future__ import annotations
from typing import Optional
import asyncio
from ..client import LLMClient
from ..exceptions import LLMProviderError, LLMTimeoutError


class OpenAIClient(LLMClient):
    """Concrete LLM client using the OpenAI Python SDK (Responses API).

    The import of the `openai` library is done inside methods so the
    module can be imported in environments where the SDK is not
    installed (tests that mock the service will not require the SDK).
    """

    def __init__(self, api_key: str, timeout: float = 15.0):
        self._api_key = api_key
        self._timeout = timeout

    async def generate(self, prompt: str, model: Optional[str] = None) -> str:
        # Lazy import to avoid requiring openai at module import time
        try:
            import openai
        except Exception as exc:
            raise LLMProviderError("OpenAI SDK not installed") from exc

        # Configure the client
        openai.api_key = self._api_key

        # Build call arguments for the Responses API
        call_kwargs = {
            "model": model or "gpt-4o-mini",
            "input": prompt,
        }

        # The official SDK is synchronous for some versions; run in thread
        loop = asyncio.get_running_loop()

        def _call():
            try:
                # Use Responses API
                return openai.responses.create(**call_kwargs)
            except Exception as e:
                raise

        try:
            resp = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=self._timeout)
        except asyncio.TimeoutError as te:
            raise LLMTimeoutError("OpenAI request timed out") from te
        except Exception as e:
            # Map provider exception to domain exception
            raise LLMProviderError("OpenAI provider error") from e

        # Parse response — keep it simple and defensive
        try:
            # SDK response shape: resp.output[0].content[0].text or resp.output_text
            if hasattr(resp, "output_text") and resp.output_text:
                return str(resp.output_text)

            # Try common nested structure
            output = getattr(resp, "output", None)
            if output and isinstance(output, list) and output:
                first = output[0]
                # content may be list of dicts
                content = first.get("content") if isinstance(first, dict) else None
                if content and isinstance(content, list) and content:
                    # concatenate text parts
                    parts = [c.get("text") or "" for c in content if isinstance(c, dict)]
                    return "".join(parts).strip()

            # Fallback to stringified response
            return str(resp)
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMProviderError("Failed to parse OpenAI response") from exc
