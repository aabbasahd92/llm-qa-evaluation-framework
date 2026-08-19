from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract LLM client interface.

    Implementations should provide an async `generate` method that
    accepts a prompt and optional model name and returns the generated
    text as a string.
    """

    @abstractmethod
    async def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text for the given prompt using an optional model name.

        Should raise exceptions defined in llm.exceptions on failure.
        """
        raise NotImplementedError
