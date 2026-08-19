from __future__ import annotations
from functools import lru_cache
import os
from typing import Optional

# Prefer pydantic-settings when available, fall back to a lightweight dataclass-based shim
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        """Configuration for LLM integration using pydantic-settings.

        Environment variables:
          - OPENAI_API_KEY
          - LLM_MODEL
        """

        openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
        llm_model: Optional[str] = Field(None, env="LLM_MODEL")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

except Exception:  # pragma: no cover - fallback path for environments lacking pydantic-settings
    from dataclasses import dataclass

    @dataclass
    class Settings:
        openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
        llm_model: Optional[str] = os.environ.get("LLM_MODEL")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
